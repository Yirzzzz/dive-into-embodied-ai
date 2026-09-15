"""Collect or evaluate LIBERO-Long Task 0 in the separate simulation environment.

Requires only LIBERO + openpi-client, not the OpenPI training dependencies.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np


def initial_state_ids(purpose, count):
    start, available = (0, 30) if purpose == "collect" else (30, 20)
    if not 1 <= count <= available:
        raise ValueError(f"{purpose} supports 1..{available} episodes")
    return list(range(start, start + count))


def configure_libero():
    import yaml

    root = pathlib.Path(__file__).resolve().parents[2]
    directory = pathlib.Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(root / "data/recap/libero-config")))
    path = directory / "config.yaml"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        benchmark_root = root / "third_party/libero/libero/libero"
        path.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(benchmark_root),
                    "bddl_files": str(benchmark_root / "bddl_files"),
                    "init_states": str(benchmark_root / "init_files"),
                    "assets": str(benchmark_root / "assets"),
                    "datasets": str(root / "data/recap/raw"),
                }
            )
        )


def observation_fields(obs):
    from examples.libero.main import _quat2axisangle

    # Raw simulator images need a 180-degree rotation. Store at 256x256,
    # matching downloaded demonstrations; the policy transforms resize to 224.
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])
        ).astype(np.float32),
    }


def run_episode(env, client, initial_state, task, replan_steps=5):
    from examples.libero.main import LIBERO_DUMMY_ACTION
    from openpi_client import image_tools

    env.reset()
    client.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done or env.check_success():
            raise RuntimeError("Episode ended during settling; no training trajectory was recorded")
    plan = collections.deque()
    frames = []
    success = False
    for _ in range(520):
        frame = observation_fields(obs)
        if not plan:
            request = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(frame["image"], 224, 224)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(frame["wrist_image"], 224, 224)
                ),
                "observation/state": frame["state"],
                "prompt": task,
            }
            actions = np.asarray(client.infer(request)["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or len(actions) < replan_steps
                or not np.isfinite(actions).all()
            ):
                raise ValueError("Policy must return finite [horizon, 7] actions with horizon >= replan_steps")
            plan.extend(actions[:replan_steps])
        frame["actions"] = plan.popleft()
        # Save the observation BEFORE its executed action, including the final
        # action. Do not store action chunks as if every predicted action ran.
        obs, _, done, _ = env.step(frame["actions"].tolist())
        frames.append(frame)
        success = bool(env.check_success())
        if done or success:
            break
    return frames, success


def write_results(output, results):
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2, default=str) + "\n")
    temporary.replace(output / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", choices=["collect", "eval"])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--policy-label", required=True, help="Checkpoint path or other unambiguous checkpoint identifier"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-episodes", type=int, help="Default: 30 for collect, 20 for eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    args = parser.parse_args()
    count = args.num_episodes if args.num_episodes is not None else (30 if args.purpose == "collect" else 20)
    ids = initial_state_ids(args.purpose, count)
    if not 1 <= args.replan_steps <= 10:
        parser.error("replan-steps must be in 1..10 for this configuration")
    if args.output.exists():
        parser.error(f"Output exists: {args.output}; choose a new directory")
    configure_libero()
    import imageio
    from examples.libero.main import _get_libero_env
    from libero.libero import benchmark
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    states = suite.get_task_init_states(0)
    if max(ids) >= len(states):
        raise ValueError(f"This LIBERO checkout only has {len(states)} initial states")
    client = WebsocketClientPolicy(args.host, args.port)
    args.output.mkdir(parents=True)
    results = {
        "purpose": args.purpose,
        "complete": False,
        "suite": "libero_10",
        "task_id": 0,
        "policy_label": args.policy_label,
        "server_metadata": client.get_server_metadata(),
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "max_steps": 520,
        "settle_steps": 10,
        "initial_state_ids": ids,
        "episodes": [],
    }
    write_results(args.output, results)
    env, description = _get_libero_env(task, 256, args.seed)
    try:
        for state_id in ids:
            frames, success = run_episode(env, client, states[state_id], description, args.replan_steps)
            arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
            stem = f"state_{state_id:03d}_{'success' if success else 'failure'}"
            record = {"initial_state_id": state_id, "success": success, "steps": len(frames), "video": stem + ".mp4"}
            if args.purpose == "collect":
                record["trajectory"] = stem + ".npz"
                np.savez_compressed(
                    args.output / record["trajectory"],
                    **arrays,
                    success=np.bool_(success),
                    task=description,
                    task_id=0,
                    initial_state_id=state_id,
                )
            imageio.mimwrite(args.output / record["video"], arrays["image"], fps=10)
            results["episodes"].append(record)
            write_results(args.output, results)
            print(f"state={state_id}, success={success}, steps={len(frames)}", flush=True)
    finally:
        env.close()
    results["complete"] = True
    results["success_rate"] = sum(row["success"] for row in results["episodes"]) / len(ids)
    write_results(args.output, results)
    print(f"Success rate: {results['success_rate']:.1%}; results: {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
