"""Value targets and n-step advantage labels for a COPY of a LeRobot v2 dataset."""

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def episode_columns(path):
    table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
    episode, frame, task = (np.asarray(table[key]) for key in table.column_names)
    if len(frame) == 0 or len(np.unique(episode)) != 1 or len(np.unique(task)) != 1:
        raise ValueError(f"Expected one nonempty, single-task episode per parquet: {path}")
    if not np.array_equal(frame, np.arange(len(frame))):
        raise ValueError(f"Frames must be ordered and contiguous from zero: {path}")
    return int(episode[0]), int(task[0]), frame


def value_targets(length, success, failure_penalty=300.0, return_scale=820.0):
    if length < 1 or not isinstance(success, bool):
        raise ValueError("Expected a positive episode length and an explicit boolean success label")
    if not np.isfinite([failure_penalty, return_scale]).all() or failure_penalty < 0 or return_scale <= 0:
        raise ValueError("failure_penalty must be nonnegative; return_scale must be positive")
    returns = -np.arange(length - 1, -1, -1, dtype=np.float32)
    if not success:
        returns -= failure_penalty
    if np.min(returns) < -return_scale:
        raise ValueError("return_scale is too small; use the same larger scale for all datasets")
    return returns / return_scale


def n_step_advantages(targets, values, n_step=10):
    """Gamma=1, one complete episode. Terminal bootstrap is exactly zero.

    G_t is the normalized Monte Carlo return. r_t = G_t - G_(t+1),
    and the final reward is G_(T-1), retaining the failure penalty.
    """
    targets, values = np.asarray(targets), np.asarray(values)
    if n_step < 1 or targets.ndim != 1 or targets.shape != values.shape or not len(targets):
        raise ValueError("Expected matching nonempty 1-D arrays and n_step >= 1")
    if not np.isfinite(targets).all() or not np.isfinite(values).all():
        raise ValueError("Targets and predictions must be finite")
    end = np.minimum(np.arange(len(targets)) + n_step, len(targets))
    # Telescoping sum of rewards; no bootstrap crosses an episode boundary.
    return targets - np.append(targets, 0)[end] + np.append(values, 0)[end] - values


def positive_labels(advantages, tasks, positive_ratio=0.3):
    if not 0 < positive_ratio < 1:
        raise ValueError("positive_ratio must be strictly between 0 and 1")
    indicators = np.zeros(len(advantages), dtype=np.int64)
    thresholds = {}
    for task in np.unique(tasks):
        mask = tasks == task
        threshold = float(np.quantile(advantages[mask], 1 - positive_ratio))
        indicators[mask] = advantages[mask] >= threshold
        thresholds[int(task)] = threshold
    return indicators, thresholds


def write_columns(path, columns):
    table = pq.read_table(path)
    for name, values in columns.items():
        array = pa.array(values)
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, array)
        else:
            table = table.append_column(name, array)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def update_features(root, columns):
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    for name in columns:
        info["features"][name] = {
            "dtype": "int64" if name == "is_positive" else "float32",
            "shape": [1],
            "names": None,
        }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2) + "\n")
    temporary.replace(path)


def predict_values(root, checkpoint, batch_size, max_frames=None):
    import jax
    import jax.numpy as jnp
    from examples.recap.config import get_configs
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints

    cfg = next(c for c in get_configs() if c.name == "pi05_recap_value")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    # Checkpoint assets are authoritative, even if current local stats changed.
    data_cfg = dataclasses.replace(
        data_cfg,
        norm_stats=checkpoints.load_norm_stats(checkpoint / "assets", data_cfg.asset_id),
    )
    model = cfg.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    predict = nnx_utils.module_jit(model.predict_value)
    # No action chunk is needed for value inference; use a single dummy action
    # sequence, still keeping the normal observation preprocessing.
    dataset = LeRobotDataset(cfg.data.repo_id, root=root)
    prompt = transforms.PromptFromLeRobotTask(dataset.meta.tasks)
    transform = transforms.compose(
        [
            prompt,
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ]
    )
    predictions = {}
    indices = np.arange(len(dataset))
    if max_frames is not None and max_frames < len(indices):
        # Same evenly spaced frames across checkpoints, covering the entire split.
        indices = np.linspace(0, len(dataset) - 1, max_frames, dtype=int)
    for start in range(0, len(indices), batch_size):
        samples = [dataset[int(i)] for i in indices[start : start + batch_size]]
        keys = [(int(s["episode_index"].item()), int(s["frame_index"].item())) for s in samples]
        processed = []
        for sample in samples:
            sample["actions"] = np.asarray(sample["actions"])[None, :]
            processed.append(transform(sample))
        padded = processed + [processed[-1]] * (batch_size - len(processed))
        batch = jax.tree.map(lambda *xs: np.stack(xs), *padded)
        values = np.asarray(predict(model_lib.Observation.from_dict(batch)))[: len(samples)]
        for key, value in zip(keys, values, strict=True):
            if key in predictions:
                raise ValueError(f"Duplicate frame in dataset: {key}")
            predictions[key] = float(value)
        if start % (100 * batch_size) == 0 or start + batch_size >= len(indices):
            print(f"Value inference: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    return predictions


def evaluate_values(paths, predictions):
    targets = {}
    for path in paths:
        ep, _, frames = episode_columns(path)
        values = np.asarray(pq.read_table(path, columns=["value_target"])["value_target"]).reshape(-1)
        targets.update({(ep, int(frame)): float(value) for frame, value in zip(frames, values, strict=True)})
    if not predictions or not predictions.keys() <= targets.keys():
        raise ValueError("Evaluation predictions must match frames in the held-out dataset")
    errors = np.asarray([value - targets[key] for key, value in predictions.items()])
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite evaluation targets or predictions")
    return {"frames": len(errors), "mse": float(np.mean(errors**2)), "mae": float(np.mean(np.abs(errors)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["targets", "advantages", "evaluate"])
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--positive-ratio", type=float, default=0.3)
    parser.add_argument("--failure-penalty", type=float, default=300.0)
    parser.add_argument("--return-scale", type=float, default=820.0)
    parser.add_argument("--max-frames", type=int, default=10000, help="Fixed validation subset; evaluate only")
    parser.add_argument("--output", type=pathlib.Path, help="Value evaluation JSON; evaluate only")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        parser.error("Dataset preparation is incomplete")
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths or not (root / "meta/info.json").is_file():
        parser.error("Expected a LeRobot v2 dataset with data/**/*.parquet and meta/info.json")
    if args.batch_size < 1 or args.n_step < 1 or not 0 < args.positive_ratio < 1:
        parser.error("batch-size and n-step must be positive; positive-ratio must be in (0, 1)")
    episodes = [episode_columns(path) for path in paths]
    if len({ep for ep, _, _ in episodes}) != len(episodes):
        parser.error("An episode must occupy exactly one parquet file (LeRobot v2 layout)")
    source_path = root / "recap_source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    if args.stage == "advantages" and source.get("split") == "eval":
        parser.error("The eval split must not contribute to advantage thresholds or ACP training")
    if args.stage == "evaluate":
        if source.get("split") != "eval" or args.checkpoint is None or args.output is None or args.max_frames < 1:
            parser.error("evaluate requires a prepared eval split, --checkpoint, --output, and positive --max-frames")
        predictions = predict_values(root, args.checkpoint.expanduser().resolve(), args.batch_size, args.max_frames)
        result = evaluate_values(paths, predictions)
        result.update(checkpoint=str(args.checkpoint.resolve()), dataset_root=str(root))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    pending = []
    if args.stage == "targets":
        metadata = {
            row["episode_index"]: row
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        for ep, _, frames in episodes:
            meta = metadata[ep]
            if meta["length"] != len(frames):
                raise ValueError(f"Episode {ep}: metadata length disagrees with parquet")
            pending.append(
                {
                    "value_target": value_targets(
                        len(frames),
                        meta.get("success"),
                        args.failure_penalty,
                        args.return_scale,
                    )
                }
            )
    else:
        if args.checkpoint is None:
            parser.error("advantages requires --checkpoint pointing to a value checkpoint step directory")
        # Validate every label before loading the large model or writing any file.
        targets = [np.asarray(pq.read_table(p, columns=["value_target"])["value_target"]).reshape(-1) for p in paths]
        for target, (_, _, frames) in zip(targets, episodes, strict=True):
            if target.shape != frames.shape or not np.isfinite(target).all() or np.any((target < -1) | (target > 0)):
                raise ValueError("Run targets first, or supply finite value_target scalars in [-1, 0]")
        predictions = predict_values(root, args.checkpoint.expanduser().resolve(), args.batch_size)
        if len(predictions) != sum(len(frames) for _, _, frames in episodes):
            raise ValueError("LeRobot loader and parquet files have different frame counts")
        task_indices = []
        for (ep, task, frames), target in zip(episodes, targets, strict=True):
            values = np.asarray([predictions[(ep, int(frame))] for frame in frames], dtype=np.float32)
            pending.append(
                {
                    "predicted_value": values,
                    "advantage": n_step_advantages(target, values, args.n_step).astype(np.float32),
                }
            )
            task_indices.extend([task] * len(frames))
        labels, thresholds = positive_labels(
            np.concatenate([p["advantage"] for p in pending]),
            np.asarray(task_indices),
            args.positive_ratio,
        )
        start = 0
        for item in pending:
            end = start + len(item["advantage"])
            item["is_positive"] = labels[start:end]
            start = end
        print(f"Per-task thresholds: {thresholds}; actual positive fraction: {labels.mean():.3f}")
    # This command intentionally updates the dataset copy supplied by the user.
    for path, columns in zip(paths, pending, strict=True):
        write_columns(path, columns)
    update_features(root, pending[0])
    print(f"Annotated {len(paths)} episodes in {root}")


if __name__ == "__main__":
    main()
