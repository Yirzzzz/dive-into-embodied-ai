# π₀.₅ + RECAP：LIBERO 长时程任务全流程复现

**验证状态：当前是待 GPU 全链路验收的最小实现。** 已验证真实数据小样本读取、stats，以及缩小模型的训练与 checkpoint 回读；尚未完成真实 π₀.₅ 的 GPU 训练和 LIBERO 仿真闭环，不能据此保证以下完整流程已跑通。

从官方 OpenPI 和 RLinf 发布的数据开始，在自己的 GPU 机器上完成一轮：

**下载数据 → 准备 LeRobot 目录 → 计算 norm stats → SFT → 计算回报 → 价值训练与验证 → 优势标注 → ACP 微调 → 仿真对照评估。**

使用 [RLinf/RECAP-Libero10-Task0-48succ-Data](https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data/tree/75b382d2c066bcedd8b030285b45c856913f1497)。
成功和失败轨迹已经包含在数据集中，第一轮离线训练不需要自行采集。
最终评估目标是 LIBERO-Long（代码名 `libero_10`）的 Task 0：将 alphabet soup 和 tomato sauce 都放进篮子。
本项目只提供需要接入 OpenPI 的增量代码，不包含 OpenPI 本体。必须先将
`codes/practices/vla/recap/` 复制为 OpenPI 的 `examples/recap/`；以下命令全部在 OpenPI 根目录执行。

## 1. 机器与环境

本版使用 **Ubuntu 22.04 + NVIDIA GPU + JAX 全参数训练**。
上游参考显存：推理大于 8 GB，全参数微调大于 70 GB；推荐 80 GB GPU，或通过 FSDP 分摊到多卡。
这些是上游参考值，本例价值训练的峰值尚未实测。24 GB 单卡和 Mac 不是本版完整训练的目标环境。
[OpenPI 硬件说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#requirements)

原始数据包含大量双视角视频；另外还需为基础权重、checkpoint 和训练表副本预留磁盘。
准备脚本使用视频软链接，避免再复制一遍视频；训练期间请保留下载目录。
先安装 NVIDIA 驱动、Git、Git LFS 和 uv。假设两个仓库放在同级目录：

```bash
# 若已有教学仓库，跳过第一行。
git clone https://github.com/Yirzzzz/dive-into-embodied-ai.git
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
cp -R ../dive-into-embodied-ai/codes/practices/vla/recap examples/recap
test -f examples/recap/run.py && echo "RECAP code is ready"

uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
nvidia-smi
uv run python -c 'import jax; print(jax.devices())'
```

最后一行应列出 CUDA GPU。固定 OpenPI 提交同时固定 LIBERO 子模块版本，入口会检查提交号。
训练与仿真使用独立 Python 环境，避免两套依赖的版本冲突。

## 2. 下载数据与确认划分

每个训练终端先设置以下变量：

```bash
# 当前目录必须是 OpenPI 根目录，并且已完成第 1 节的代码复制。
test -f examples/recap/run.py || { echo "缺少 examples/recap，请先复制 RECAP 代码"; exit 1; }
cp -R /data1/evan/dive-into-embodied-ai/codes/practices/vla/recap \
  /data1/evan/openpi/examples/recap

export HF_LEROBOT_HOME="$PWD/data/lerobot"
export RECAP_SFT_REPO_ID="local/libero10_fewshot_sft"
export RECAP_REPO_ID="local/libero10_task0_train"
export RECAP_EVAL_REPO_ID="local/libero10_task0_eval"

# huggingface.co 无法访问时，在启动 Python 前指定镜像：
export HF_ENDPOINT="https://hf-mirror.com"

uv run python examples/recap/run.py data download
```

可直接访问 Hugging Face 时不设置 `HF_ENDPOINT`。脚本会显示实际使用的 endpoint，并在下载后逐个检查三个划分的 metadata、Parquet 和双视角视频；网络失败时不会再把空目录或部分目录报告为下载成功。保留未完成目录后重跑即可断点续传。

下载固定 revision `75b382d2c066bcedd8b030285b45c856913f1497` 的三个完整子目录。
输出位于 `data/recap/rlinf/`。下面的数量根据该版本实际元数据核对：

| 发布目录 | 内容 | 本例用途 |
| --- | --- | --- |
| `libero10_task0_sft` | 30 条成功示教，覆盖 10 个任务，每任务 3 条 | few-shot SFT 与 norm stats |
| `libero10_task0_train` | Task 0 的 4,096 条 rollout，1,999 成功、2,097 失败 | 价值训练、优势标注、ACP |
| `libero10_task0_eval` | Task 0 的 64 条 rollout，27 成功、37 失败 | 价值模型验证与 checkpoint 选择 |

**SFT 目录名不代表内部只有 Task 0。** 目标任务在它的 `tasks.jsonl` 中编号为 5，对应 episode 15、16、17；这个数据集编号与仿真套件的 Task 0 编号不同。
本例保留全部 30 条示教，沿用发布方的 few-shot 数据范围，通过任务文本区分任务。
为了保持 OpenPI 接入简单，价值模型和 ACP 使用 `train`；本版未额外实现示教与 rollout 的平衡混采。

`eval` 是同一采集策略的留出轨迹，用于价值监督验证，**不是训练后策略的仿真成功率**。
数据集名称中的 `48succ`、发布方的 few-shot 基线及提升幅度均属于原实验设置；本地训练结果需要重新测量。
[RLinf 原教程与实验说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html#dataset)

## 3. 准备本地 LeRobot 目录

```bash
uv run python examples/recap/run.py data prepare --split sft
uv run python examples/recap/run.py data prepare --split train
uv run python examples/recap/run.py data prepare --split eval
```

数据已经是 LeRobot 格式，无需 TensorFlow/RLDS 转换。脚本复制训练表、链接视频，统一为固定 OpenPI 所用的 LeRobot v2.1 元数据布局。
它检查逐回合长度、任务、成功标记和视频是否齐全，修正发布元数据中遗留的 split 范围、视频数、chunk 数，并重建连续帧索引。
原始图像、动作、状态和各划分的 fps 保持不变；不重复翻转图像或转换增量动作。

**输出：** `$HF_LEROBOT_HOME` 下的三个 `local/...` 目录。
训练副本将 `is_success` 统一为 episode 元数据中的 `success`；公开数据已有的 return/reward 留在原始目录，本例会自行计算监督目标。
已有输出不会被覆盖。中断的输出带 `INCOMPLETE` 标记，排查后可用 `--repo-id local/新名字` 重做，并同步修改环境变量。

## 4. 自己计算统计量，训练 SFT

```bash
uv run python examples/recap/run.py stats --config-name pi05_recap_sft

# 可先用独立实验名跑 2 步，检查显存和数据管道；不能用于结果评估。
uv run python examples/recap/run.py train pi05_recap_sft \
  --exp-name smoke --num-train-steps 2 --batch-size 8 --no-wandb-enabled

# 从 pi05_base 初始化，本例默认 45,000 步。
uv run python examples/recap/run.py train pi05_recap_sft \
  --exp-name baseline --batch-size 8 --fsdp-devices 1 --no-wandb-enabled

SFT_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_sft baseline)"
```

**统计输出：** `assets/pi05_recap_sft/local/libero10_fewshot_sft/norm_stats.json`。
**训练输出：** `checkpoints/pi05_recap_sft/baseline/`。`checkpoint` 命令选择含 `params/` 的最大 step。
基础权重由 OpenPI 下载并缓存，无需私有 checkpoint。SFT、value、ACP 共享这份只由 SFT 数据计算的统计量，验证集不参与统计。
三个配置沿用 π₀.₅ 默认的离散状态编码，将归一化后的机器人状态写入输入 token。
多卡训练同时调整 `--batch-size` 和 `--fsdp-devices`；续训使用同一实验名加 `--resume`。

这里重新训练出的 SFT 不一定就是发布方采集 rollout 时使用的策略，也不保证得到 48.8% 成功率。
它作为本地初始化和对照；现成 rollout 始终来自发布方的采集策略。

## 5. 计算训练集和验证集回报

```bash
uv run python examples/recap/run.py annotate targets \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_REPO_ID"
uv run python examples/recap/run.py annotate targets \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_EVAL_REPO_ID"
```

两份副本分别新增 `value_target`。采用 γ=1：普通步奖励 -1，成功末步 0，失败末步 -300，回报统一除以 820，得到 `[-1, 0]` 的监督值。
这个固定尺度在训练、验证和优势计算中一致使用。两份数据保持分离，验证回报只用于衡量预测误差。

## 6. 训练价值模型，监控验证误差

从 SFT 骨干初始化，新增价值头随机初始化。分段训练到 10,000、20,000、30,000 步，每段结束在相同的验证帧上评估：

```bash
resume_flag=""
for steps in 10000 20000 30000; do
  RECAP_INIT_PARAMS="$SFT_CKPT/params" \
  uv run python examples/recap/run.py train pi05_recap_value \
    --exp-name value --num-train-steps "$steps" \
    --batch-size 8 --fsdp-devices 1 --save-interval 10000 --keep-period 1 \
    --no-wandb-enabled $resume_flag || break

  VALUE_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_value value)" || break
  uv run python examples/recap/run.py annotate evaluate \
    --dataset-root "$HF_LEROBOT_HOME/$RECAP_EVAL_REPO_ID" \
    --checkpoint "$VALUE_CKPT" --batch-size 8 --max-frames 10000 \
    --output "data/recap/value_eval/${steps}.json" || break
  resume_flag="--resume"
done
```

`--num-train-steps` 是累计步数，后两段恢复模型和优化器；学习率计划仍固定为 30,000 步。
`--save-interval 10000 --keep-period 1` 配合上述分段，只保存并保留三个段末 checkpoint，便于比较和选择。
价值模型更新 SigLIP、VLM 和价值头，冻结动作分支，输出 201 桶分布，以 two-hot 交叉熵拟合回报。
验证不更新参数，默认固定抽取均匀覆盖验证集的 10,000 帧，报告标量价值的 MSE、MAE。
训练损失下降而验证误差上升时，应优先检查过拟合；验证误差不能替代策略成功率。

## 7. 选择价值 checkpoint，计算优势

使用验证 MSE 最小的 checkpoint：

```bash
VALUE_CKPT="$(python3 - <<'PY'
import json
from pathlib import Path
runs = [json.loads(p.read_text()) for p in Path('data/recap/value_eval').glob('*.json')]
runs = [r for r in runs if (Path(r['checkpoint']) / 'params').is_dir()]
if not runs:
    raise SystemExit('No evaluated checkpoint remains on disk')
print(min(runs, key=lambda r: r['mse'])['checkpoint'])
PY
)"

uv run python examples/recap/run.py annotate advantages \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_REPO_ID" \
  --checkpoint "$VALUE_CKPT" --batch-size 8 --n-step 10 --positive-ratio 0.3
```

优势采用 `A_t = Σ r + V(s_{t+10}) - V(s_t)`，终止时 bootstrap 为 0，保留失败惩罚且不跨 episode。
只在 `train` 内按 70% 分位数生成正/负标签；并列值可能使正样本超过 30%。
输出新增 `predicted_value`、`advantage`、`is_positive`。验证集不参与优势阈值计算，也不进入 ACP。

## 8. ACP 微调

```bash
RECAP_INIT_PARAMS="$SFT_CKPT/params" \
uv run python examples/recap/run.py train pi05_recap_acp \
  --exp-name acp --batch-size 8 --fsdp-devices 1 --no-wandb-enabled

ACP_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_acp acp)"
```

ACP 默认训练 45,000 步，任务文本添加 `Advantage: positive/negative`，训练时以 0.3 概率去掉条件；推理始终使用 positive。

## 9. LIBERO 仿真对照评估

安装独立 Python 3.8 仿真环境，依赖沿用固定提交的 [OpenPI LIBERO 示例](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero)：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1-mesa-glx libosmesa6 libglfw3 libglib2.0-0
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python \
  -e packages/openpi-client -e third_party/libero
```

终端 A 启动 SFT 服务；新终端需要重新设置第 2 节变量，并查询 checkpoint：

```bash
SFT_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_sft baseline)"
uv run python examples/recap/run.py serve --port 8000 policy:checkpoint \
  --policy.config pi05_recap_sft --policy.dir "$SFT_CKPT"
```

终端 B 执行：

```bash
SFT_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_sft baseline)"
MUJOCO_GL=egl examples/libero/.venv/bin/python examples/recap/run.py rollout eval \
  --output data/recap/eval_sft --policy-label "$SFT_CKPT"
```

评估结束，在终端 A 按 Ctrl+C 停止 SFT 服务，再启动 ACP 服务：

```bash
ACP_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_acp acp)"
uv run python examples/recap/run.py serve --port 8000 policy:checkpoint \
  --policy.config pi05_recap_acp --policy.dir "$ACP_CKPT"
```

终端 B 使用相同协议评估 ACP：

```bash
ACP_CKPT="$(uv run python examples/recap/run.py checkpoint pi05_recap_acp acp)"
MUJOCO_GL=egl examples/libero/.venv/bin/python examples/recap/run.py rollout eval \
  --output data/recap/eval_acp --policy-label "$ACP_CKPT"

python3 - <<'PY'
import json
from pathlib import Path
for name in ('sft', 'acp'):
    result = json.loads(Path(f'data/recap/eval_{name}/results.json').read_text())
    assert result['complete'], f'{name} evaluation is incomplete'
    print(name, result['success_rate'], result['initial_state_ids'])
PY
```

默认评估初始状态 30..49，共 20 回合；每回合最多 520 个动作，先等待 10 步，每 5 步重新规划。
两模型使用相同初始状态、环境 seed 和执行协议。这些状态不保证与发布方离线数据的初始状态无重叠。
脚本自动配置 LIBERO 资源路径，输出视频及 `results.json`；异常会中止，不当作普通失败计入完整报告。

默认 20 回合是教程级验证。要声称算法增益，还需要多随机种子，以及相同新增数据、更新步数下继续普通 SFT 的对照。
不能把原 SFT 与 ACP 的全部差异归因于 RECAP。

## 实现与验证边界

数据来自 RLinf；算法代码提取自 [Yirzzzz/pi0.6](https://github.com/Yirzzzz/pi0.6)，基础模型、训练器、归一化和服务复用官方 OpenPI。
这是使用该数据集的 OpenPI 最小接入版本。RLinf 原版的价值模型架构、混采、CFG 训练与推理实现并未完整移植，因此不宣称等价复现其 48.8% → 66.5% 数值。
本版也保留上游动作损失，未移植个人分支的 7 维 action loss mask。

```bash
uv run python -m pytest examples/recap/test_recap.py -q
```

本地 29 项 CPU 测试通过，覆盖数据准备、回报与优势、验证集隔离，以及完整模型在上游训练器中的初始化、反向传播、优化器和 EMA 的形状检查。另用三个划分各一条真实轨迹检查了视频/动作读取，并通过上游入口在 SFT 小样本上计算了 norm stats。

另外使用随机初始化的缩小版骨干，实际验证了 SFT checkpoint → 价值模型初始化 → CPU 梯度更新 → 保存 → 回读预测；这不等于真实 π₀.₅ 权重验证，也不能用来判断显存或任务成功率。完整数据集处理、正式 GPU 训练及真实 LIBERO 评估尚未完成，没有可报告的本例实测成功率。
