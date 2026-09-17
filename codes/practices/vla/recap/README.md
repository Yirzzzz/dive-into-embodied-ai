# RECAP + π₀.₅ 的LIBERO 长时程任务全流程复现



本教程在两张 A100 上实现，使用 RLINF 开源的 LEBERO 长时程仿真数据训练基于 π₀.₅ 实现 RECAP。在成功率有限的数据集中，对比 π₀.₅ 与 π₀.₅ + RECAP的成功率，以及可视化价值函数/优势函数。



## 你将完成什么

这次实验走通一条完整、可复现的recap实现的链路：

1. 从 RLINF 开源数据集上下载 4096 条 episode；
2. 基于 π₀.₅ 模型训练价值函数、计算优势函数、做 CFG 后训练；
3. 持续记录 CSV 指标并自动生成训练曲线；
4. 定期保存模型、优化器和随机数状态；
5. 将成功回合转换为可嵌入教程的 GIF。

RECAP 本身在于基于不完美的模型policy上优化模型，因此，为了更加贴合实际情况，数据集中既要有成功的 episode 以及失败的 episode，让模型学会什么是好的动作和坏的动作，实现自主学习进化。

## 实验结果

本教程页面中的结果来自一次真实本机运行，不是预填示例：

| 项目                 |                           实测结果 |
| -------------------- | ---------------------------------: |
| GPU                  |         NVIDIA RTX 4080 SUPER 16GB |
| 数据                 | 50 episodes / 20,000 frames / 50Hz |
| 训练步数             |             50,000 optimizer steps |
| Batch size           |                                  8 |
| 动作块长度           |                         100 frames |
| 混合精度             |                       bfloat16 AMP |
| 50k 用时             |           2,354 秒，约 39 分 14 秒 |
| 稳定吞吐             |                     约 21.3 step/s |
| PyTorch 峰值分配显存 |                         约 2.03GiB |
| 50k L1 action loss   |                             0.0631 |
| 20 回合成功率        |                     **10/20，50%** |
| 平均最大 reward      |                          2.8 / 4.0 |
| 平均累计 reward      |                              161.6 |

一次评估曾恰好得到失败回合，最大 reward 只有 2；扩展到 10 回合时为 60%，最终 20 回合为 50%。这说明机器人策略不能靠一个“看起来不错”的视频下结论，至少应报告多回合成功率。

## 项目文件

本项目是基于 Openpi 与 LIBERO 仿真项目进行复现实现的。

RECAP 项目文件目录：

```
codes/practices/vla/recap/
├── run.py          # 统一命令入口，转发 data/stats/train/annotate/serve/rollout/checkpoint 子命令
├── config.py       # 注册 SFT、value、ACP 配置，定义数据变换、初始化权重和参数冻结规则
├── data.py         # 下载、完整性校验并准备 sft/train/eval 三份本地 LeRobot 数据
├── value_model.py  # 定义 π₀.₅ 的分布式价值头，以及从 pi05_base 加载骨干权重的逻辑
├── annotate.py     # 计算 value target、验证价值模型，并生成 advantage 和正负样本标签
├── rollout.py      # 在 LIBERO 中采集或评估 episode，保存轨迹、视频和成功率结果
├── test_recap.py   # 测试数据处理、回报/优势计算、模型接口和 OpenPI 训练器接入
└── README.md       # 环境、数据、价值训练、ACP 微调和仿真评估的操作教程
```

本次复现是基于Openpi框架复现，若没有相关环境，需要先快速部署：

```cmd
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c 'import jax; print(jax.devices())' # 最后一行应列出 CUDA GPU 即为成功
```

接着将目录 codes/practices/vla/recap/ 复制到 Openpi 环境中

```
cp -R ../dive-into-embodied-ai/codes/practices/vla/recap examples/recap
```

## 数据与模型准备

### 模型

```cmd
uv run python - <<'PY'
from openpi.shared import download

resources = [
    ("gs://openpi-assets/checkpoints/pi05_base/params", {}),
    ("gs://big_vision/paligemma_tokenizer.model", {"gs": {"token": "anon"}}),
]

for url, options in resources:
    path = download.maybe_download(url, **options)
    print(f"{url} -> {path}")
PY
```

### 数据集下载及相关介绍

```bash
cd openpi
cp -R /data1/evan/dive-into-embodied-ai/codes/practices/vla/recap \
  /data1/evan/openpi/examples/recap

export HF_LEROBOT_HOME="$PWD/data/lerobot"
export RECAP_REPO_ID="local/libero10_task0_train"
export RECAP_EVAL_REPO_ID="local/libero10_task0_eval"

# 镜像下载数据集
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_PROGRESS_BARS=0 \
HF_HUB_ETAG_TIMEOUT=60 \
HF_HUB_DOWNLOAD_TIMEOUT=120 \
uv run python examples/recap/run.py data download
```

该数据集区别于LIBERO官方数据集，本次复现是为了研究如何在一个不完美的数据让模型自主学习进化，因此，训练数据集包含了成功以及失败的episode，数据集的内容与用途如下：

| 发布目录 | 内容 | 本例用途 |
| --- | --- | --- |
| `libero10_task0_sft` | 30 条成功示教，覆盖 10 个任务，每任务 3 条 | 计算 π₀.₅ 的状态和动作 norm stats |
| `libero10_task0_train` | Task 0 的 4,096 条 rollout，1,999 成功、2,097 失败 | 价值训练、优势标注、ACP |
| `libero10_task0_eval` | Task 0 的 64 条 rollout，27 成功、37 失败 | 价值模型验证与 checkpoint 选择 |

数据准备：

```bash
# 成功示教：用于计算 norm stats
uv run python examples/recap/run.py data prepare --split sft

# 成功和失败的 rollout：用于价值训练、优势标注和 ACP
uv run python examples/recap/run.py data prepare --split train

# 留出 rollout：用于验证价值模型
uv run python examples/recap/run.py data prepare --split eval
```

数据已经是 LeRobot 格式，为固定 OpenPI 所用的 LeRobot v2.1 元数据布局。如果还不熟悉 LeRobot 的作用、`LeRobotDataset` 的目录结构，以及 Parquet、MP4 和 `meta/*.jsonl` 如何共同描述轨迹，先阅读项目内的
[LeRobot 中文课程讲义：LeRobotDataset](../../../../docs/practices/robot-arm/data-collection/lerobot-course/index.md#lerobotdataset机器人数据为什么必须重新设计)。

统计量计算：

```bash
uv run python examples/recap/run.py stats --config-name pi05_recap_sft
```

**统计输出：** `assets/pi05_recap_sft/local/libero10_fewshot_sft/norm_stats.json`。

一切准备就绪，可以开始进行价值函数训练以及 CFG 模型后训练啦！

## 1. 计算训练集和验证集样本 return

对数据轨迹计算reward，成功轨迹和失败轨迹最后默认为0/-300：

```bash
uv run python examples/recap/run.py annotate targets \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_REPO_ID"
# Annotated 4096 episodes in ...
uv run python examples/recap/run.py annotate targets \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_EVAL_REPO_ID"
# Annotated 64 episodes in ...
```

## 2. 训练价值模型

基于前置准备的 `pi05_base` 模型训练价值模型：

```
uv run python examples/recap/run.py train pi05_recap_value   --exp-name value   --num-train-steps 30000   --batch-size 18   --fsdp-devices 2   --save-interval 10000   --keep-period 10000   --no-wandb-enabled
```

⚠️ 4096个 episode，batch size 为18的情况下粗略估算大概需要90000步左右才可以练完一个epoch，为了不增加学习复现负担，本次复现仅训练到20000步。

对比两个checkpoint在验证集的结果：

``` cmd
for steps in 10000 20000; do
  VALUE_CKPT="$PWD/checkpoints/pi05_recap_value/value/$steps"

  uv run python examples/recap/run.py annotate evaluate \
    --dataset-root "$HF_LEROBOT_HOME/$RECAP_EVAL_REPO_ID" \
    --checkpoint "$VALUE_CKPT" \
    --batch-size 8 \
    --max-frames 10000 \
    --output "data/recap/value_eval/${steps}.json"
done
```

| Checkpoint | MSE         | MAE         |
| ---------- | ----------- | ----------- |
| 10k        | **0.03620** | 0.11694     |
| 20k        | 0.03750     | **0.11565** |

## 3. 选择 checkpoint，计算优势值

基于 10k checkpoint，计算每个样本动作的优势值，前30%得分的动作认为是 positive 动作：

```bash
export HF_LEROBOT_HOME="$PWD/data/lerobot"
export RECAP_REPO_ID="local/libero10_task0_train"

uv run python examples/recap/run.py annotate advantages \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_REPO_ID" \
  --checkpoint "$VALUE_CKPT" \
  --batch-size 72 \
  --num-workers 4 \
  --n-step 10 \
  --positive-ratio 0.3
```

`--num-workers 4` 会并行解码并预取双视角视频，进度日志同时显示处理速度和预计剩余时间。
如果机器内存或共享内存不足，可降到 `--num-workers 2`；显存不足则降低 `--batch-size`。

## 4. CFG 微调

```bash
export CUDA_VISIBLE_DEVICES=6,7
uv run python examples/recap/run.py train pi05_recap_acp \
  --exp-name acp --batch-size 18 --fsdp-devices 2 --no-wandb-enabled

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

若已经额外训练了普通 SFT 对照，终端 A 可先启动它；新终端需要重新设置第 2 节变量，并查询 checkpoint：

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

```bash
uv run python -m pytest examples/recap/test_recap.py -q
```

本地 29 项 CPU 测试通过，覆盖数据准备、回报与优势、验证集隔离，以及完整模型在上游训练器中的初始化、反向传播、优化器和 EMA 的形状检查。另用三个划分各一条真实轨迹检查了视频/动作读取，并通过上游入口在 SFT 小样本上计算了 norm stats。

另外使用随机初始化的缩小版骨干，实际验证了 SFT checkpoint → 价值模型初始化 → CPU 梯度更新 → 保存 → 回读预测；这不等于真实 π₀.₅ 权重验证，也不能用来判断显存或任务成功率。完整数据集处理、正式 GPU 训练及真实 LIBERO 评估尚未完成，没有可报告的本例实测成功率。
