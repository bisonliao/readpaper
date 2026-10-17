# 使用视觉观测训练 Lift：IQL 学习笔记

## 背景和目的

第一个目标已经完成：使用 robomimic 的 Lift PH 数据集、视觉观测和 BC 训练策略，训练结果达到 100% rollout 成功率。

本文记录第二个目标：

> 在 robomimic 已有的 robosuite Lift 任务上，使用视觉观测和 Implicit Q-Learning（IQL）训练一个能够完成 Lift 的 agent。

这里继续复用第一个目标生成的图像数据集，但训练算法改为 IQL。IQL 属于离线强化学习，不只是直接拟合示范动作，因此除了 `obs`、动作和奖励，还需要 `next_obs` 以及 `dones`。

本文不修改 robomimic 已有代码或模板配置，只使用新的实验配置：

```text
robomimic/exps/tutorial/lift_visual_iql.json
```

## IQL 和 BC 的主要区别

BC 直接学习示范动作：

```text
观测 obs -> 动作 action
```

IQL 会同时训练：

- `critic`：估计状态和动作的 Q 值；
- `vf`：估计状态价值 V；
- `actor`：根据优势值对示范动作加权学习。

因此 IQL 的每个样本需要一个完整 transition：

```text
(obs, action, reward, next_obs, done)
```

配置中的关键差异是：

```json
"hdf5_load_next_obs": true
```

如果图像数据集没有 `next_obs`，IQL 的 critic target 无法计算。前一个目标提取图像时没有使用 `--exclude-next-obs`，所以已有的 `image_v15.hdf5` 可以直接用于本文。

## 前置条件

假设已经完成 [bc_pixel_lift.md](/home/bison/robomimic/bc_pixel_lift.md) 中的步骤，并且存在：

```text
datasets/lift/ph/image_v15.hdf5
```

这个数据集应包含：

```text
data/demo_*/obs/agentview_image
data/demo_*/obs/robot0_eye_in_hand_image
data/demo_*/next_obs/agentview_image
data/demo_*/next_obs/robot0_eye_in_hand_image
data/demo_*/actions
data/demo_*/rewards
data/demo_*/dones
```

当前环境中使用以下组合：

```text
robosuite 1.5.2
mujoco 3.3.0
```

没有 X11/Wayland 显示时，先设置 EGL 离屏渲染：

```bash
conda activate robomimic
cd /home/bison/robomimic

export MUJOCO_GL=egl
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/mpl
```

## 第一步：检查图像数据集是否满足 IQL 要求

执行：

```bash
python - <<'PY'
import h5py

path = "datasets/lift/ph/image_v15.hdf5"
with h5py.File(path, "r") as f:
    demo = sorted(f["data"].keys())[0]
    obs_keys = list(f["data"][demo]["obs"].keys())
    next_obs_keys = list(f["data"][demo]["next_obs"].keys())
    print("obs keys:", obs_keys)
    print("next_obs keys:", next_obs_keys)
    print("agentview obs:", f["data"][demo]["obs"]["agentview_image"].shape)
    print("agentview next_obs:", f["data"][demo]["next_obs"]["agentview_image"].shape)
    print("actions:", f["data"][demo]["actions"].shape)
    print("rewards:", f["data"][demo]["rewards"].shape)
    print("dones:", f["data"][demo]["dones"].shape)
PY
```

两路图像的 `obs` 和 `next_obs` 都应为类似 `(T, 84, 84, 3)` 的数组，并且 `actions`、`rewards`、`dones` 的第一个维度与轨迹长度一致。

如果 `next_obs` 不存在，需要从 raw 数据重新提取。IQL 不应使用下面这个节省空间的选项：

```bash
--exclude-next-obs
```

## 第二步：理解 IQL 图像配置

独立配置文件是：

```text
robomimic/exps/tutorial/lift_visual_iql.json
```

它使用和 BC 视觉实验相同的输入：

```json
"low_dim": [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos"
],
"rgb": [
    "agentview_image",
    "robot0_eye_in_hand_image"
]
```

这仍然是“视觉 + 本体感觉”输入，而不是纯视觉。如果要做纯视觉 IQL，应在新配置中把 `obs.low_dim` 改为 `[]`。

配置使用 `ResNet18Conv + SpatialSoftmax` 作为 RGB 编码器，并使用 `CropRandomizer` 做随机裁剪。IQL 的 actor、value function 和 critic 都会处理这些观测，因此它比 BC 更占内存。

IQL 的关键算法参数采用 robomimic 生成论文配置中的设置：

| 配置项 | 当前值 | 意义 |
| --- | ---: | --- |
| `algo.discount` | `0.99` | 奖励折扣因子。 |
| `algo.target_tau` | `0.005` | critic target 网络软更新系数。 |
| `algo.vf_quantile` | `0.7` | expectile value regression 的分位系数。 |
| `algo.adv.beta` | `3.0` | 优势加权 actor 更新的温度系数。 |
| `optim_params.critic.learning_rate.initial` | `3e-4` | critic 学习率。 |
| `optim_params.vf.learning_rate.initial` | `3e-4` | value function 学习率。 |
| `optim_params.actor.learning_rate.initial` | `3e-4` | actor 学习率。 |
| `algo.actor.layer_dims` | `[256, 256, 256]` | actor 的 MLP 层。 |
| `algo.critic.layer_dims` | `[256, 256, 256]` | critic 和 value function 的 MLP 层。 |
| `algo.critic.ensemble.n` | `2` | 两个 Q 网络组成的 critic ensemble。 |

当前机器没有 CUDA，且 IQL 会为 actor、value、两个 critic 和两个 target critic 创建视觉网络，因此独立配置采用：

```json
"num_data_workers": 0,
"cuda": false,
"batch_size": 4
```

我实际测试过 `batch_size=4` 的图像 IQL smoke test，可以完成训练和 rollout。最初使用 `batch_size=16` 时进程在第一个训练 batch 前异常结束，未留下 Python traceback，属于当前机器的资源限制。若换用显存或内存更大的机器，可以在 smoke test 成功后逐步尝试增大 batch size。

## 第三步：检查 IQL 配置可以被加载

```bash
python -m json.tool \
    robomimic/exps/tutorial/lift_visual_iql.json \
    >/dev/null

python - <<'PY'
import json
from robomimic.config import config_factory

path = "robomimic/exps/tutorial/lift_visual_iql.json"
raw = json.load(open(path))
config = config_factory(raw["algo_name"])
with config.values_unlocked():
    config.update(raw)
config.lock()

print("algorithm:", config.algo_name)
print("RGB keys:", list(config.observation.modalities.obs.rgb))
print("load next_obs:", config.train.hdf5_load_next_obs)
print("batch size:", config.train.batch_size)
print("critic learning rate:", config.algo.optim_params.critic.learning_rate.initial)
PY
```

应看到：

```text
algorithm: iql
RGB keys: ['agentview_image', 'robot0_eye_in_hand_image']
load next_obs: True
batch size: 4
critic learning rate: 0.0003
```

## 第四步：运行 IQL 图像 smoke test

`--debug` 会把训练缩短为 2 个 epoch、每个 epoch 3 个 batch，并进行 2 个、每个最多 10 步的 rollout：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/lift_visual_iql.json \
    --dataset datasets/lift/ph/image_v15.hdf5 \
    --debug
```

输出中应确认：

```text
using obs modality: rgb with keys:
['agentview_image', 'robot0_eye_in_hand_image']
```

模型摘要中应同时出现：

- `actor`；
- `critic` 和 `critic_target`；
- `vf`；
- `VisualCore`；
- `ResNet18Conv`；
- 两路 RGB 图像的 `ObservationEncoder`。

环境配置中应有：

```text
has_offscreen_renderer: true
use_camera_obs: true
```

smoke test 期间 batch size 为 4 时，IQL 的训练日志会出现 `actor/loss`、`critic/critic1_loss`、`vf/v_loss` 等指标，并完成 rollout。smoke test 的成功率通常没有意义，因为只训练了 2 个 epoch、每个 epoch 只有 3 个 batch；它只用于验证数据、网络和环境连接正确。

## 第五步：正式训练

smoke test 成功后，去掉 `--debug`：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/lift_visual_iql.json \
    --dataset datasets/lift/ph/image_v15.hdf5
```

正式配置当前设置为 600 个 epoch，每 20 个 epoch 评测一次，每次评测 10 个 rollout。输出根目录是：

```text
iql_trained_models/
```

训练期间重点观察 TensorBoard 中：

- `Train/actor/loss`；
- `Train/critic/critic1_loss`；
- `Train/vf/v_loss`；
- `Rollout/Return/image_v15`；
- `Rollout/Success_Rate/image_v15`。

IQL 的 critic、value 和 actor loss 不需要像 BC loss 一样单调下降。最终是否完成 Lift，应以 rollout 的 `Success_Rate` 为准。

启动 TensorBoard：

```bash
tensorboard --logdir iql_trained_models --bind_all
```

## 第六步：使用训练好的 IQL checkpoint 评估

训练完成后，找到某个 checkpoint，例如：

```text
iql_trained_models/lift_visual_iql/<timestamp>/models/model_epoch_*.pth
```

使用它执行 50 个 rollout，并保存视频：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robomimic/scripts/run_trained_agent.py \
    --agent /path/to/model_epoch_XXX.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --seed 0 \
    --video_path /tmp/lift_visual_iql_eval.mp4 \
    --camera_names agentview robot0_eye_in_hand
```

脚本从 checkpoint 中恢复 `env_metadata` 和 `shape_metadata`，因此知道要创建 robosuite Lift 环境，并知道策略需要两路图像和低维观测。终端中的：

```text
Average Rollout Stats
{
    "Success_Rate": ...,
    "Num_Success": ...
}
```

是最终评估结果。`Success_Rate` 应按多个 episode 的成功数除以 episode 总数理解。

## 常见问题

### 为什么 IQL 必须保留 `next_obs`

IQL 的 critic target 使用下一状态的 value：

```text
Q target = reward + (1 - done) * discount * V(next_obs)
```

因此必须同时满足：

```json
"hdf5_load_next_obs": true
```

以及 HDF5 中存在 `data/demo_*/next_obs/*`。如果使用 `--exclude-next-obs` 生成图像数据，IQL 会在读取 batch 时缺少必要字段。

### 为什么会看到 `object not found` 等观测警告

robosuite rollout 返回的观测通常比策略配置中使用的观测键更多。策略只读取配置中列出的 `low_dim` 和 `rgb` 键，因此这些额外键会被 robomimic 注册为默认 low_dim，并不会自动成为 IQL 输入。

### 为什么图像 IQL 比图像 BC 占用更多内存

BC 通常只有一个策略网络。IQL 同时维护 actor、value function、critic ensemble 和 critic target；当每个网络都处理 RGB 图像时，视觉编码器和反向传播缓存会显著增加。当前环境先使用 batch size 4 和 0 个 DataLoader worker 通过 smoke test，再根据资源逐步增大。

## 数据采样方式与 epoch 的含义

BC 和 IQL 在 robomimic 中使用同一套 DataLoader。区别在于 batch 中需要的字段和更新方式：

| 项目 | BC | Offline RL / IQL |
| --- | --- | --- |
| 数据来源 | 固定离线数据集 | 固定离线数据集 |
| 需要的字段 | `obs`、`action` | `obs`、`action`、`reward`、`next_obs`、`done` |
| 训练时是否与环境交互 | 不需要 | 不需要 |
| 更新内容 | 直接拟合动作 | 更新 critic、value 和 actor |

对于单个数据集，`SequenceDataset.get_dataset_sampler()` 返回 `None`，训练脚本因此使用带 `shuffle=True` 的 PyTorch `DataLoader`。如果完整遍历 DataLoader，其行为可以理解为：

```text
随机打乱数据索引
→ 按 batch 依次读取
→ 每个样本通常访问一次
→ 下一个 DataLoader 迭代重新打乱
```

不过，robomimic 中的一个逻辑 epoch 不一定完整遍历数据集。训练循环按照：

```python
config.experiment.epoch_every_n_steps
```

指定的 batch 数执行。当前 IQL 配置是：

```json
"epoch_every_n_steps": 100
```

因此每个 epoch 只执行 100 个 batch。只有将该值设为 `null` 时，代码才会使用完整的 DataLoader 长度作为一个 epoch 的步数。`DataLoader` 的 `drop_last=true` 还会丢弃最后不足一个完整 batch 的样本。

当前 Lift 图像数据集大约包含 9666 个 transition，IQL 配置使用：

```text
batch_size = 4
epoch_every_n_steps = 100
```

完整遍历一次数据集大约需要：

```text
9666 / 4 ≈ 2416 个 batch
```

但当前一个逻辑 epoch 只有 100 个 batch，即约 400 个样本。训练 600 个 epoch 时，累计处理约：

```text
600 × 100 × 4 = 240000 个样本
```

相当于重复使用整个数据集约 24.8 次。这里的 `num_epochs=600` 不能简单理解为完整数据集被遍历 600 次。

如果使用多个数据集，robomimic 的 `MetaDataset` 会使用带放回的加权随机采样：一个 epoch 中可能重复采到同一个 transition，也可能有些 transition 没有被采到。采样权重由各数据集的配置决定。

因此，当前 IQL 的实际训练流程是：

```text
固定的离线 transition 数据集
→ DataLoader 随机产生 mini-batch
→ 每个逻辑 epoch 处理固定数量 batch
→ IQL 更新 critic、V 和 actor
→ 每隔若干 epoch 使用 Lift 环境 rollout 评测
```

Offline RL 并没有规定必须使用“完整遍历”或“有放回随机采样”。采样方式由 replay buffer 或 DataLoader 实现决定；在 robomimic 中，BC 和 IQL 的采样机制相同，IQL 只是额外读取 `reward`、`next_obs` 和 `done`，并使用不同的损失更新网络。
