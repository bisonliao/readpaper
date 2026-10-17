# 使用视觉观测训练 Lift：BC 学习笔记

## 背景和目的

robomimic 的配置教程介绍了如何启动训练，却没有用一个完整的小项目说明如何从图像数据开始训练一个策略。本文记录我使用 robomimic 和 robosuite 完成第一个学习目标的过程：

> 在 robomimic 已有的 robosuite Lift 任务上，使用视觉观测和 Behavior Cloning（BC）训练一个能够执行 Lift 的 agent。

后续学习目标还包括：

1. 使用视觉观测和 IQL 训练 Lift；
2. 将自己的 PickPlace 环境接入 robomimic，并使用视觉观测和 BC 训练；
3. 使用视觉观测和 IQL 训练自己的 PickPlace 任务。

本文只覆盖第一个目标。

这里的“视觉 BC”采用 robomimic 官方 image 实验的默认输入：两个相机图像，加上末端位姿和夹爪位置。因此它是“视觉 + 本体感觉”输入。如果只希望使用纯图像输入，可以在实验配置中把 `obs.low_dim` 设为空列表。

## 版本和渲染环境

robomimic v0.5.0 的 Lift 数据集使用 robosuite v1.5.1 的环境元数据。当前环境安装的是 robosuite v1.5.2。为避免新版本 MuJoCo 与 robosuite 的离屏渲染兼容问题，将 MuJoCo 固定为 v3.3.0：

```bash
conda activate robomimic
python -m pip install --force-reinstall "mujoco==3.3.0"
```

当前运行环境没有 X11/Wayland 桌面显示，但 EGL 软件离屏渲染可用。因此训练和生成视频前设置：

```bash
export MUJOCO_GL=egl
export NUMBA_DISABLE_JIT=1
```

`NUMBA_DISABLE_JIT=1` 用于避免当前环境中 robosuite 的 Numba 缓存初始化问题。若本机已经配置好 robosuite 的 private macros，也可以不设置它。

验证版本：

```bash
python - <<'PY'
import sys
import mujoco
from importlib.metadata import version

print(sys.executable)
print("robomimic:", version("robomimic"))
print("robosuite:", version("robosuite"))
print("mujoco:", mujoco.__version__)
PY
```

## 为什么不能直接使用现有的 low_dim 数据集

仓库中的 `datasets/lift/ph/low_dim_v15.hdf5` 只包含低维观测，例如：

```text
object
robot0_eef_pos
robot0_eef_quat
robot0_gripper_qpos
robot0_joint_pos
robot0_joint_vel
```

它没有 `agentview_image` 或 `robot0_eye_in_hand_image`。仅仅在配置中填写 RGB 观测名称不能把低维数据变成图像数据。

robomimic 的 `download_datasets.py` 目前将 Lift PH 的 image 下载链接登记为 `None`。因此正确流程是先下载 raw 数据，再用 `dataset_states_to_obs.py` 从 MuJoCo 状态重新渲染图像。

## 第一步：下载 Lift PH 原始数据

在仓库根目录执行：

```bash
cd /home/bison/robomimic

python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types ph \
    --hdf5_types raw \
    --download_dir datasets
```

下载完成后，原始数据应位于：

```text
datasets/lift/ph/demo_v15.hdf5
```

可以先使用 `--dry_run` 检查路径和数据类型：

```bash
python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types ph \
    --hdf5_types raw \
    --download_dir datasets \
    --dry_run
```

## 第二步：从 raw 数据提取图像观测

使用 Lift 的外部相机 `agentview` 和腕部相机 `robot0_eye_in_hand`，图像大小设置为 84×84：

```bash
export MUJOCO_GL=egl
export NUMBA_DISABLE_JIT=1

python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/lift/ph/demo_v15.hdf5 \
    --output_name image_v15.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 \
    --camera_width 84 \
    --compress
```

这里没有使用 `--exclude-next-obs`。虽然纯 BC 不需要 `next_obs`，但后续 IQL 训练需要它；保留它可以让同一个图像数据集继续用于第二个学习目标。

输出文件应为：

```text
datasets/lift/ph/image_v15.hdf5
```

检查数据结构：

```bash
python - <<'PY'
import h5py

path = "datasets/lift/ph/image_v15.hdf5"
with h5py.File(path, "r") as f:
    demo = sorted(f["data"].keys())[0]
    obs = f["data"][demo]["obs"]
    print("obs keys:", list(obs.keys()))
    print("agentview:", obs["agentview_image"].shape)
    print("wrist:", obs["robot0_eye_in_hand_image"].shape)
    print("has next_obs:", "next_obs" in f["data"][demo])
PY
```

预期结果类似：

```text
agentview: (trajectory_length, 84, 84, 3)
wrist: (trajectory_length, 84, 84, 3)
has next_obs: True
```

图像在 HDF5 中保存为 `HWC` 格式；robomimic 的观测处理代码会在送入网络前转换为适合 PyTorch 的格式。

## 第三步：使用独立的视觉 BC 配置

为了不修改仓库已有的模板配置，实验配置单独保存在：

```text
robomimic/exps/tutorial/lift_visual_bc.json
```

该配置的关键设置是：

```json
{
    "algo_name": "bc",
    "experiment": {
        "render_video": true,
        "rollout": {
            "enabled": true,
            "horizon": 400,
            "rate": 20
        }
    },
    "train": {
        "num_data_workers": 2,
        "hdf5_cache_mode": "low_dim",
        "batch_size": 16,
        "num_epochs": 600
    },
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos"
                ],
                "rgb": [
                    "agentview_image",
                    "robot0_eye_in_hand_image"
                ]
            }
        }
    }
}
```

RGB 编码器使用 `ResNet18Conv + SpatialSoftmax`，训练时使用 `CropRandomizer`。这些设置来自 `generate_paper_configs.py` 中的官方 image 实验配置。

注意：`examples/train_bc_rnn.py` 使用的是 BC-RNN。本文使用 `robomimic/scripts/train.py` 加载 `algo_name: "bc"` 的配置，训练的是 plain BC。

## 配置文件逐项说明

`lift_visual_bc.json` 是标准 JSON，不能直接加入 `//` 或 `#` 注释，否则 `train.py` 的 `json.load` 会解析失败。下面按配置层级说明各项意义。

### `algo_name`

```json
"algo_name": "bc"
```

选择 robomimic 的算法实现。`bc` 是 Behavior Cloning：根据示范动作监督训练策略。

### `experiment`

这部分控制实验输出、checkpoint、环境评测和视频：

| 配置项 | 意义 |
| --- | --- |
| `name` | 实验名称，用于输出目录和日志标识。 |
| `validate` | 是否计算验证集 loss。当前为 `false`。 |
| `logging.terminal_output_to_txt` | 是否把终端输出同时写入 `log.txt`。 |
| `logging.log_tb` | 是否写 TensorBoard 标量。 |
| `logging.log_wandb` | 是否使用 Weights & Biases。 |
| `logging.wandb_proj_name` | W&B 项目名。 |
| `save.enabled` | 是否保存 checkpoint。 |
| `save.every_n_seconds` | 按时间保存；`null` 表示关闭这个条件。 |
| `save.every_n_epochs` | 每隔多少个 epoch 保存一次。 |
| `save.epochs` | 额外指定保存的 epoch 列表。 |
| `save.on_best_validation` | validation loss 最好时保存。 |
| `save.on_best_rollout_return` | rollout return 最好时保存。 |
| `save.on_best_rollout_success_rate` | rollout 成功率最好时保存。 |
| `epoch_every_n_steps` | 一个训练 epoch 执行的 batch 数上限。 |
| `validation_epoch_every_n_steps` | 验证 epoch 执行的 batch 数上限。 |
| `env` | 覆盖数据集中的环境名；`null` 表示使用数据集元数据。 |
| `additional_envs` | 额外评测环境；`null` 表示不添加。 |
| `render` | 是否打开桌面窗口；无 X11/Wayland 时应为 `false`。 |
| `render_video` | 是否启用离屏渲染并保存 rollout 视频；图像观测需要离屏渲染。 |
| `keep_all_videos` | 是否保留每次评测产生的全部视频。 |
| `video_skip` | 每隔多少个环境步写一帧视频。 |
| `env_meta_update_dict` | 覆盖数据集环境元数据的字段。 |
| `ckpt_path` | 从已有 checkpoint 初始化；`null` 表示从头训练。 |

`experiment.rollout` 控制训练中的策略评测：

| 配置项 | 意义 |
| --- | --- |
| `enabled` | 是否执行 rollout 评测。 |
| `n` | 每次评测的 episode 数。 |
| `horizon` | 每个 episode 的最大环境步数。 |
| `rate` | 每隔多少个 epoch 评测一次。 |
| `warmstart` | 跳过前多少个 epoch 后才开始评测。 |
| `terminate_on_success` | 任务成功后是否立即结束 episode。 |

### `train`

这部分控制数据集、DataLoader 和训练循环：

| 配置项 | 意义 |
| --- | --- |
| `data` | 数据集路径；命令行 `--dataset` 会覆盖它。 |
| `output_dir` | 日志、模型、视频和保存配置的根目录。 |
| `normalize_weights_by_ds_size` | 多数据集训练时是否按数据集大小调整权重。 |
| `num_data_workers` | 图像 DataLoader 的 worker 数量。 |
| `hdf5_cache_mode` | HDF5 缓存模式；`low_dim` 只缓存低维数据，图像按 batch 读取。 |
| `hdf5_use_swmr` | 是否使用 HDF5 SWMR 读取模式。 |
| `hdf5_load_next_obs` | 是否预加载 `next_obs`；BC 不需要，当前关闭。 |
| `hdf5_normalize_obs` | 是否使用数据集统计量标准化观测。 |
| `hdf5_filter_key` | 训练 demonstration 的 mask key；`null` 使用全部数据。 |
| `hdf5_validation_filter_key` | 验证 demonstration 的 mask key。 |
| `seq_length` | 输入的连续时间步数；`1` 表示单步 plain BC。 |
| `pad_seq_length` | 短序列是否补齐到 `seq_length`。 |
| `frame_stack` | 每个时间步堆叠的连续帧数。 |
| `pad_frame_stack` | episode 开头不足帧数时是否补齐。 |
| `dataset_keys` | 从 HDF5 读取的字段：`actions`、`rewards`、`dones`。 |
| `action_keys` | 策略预测的动作字段；这里是 `actions`。 |
| `action_config.actions.normalization` | 动作归一化方式；`null` 表示不额外指定。 |
| `goal_mode` | goal-conditioned 模式；`null` 表示不使用单独 goal。 |
| `cuda` | 有 GPU 时是否使用 CUDA。 |
| `batch_size` | 每个优化 batch 的样本数。 |
| `num_epochs` | 总训练 epoch 数。 |
| `seed` | NumPy 和 PyTorch 随机种子。 |
| `max_grad_norm` | 梯度裁剪阈值；`null` 表示不裁剪。 |

### `algo`

`algo.optim_params.policy` 中，`optimizer_type: "adam"` 选择 Adam，`learning_rate.initial` 是初始学习率，`decay_factor` 和 `epoch_schedule` 控制可选的分段降学习率，`regularization.L2` 是 L2 正则强度。

`algo.loss` 中的 `l2_weight`、`l1_weight` 和 `cos_weight` 分别是动作平方误差、绝对误差和余弦误差的权重。本配置只使用 L2 动作损失。

`actor_layer_dims: [1024, 1024]` 表示观测编码结果送入动作头之前经过两个 1024 维全连接层。

本配置关闭 `algo.gaussian.enabled`，使用 `algo.gmm` 作为动作分布：

| 配置项 | 意义 |
| --- | --- |
| `gmm.enabled` | 启用混合高斯动作头。 |
| `gmm.num_modes` | 高斯混合成分数量，当前为 5。 |
| `gmm.min_std` | 动作标准差下限，避免分布退化。 |
| `gmm.std_activation` | 将网络输出转换为正标准差的函数。 |
| `gmm.low_noise_eval` | 评测时使用低噪声动作。 |

`algo.vae`、`algo.rnn` 和 `algo.transformer` 是其他策略结构的预留配置，当前都通过 `enabled: false` 关闭。因此本文训练的是 plain BC，而不是 BC-RNN、BC-VAE 或 BC-Transformer。VAE 中的 latent、prior、decoder 字段，RNN 中的 horizon、hidden_dim、层数，以及 Transformer 中的 context length、层数和 dropout，在对应结构关闭时都不会生效。

### `observation.modalities`

这部分决定策略实际读取哪些观测：

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

`low_dim` 是末端位置、末端姿态和夹爪位置；`rgb` 是外部相机和腕部相机。`depth`、`scan` 为空，表示不使用深度图和扫描数据。`goal` 下所有列表为空，表示不是 goal-conditioned 策略。

如果要改成纯视觉输入，只需在一份新配置中把 `obs.low_dim` 改成 `[]`。如果把 `obs.rgb` 改为空，训练就会退回低维 BC。

### `observation.encoder`

`encoder.low_dim.core_class: null` 表示低维值直接拼接，不经过额外网络。

RGB 编码器使用 `VisualCore`，其关键字段如下：

| 配置项 | 意义 |
| --- | --- |
| `core_class` | 编码器类型；这里是 `VisualCore`。 |
| `core_kwargs.feature_dimension` | 视觉特征维度，当前为 64。 |
| `core_kwargs.backbone_class` | CNN 骨干，这里是 `ResNet18Conv`。 |
| `backbone_kwargs.pretrained` | 是否加载预训练权重；当前为 `false`。 |
| `backbone_kwargs.input_coord_conv` | 是否追加坐标通道；当前关闭。 |
| `pool_class` | 池化方式，这里是 `SpatialSoftmax`。 |
| `pool_kwargs.num_kp` | 每个图像流的空间关键点数量。 |
| `pool_kwargs.learnable_temperature` | 是否学习 SpatialSoftmax 温度。 |
| `pool_kwargs.temperature` | 固定或初始温度。 |
| `pool_kwargs.noise_std` | 特征噪声标准差；当前为 0。 |
| `obs_randomizer_class` | 图像增强类型，这里是 `CropRandomizer`。 |
| `obs_randomizer_kwargs.crop_height/crop_width` | 随机裁剪的高和宽。 |
| `obs_randomizer_kwargs.num_crops` | 每张图像生成的裁剪数量。 |
| `obs_randomizer_kwargs.pos_enc` | 是否加入裁剪位置编码。 |

`encoder.depth` 和 `encoder.scan` 虽然保留在配置结构中，但因为对应 modality 列表为空，当前训练不会使用它们。

## 第四步：先运行 smoke test

`--debug` 会把训练缩短为 2 个 epoch、每个 epoch 3 个梯度步骤，并进行短 rollout。执行：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/lift_visual_bc.json \
    --dataset datasets/lift/ph/image_v15.hdf5 \
    --debug
```

输出中必须出现类似内容：

```text
using obs modality: rgb with keys:
['agentview_image', 'robot0_eye_in_hand_image']
```

环境信息中应包含：

```text
has_offscreen_renderer: true
use_camera_obs: true
```

模型摘要中应出现图像编码器，例如 `VisualCore`、`ResNet18Conv` 和对应的 RGB `ObservationEncoder`。如果输出中是 `rgb: []`，说明仍然使用的是低维配置。

## 第五步：正式训练

smoke test 成功后，去掉 `--debug`：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/lift_visual_bc.json \
    --dataset datasets/lift/ph/image_v15.hdf5
```

训练过程会每 20 个 epoch 进行 rollout，并在 `render_video=true` 时写入视频。配置中的 `hdf5_cache_mode="low_dim"` 只缓存低维数据，图像按 batch 从 HDF5 读取，以降低内存占用。

## 第六步：独立评估 checkpoint

训练完成后，使用保存的 checkpoint 进行独立 rollout：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
python robomimic/scripts/run_trained_agent.py \
    --agent /path/to/model.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --video_path /tmp/lift_visual_bc_eval.mp4 \
    --camera_names agentview robot0_eye_in_hand
```

最后查看输出中的：

```text
Average Rollout Stats
{
    "Success_Rate": ...,
    "Num_Success": ...
}
```

`Success_Rate` 才是判断 agent 是否完成 Lift 任务的主要指标。训练 loss 下降并不等价于任务成功，因此必须结合 rollout 结果判断。



## 环境与 rollout 评测释疑

训练命令不需要额外传入 `Lift` 任务名。`train.py` 会从数据集的 `data.attrs["env_args"]` 读取环境元数据，其中包含：

```json
{
    "env_name": "Lift",
    "type": "robosuite",
    "env_kwargs": {}
}
```

训练脚本据此创建 Lift 环境。配置中的：

```json
"rollout": {"enabled": true}
```

表示训练期间用当前策略执行环境评测，因此 TensorBoard 中的 `Rollout/*` 是评测指标，例如 `Return`、`Success_Rate` 和 `Horizon`。`render_video` 只控制是否保存评测视频；关闭它仍可以保留 rollout 指标。

保存 checkpoint 时，robomimic 会同时保存模型、训练配置、`env_metadata` 和 `shape_metadata`。因此运行：

```bash
python robomimic/scripts/run_trained_agent.py \
    --agent /path/to/model.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --video_path /tmp/lift_bc.mp4 \
    --camera_names agentview robot0_eye_in_hand
```

脚本会从 checkpoint 恢复 Lift 环境及其图像观测配置，不需要再指定任务名。`--env` 只用于强制覆盖 checkpoint 中保存的环境名称。
