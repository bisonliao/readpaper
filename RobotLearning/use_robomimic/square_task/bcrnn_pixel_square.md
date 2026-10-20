### 使用 robomimic 论文配置训练 Square MH 视觉 BC-RNN

本文说明如何使用当前 robomimic 项目已有的代码、Square Multi-Human（MH）图像数据和论文配置，训练基于视觉观测的 BC-RNN 策略。

这里的 BC-RNN 配置实际会启用 GMM 动作头，因此训练算法类是 `BC_RNN_GMM`，训练目标是动作分布的负对数似然，而不是确定性 BC 的 MSE。

### 1. 训练目标和数据

Square 任务要求机器人抓取方形螺母，并将其插入杆上。MH 数据集包含多个不同熟练程度操作者的示范，数据质量和动作策略具有一定差异。

视觉策略使用以下观测：

| 类型 | 观测 |
|---|---|
| 低维 proprioception | `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos` |
| 外部相机 | `agentview_image` |
| 腕部相机 | `robot0_eye_in_hand_image` |
| 图像尺寸 | 84×84 |

获取数据并转换数据文件格式：

```text
python robomimic/scripts/download_datasets.py --tasks square --dataset_types mh --hdf5_types raw
python robomimic/scripts/dataset_states_to_obs.py     --dataset datasets/square/mh/demo_v15.hdf5     --output_name image_v15.hdf5     --done_mode 2     --camera_names agentview robot0_eye_in_hand     --camera_height 84     --camera_width 84     --compress
```

`image_v15.hdf5` 已包含图像观测和 `train` / `valid` filter key，可以直接用于训练。若从原始数据重新生成图像数据，应先划分 train/valid，再执行图像转换，这样 filter key 会被复制到后处理后的 HDF5 文件。

### 2. 生成并使用论文一致的配置

本实验使用的配置不是从普通 `exps/templates/bc.json` 手工拼出来的，而是 robomimic 官方提供的论文配置生成脚本产生的。脚本会按照论文中的实验分组、数据集类型、观测类型和算法超参数生成配置文件。

在项目根目录执行：

```bash
python robomimic/scripts/generate_paper_configs.py --output_dir /tmp/experiment_results
```

生成后，Square MH 图像 BC-RNN 配置位于：

```text
robomimic/exps/paper/core/square/mh/image/bc_rnn.json
```


本教程使用的正式配置正是上面生成的 `core/square/mh/image/bc_rnn.json`。它已经包含论文中的 BC-RNN-GMM、图像编码器、RNN hidden dim、训练步数和评测协议。实际运行时只对它做了适应本机数据和资源的修改：

| 项目 | 论文生成配置 | 实际运行中的调整 |
|---|---|---|
| 数据路径 | `image_v141.hdf5` | 命令行覆盖为 `image_v15.hdf5` |
| 输出目录 | `/tmp/experiment_results/...` | 使用本地 `bc_rnn_trained_models` 目录 |
| rollout 视频 | 开启 | 关闭 `render_video` 以降低内存和磁盘开销 |
| rollout episode 数 | 50 | 改为 20 以缩短评测时间 |
| rollout horizon | 500 | 改为 400 以缩短失败 episode 的执行时间 |
| rollout warmup | 0 | 改为 40，前 40 个 epoch 不做在线评测 |

这些调整没有改变 BC-RNN-GMM 的核心训练结构：GMM 仍然开启，RNN hidden dim 仍为 1000，序列长度仍为 10，学习率仍为 1e-4。修改后保存为：

```text
robomimic/exps/tutorial/square_visual_bc_rnn.json
```


主要参数如下：

| 配置项 | 论文配置 |
|---|---:|
| 算法入口 | `algo_name = "bc"`，通过 RNN/GMM 选出 `BC_RNN_GMM` |
| 图像 backbone | ResNet-18 |
| 图像池化 | SpatialSoftmax，32 keypoints |
| 图像随机裁剪 | 76×76，从 84×84 输入随机裁剪 |
| RNN | 2 层 LSTM |
| RNN hidden dim | 1000 |
| RNN horizon | 10 |
| `train.seq_length` | 10 |
| actor MLP | 空列表，RNN 直接连接动作输出头 |
| GMM | 开启，5 个 mixture modes |
| GMM 最小标准差 | 0.0001 |
| 学习率 | 1e-4，Adam |
| batch size | 16 |
| 每 epoch 更新数 | 500 |
| 验证更新数 | 50 |
| 训练 epoch 数 | 600 |
| rollout 数量 | 50 |
| rollout horizon | 500 |
| rollout 间隔 | 每 20 epoch |
| train/valid | `mask/train` 与 `mask/valid` |

论文配置中的 `epoch_every_n_steps=500` 是固定 optimizer update 数，不代表每个 epoch 都完整遍历一次数据集。这样不同任务和数据集可以使用相近的训练预算。

### 3. 数据版本说明

论文配置文件默认指向：

```text
datasets/square/mh/image_v141.hdf5
```

但当前工作区下载转换得到的是：

```text
datasets/square/mh/image_v15.hdf5
```

因此使用当前数据时，实际含义是“论文超参数 + 当前 v15 图像数据”，而不是论文原始实验的完全复现。图像任务对 robosuite 版本、渲染器、纹理和相机输出很敏感。若要尽可能复现论文数值，需要使用与论文数据和环境版本匹配的 `image_v141.hdf5` 及对应 robosuite 版本；如果使用当前 v15 数据，应把结果视为当前环境下的复现实验。

### 4. 检查数据集

先检查数据结构、轨迹长度和 filter key：

```bash
python robomimic/scripts/get_dataset_info.py \
    --dataset datasets/square/mh/image_v15.hdf5
```

应确认输出中存在：

```text
mask/train
mask/valid
agentview_image
robot0_eye_in_hand_image
```

可用下面的命令检查 train/valid 的 episode 数量：

```bash
python - <<'PY'
import h5py

path = "datasets/square/mh/image_v15.hdf5"
with h5py.File(path, "r") as f:
    for key in ["train", "valid"]:
        demos = [x.decode() if isinstance(x, bytes) else x for x in f["mask"][key][()]]
        print(key, "episodes:", len(demos))
PY
```

MH 数据通常使用 90% episode 训练、10% episode 验证。训练和验证必须使用互不重叠的 episode，而不是随机混合同一条轨迹中的 transition。

### 5. 先运行小规模 sanity check

在正式训练前，先用 `--debug` 检查环境、视觉输入、模型构建和 rollout 是否正常：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/square_visual_bc_rnn.json \
    --dataset datasets/square/mh/image_v15.hdf5 \
    --debug
```

`--debug` 会缩短训练为少量 step，并将输出写到临时目录。sanity check 应至少确认：

- ResNet-18 和 SpatialSoftmax 成功建立；
- `agentview_image` 与 `robot0_eye_in_hand_image` 能被读取；
- RNN hidden dim 为 1000；
- GMM 动作头被启用；
- 环境可以创建并执行 rollout；
- 没有 CUDA out-of-memory、HDF5 锁错误或图像 shape 错误。

### 6. 正式训练

使用当前已经生成的 `image_v15.hdf5`，通过命令行 `--dataset` 覆盖论文 JSON 中的旧数据路径：

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robomimic/scripts/train.py \
    --config robomimic/exps/tutorial/square_visual_bc_rnn.json \
    --dataset datasets/square/mh/image_v15.hdf5
```

`--dataset` 会覆盖 JSON 中的 `train.data`，但不会改变其他论文超参数。





### 7. 本次实际训练结果

按照上述论文生成配置并结合本机资源做调整后，Square MH 图像 BC-RNN 在当前 `image_v15.hdf5` 数据上的在线 rollout 最高成功率约为：

```text
70%
```

本次结果说明论文配置在当前环境下可以有效训练出 Square 视觉策略，但它不应被当作论文原始结果的完全复现，原因包括：

- 本地使用的是 `image_v15.hdf5`，论文生成配置默认指向 `image_v141.hdf5`；
- 当前 robosuite / MuJoCo 渲染环境可能与论文实验环境不同；
- 本次使用 20 个 rollout episode，成功率步长为 5%；
- 本次是单个随机种子，而论文结果通常基于多个 seed 的统计；
- 本次关闭了 rollout 视频并缩短了 horizon 和评测频率。

因此，70% 更准确的表述是：

> 在当前 robomimic 代码、`image_v15.hdf5` 数据和论文生成的 BC-RNN-GMM 配置基础上，经资源约束调整后，Square MH 视觉策略单次训练的最高在线成功率约为 70%。

