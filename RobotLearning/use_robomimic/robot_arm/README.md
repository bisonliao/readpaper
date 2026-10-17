# Custom PickPlace in robomimic

This directory connects `robot_arm.RobotEnv.PixelTaskEnv`, the expert NPZ
dataset, and robomimic. The integration is split into a reusable environment
and dataset layer, followed by algorithm-specific configurations.

No robomimic source file is modified.

## 1. Integrating a custom environment, task, and dataset

The integration has three independent parts:

```text
expert NPZ files
    -> convert_expert_npz.py
    -> robomimic HDF5 dataset

PixelTaskEnv
    -> env_adapter.py
    -> robomimic EnvBase interface

env_registry.py
    -> runtime registration for custom environment type 4
```

The runtime launchers register the custom environment before delegating to
robomimic:

```text
launch_train.py -> robomimic.scripts.train
launch_eval.py  -> robomimic.scripts.run_trained_agent
```

The expert data contains two stacked frames with three RGB views. The first
version keeps the newest frame and writes three standard HWC RGB keys:

```text
xy_image: (T, 96, 96, 3)
xz_image: (T, 96, 96, 3)
yz_image: (T, 96, 96, 3)
proprio:  (T, 26)
```

The converted dataset also contains `actions`, `rewards`, `dones`, and
`next_obs`. `dones` is `terminated | truncated`. The last `next_obs` is a copy
of the final observation so every transition has a successor observation.

Convert the expert data:

```bash
python robot_arm/robomimic_pick_place/convert_expert_npz.py \
    --input robot_arm/expert_showcase_v4 \
    --output robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5 \
    --success-only
```

The converter keeps all episodes in the HDF5 file. `--success-only` controls
the training candidates and writes these masks:

```text
mask/all
mask/success
mask/train
mask/valid
mask/train_success
mask/valid_success
```

The current BC and IQL configurations use `mask/train` and `mask/valid`, so
only successful demonstrations are used for the default training split while
failed episodes remain available for analysis.

Inspect the converted dataset:

```bash
python robot_arm/robomimic_pick_place/inspect_dataset.py \
    --dataset robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5
```

The adapter exposes this robomimic observation dictionary:

```python
{
    "xy_image": uint8 RGB image,
    "xz_image": uint8 RGB image,
    "yz_image": uint8 RGB image,
    "proprio": float32 vector,
}
```

The action remains the environment's four-dimensional normalized action:

```text
[dx, dy, dz, gripper]
```

The HDF5 `data.attrs["env_args"]` stores the custom environment type and
constructor arguments. The runtime registry maps that type to
`RobomimicPixelPickPlaceEnv`, allowing checkpoints to restore the same task.

Both launchers support module execution and direct file execution:

```bash
python -m robot_arm.robomimic_pick_place.launch_train ...
python robot_arm/robomimic_pick_place/launch_train.py ...
```

The same two forms are available for `launch_eval.py`.

## 2. Algorithm examples

The environment adapter and HDF5 dataset are shared by all robomimic
algorithms. An algorithm is selected by the configuration's `algo_name` and
the corresponding fields under `algo`.

### 2.1 Behavior Cloning (BC)

Configuration:

```text
robot_arm/robomimic_pick_place/configs/bc_pixel_pick_place.json
```

This file intentionally contains the complete BC configuration hierarchy from
the Lift visual BC example: optimizer, losses, Gaussian and GMM distributions,
VAE/RNN/Transformer placeholders, observation encoders, logging, checkpoint,
and rollout settings.

The custom-task values are:

| Setting | Value | Reason |
| --- | --- | --- |
| `obs.low_dim` | `proprio` | The custom environment exposes one 26-dimensional proprio vector. |
| `obs.rgb` | `xy_image`, `xz_image`, `yz_image` | Three orthogonal camera views. |
| RGB crop | `84 x 84` | Source frames are `96 x 96`. |
| `actor_layer_dims` | `[512, 512]` | Lower memory cost for three image streams. |
| `gmm.enabled` | `false` | First deterministic baseline; easier to diagnose integration and action errors. |
| `rollout.horizon` | `150` | Matches the custom environment horizon. |
| `hdf5_filter_key` | `train` | Uses the converter's episode split. |
| `experiment.validate` | `true` | Evaluates the held-out `valid` split. |
| `batch_size` | `8` | Conservative visual-training default. |

The full Gaussian/GMM sections remain visible even though both are disabled.
If deterministic BC underfits or the demonstrations contain genuinely
multimodal actions, enable GMM and compare against this baseline.

Smoke test:

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robot_arm/robomimic_pick_place/launch_train.py \
    --config robot_arm/robomimic_pick_place/configs/bc_pixel_pick_place.json \
    --dataset robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5 \
    --debug
```

Formal training:

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robot_arm/robomimic_pick_place/launch_train.py \
    --config robot_arm/robomimic_pick_place/configs/bc_pixel_pick_place.json \
    --dataset robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5
```

Evaluate a checkpoint:

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robot_arm/robomimic_pick_place/launch_eval.py \
    --agent /path/to/model.pth \
    --n_rollouts 100 \
    --horizon 150 \
    --video_path /tmp/pick_place_bc.mp4 \
    --camera_names xy xz yz
```

### 2.2 Implicit Q-Learning (IQL)

Configuration:

```text
robot_arm/robomimic_pick_place/configs/iql_pixel_pick_place.json
```

This file has the same complete field hierarchy as the Lift visual IQL
configuration. The main IQL-specific requirement is:

```json
"hdf5_load_next_obs": true
```

IQL needs `obs`, `action`, `reward`, `next_obs`, and `done` to build its TD
target. The converted custom dataset already contains all of these fields, so
no additional converter or environment change is needed.

The custom IQL values are:

| Setting | Value | Reason |
| --- | --- | --- |
| `obs.low_dim` | `proprio` | Custom proprioception representation. |
| `obs.rgb` | `xy_image`, `xz_image`, `yz_image` | Custom camera views. |
| `hdf5_load_next_obs` | `true` | Required by IQL. |
| `hdf5_filter_key` / validation key | `train` / `valid` | Converter-provided episode masks. |
| `rollout.horizon` | `150` | Custom environment horizon. |
| actor Gaussian | fixed std, `init_std=0.1` | Stable small-batch starting point. |
| advantage | `clip_adv_value=5.0`, `beta=1.0` | Stable IQL baseline. |
| `batch_size` | `2` | IQL maintains actor, value, two critics, and two target critics with three image streams. |

Smoke test:

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robot_arm/robomimic_pick_place/launch_train.py \
    --config robot_arm/robomimic_pick_place/configs/iql_pixel_pick_place.json \
    --dataset robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5 \
    --debug
```

Formal training:

```bash
MUJOCO_GL=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
python robot_arm/robomimic_pick_place/launch_train.py \
    --config robot_arm/robomimic_pick_place/configs/iql_pixel_pick_place.json \
    --dataset robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5
```

Evaluation uses the same `launch_eval.py` command as BC. The checkpoint stores
the algorithm, observation shapes, and custom environment metadata.

### 2.3 Other robomimic algorithms

BC and IQL are examples, not limits of the adapter. Other algorithms can reuse
the same HDF5 dataset and environment launcher when their data requirements are
met:

```text
BC-RNN       requires sequence configuration and temporal policy inputs
BCQ/CQL      require actions, rewards, next_obs, and dones
TD3-BC       requires offline transitions and next_obs
Diffusion    requires the fields expected by its sequence configuration
```

The reusable integration boundary is:

```text
custom environment adapter + robomimic HDF5 dataset
    -> algorithm-specific config
    -> same launcher, checkpoint, TensorBoard, and rollout path
```

For every new algorithm, check its config defaults and
`process_batch_for_training` implementation before deciding whether `next_obs`,
longer sequences, frame stacking, or additional dataset fields are required.
