"""Convert PixelTaskEnv expert NPZ episodes into a robomimic HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .env_adapter import ENV_NAME, ENV_VERSION, CUSTOM_ENV_TYPE


VIEW_NAMES = ("xy", "xz", "yz")
RGB_CHANNELS = 3
FRAME_CHANNELS = len(VIEW_NAMES) * RGB_CHANNELS


def _scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def _load_episode(path: Path):
    data = np.load(path, allow_pickle=True)
    required = {"image", "proprio", "action", "reward", "terminated", "truncated", "success"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    image = np.asarray(data["image"])
    proprio = np.asarray(data["proprio"])
    action = np.asarray(data["action"])
    reward = np.asarray(data["reward"], dtype=np.float32)
    terminated = np.asarray(data["terminated"], dtype=bool)
    truncated = np.asarray(data["truncated"], dtype=bool)
    success = bool(_scalar(data["success"]))
    if image.ndim != 4 or image.shape[1] < FRAME_CHANNELS or image.shape[1] % FRAME_CHANNELS != 0:
        raise ValueError(f"{path}: expected image shape (T, 9*k, H, W), got {image.shape}")
    length = image.shape[0]
    if any(len(x) != length for x in (proprio, action, reward, terminated, truncated)):
        raise ValueError(f"{path}: obs/action/reward/done lengths disagree")
    if proprio.shape[1:] != (26,):
        raise ValueError(f"{path}: expected proprio shape (T, 26), got {proprio.shape}")
    if action.shape[1:] != (4,):
        raise ValueError(f"{path}: expected action shape (T, 4), got {action.shape}")

    # The expert recorder stores the most recent frame stack in channel order.
    # The latest frame is the final 9 channels; each view becomes one HWC RGB key.
    latest = image[:, -FRAME_CHANNELS:]
    view_images = {}
    for index, view in enumerate(VIEW_NAMES):
        channels = latest[:, index * RGB_CHANNELS : (index + 1) * RGB_CHANNELS]
        view_images[f"{view}_image"] = np.transpose(channels, (0, 2, 3, 1)).copy()

    dones = np.logical_or(terminated, truncated)
    # Every HDF5 demo is an episode boundary. Ensure the final transition is
    # terminal even if an externally produced recorder omitted the final flag.
    dones[-1] = True
    obs = {**view_images, "proprio": proprio.astype(np.float32, copy=True)}
    next_obs = {key: value[1:].copy() for key, value in obs.items()}
    for key, value in obs.items():
        next_obs[key] = np.concatenate([next_obs[key], value[-1:]], axis=0)
    metadata = {}
    if "episode_metadata" in data.files:
        metadata = json.loads(str(_scalar(data["episode_metadata"])))
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": action.astype(np.float32, copy=True),
        "rewards": reward,
        "dones": dones.astype(np.uint8),
        "success": success,
        "metadata": metadata,
    }


def _write_mask(group, name, episodes):
    group.create_dataset(name, data=np.asarray(episodes, dtype="S"))


def convert(input_dir: Path, output_path: Path, success_only: bool, train_ratio: float, seed: int):
    files = sorted(input_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz files found under {input_dir}")
    loaded = []
    for path in files:
        episode = _load_episode(path)
        episode["source"] = path.name
        loaded.append(episode)

    rng = np.random.default_rng(seed)
    candidate = [i for i, ep in enumerate(loaded) if ep["success"]] if success_only else list(range(len(loaded)))
    if len(candidate) < 2:
        raise ValueError("need at least two candidate episodes for train/valid split")
    order = rng.permutation(candidate)
    split = min(max(1, int(round(len(order) * train_ratio))), len(order) - 1)
    train_ids = order[:split].tolist()
    valid_ids = order[split:].tolist()
    all_ids = list(range(len(loaded)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_args = {
        "env_name": ENV_NAME,
        "env_version": ENV_VERSION,
        "type": CUSTOM_ENV_TYPE,
        "env_kwargs": {
            "task": "pick_place",
            "image_size": 96,
            "frame_stack": 1,
            "max_episode_steps": 150,
            "action_repeat": 8,
            "camera_scale": 1.0,
            "seed": None,
        },
    }
    with h5py.File(output_path, "w") as handle:
        data_group = handle.create_group("data")
        masks = handle.create_group("mask")
        total = 0
        for index, episode in enumerate(loaded):
            demo = data_group.create_group(f"demo_{index:06d}")
            for key, value in episode["obs"].items():
                demo.create_dataset(f"obs/{key}", data=value, compression="gzip")
            for key, value in episode["next_obs"].items():
                demo.create_dataset(f"next_obs/{key}", data=value, compression="gzip")
            demo.create_dataset("actions", data=episode["actions"])
            demo.create_dataset("rewards", data=episode["rewards"])
            demo.create_dataset("dones", data=episode["dones"])
            demo.attrs["num_samples"] = len(episode["actions"])
            demo.attrs["success"] = episode["success"]
            demo.attrs["source_file"] = episode["source"]
            demo.attrs["episode_metadata"] = json.dumps(episode["metadata"], sort_keys=True)
            total += len(episode["actions"])

        data_group.attrs["total"] = total
        data_group.attrs["env_args"] = json.dumps(env_args, sort_keys=True)
        _write_mask(masks, "all", [f"demo_{i:06d}" for i in all_ids])
        _write_mask(masks, "train", [f"demo_{i:06d}" for i in train_ids])
        _write_mask(masks, "valid", [f"demo_{i:06d}" for i in valid_ids])
        success_ids = [i for i, ep in enumerate(loaded) if ep["success"]]
        _write_mask(masks, "success", [f"demo_{i:06d}" for i in success_ids])
        _write_mask(masks, "train_success", [f"demo_{i:06d}" for i in train_ids if loaded[i]["success"]])
        _write_mask(masks, "valid_success", [f"demo_{i:06d}" for i in valid_ids if loaded[i]["success"]])
    print(f"wrote {len(loaded)} episodes and {total} transitions to {output_path}")
    print(f"success episodes: {sum(ep['success'] for ep in loaded)} / {len(loaded)}")
    print(f"train episodes: {len(train_ids)}, valid episodes: {len(valid_ids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    convert(args.input, args.output, args.success_only, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
