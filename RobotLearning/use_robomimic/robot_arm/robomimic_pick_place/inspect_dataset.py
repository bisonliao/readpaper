"""Inspect a converted custom PickPlace HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def inspect(path: Path):
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"].keys())
        if not demos:
            raise ValueError("dataset contains no demonstrations")
        first = handle["data"][demos[0]]
        expected_obs = {"xy_image", "xz_image", "yz_image", "proprio"}
        if set(first["obs"].keys()) != expected_obs:
            raise ValueError(f"unexpected obs keys: {list(first['obs'].keys())}")
        total = 0
        successes = 0
        for demo_name in demos:
            demo = handle["data"][demo_name]
            length = int(demo.attrs["num_samples"])
            total += length
            successes += int(bool(demo.attrs.get("success", False)))
            if set(demo["obs"].keys()) != set(demo["next_obs"].keys()):
                raise ValueError(f"{demo_name}: obs and next_obs keys differ")
            if any(len(demo["obs"][key]) != length for key in expected_obs):
                raise ValueError(f"{demo_name}: observation lengths disagree")
            if len(demo["actions"]) != length or len(demo["rewards"]) != length or len(demo["dones"]) != length:
                raise ValueError(f"{demo_name}: transition lengths disagree")
        env_args = json.loads(handle["data"].attrs["env_args"])
        print("episodes:", len(demos))
        print("transitions:", total)
        print("success episodes:", successes)
        print("env:", json.dumps(env_args, indent=2, sort_keys=True))
        print("obs shapes:")
        for key in sorted(expected_obs):
            dataset = first["obs"][key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        print("masks:", sorted(handle["mask"].keys()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.dataset)


if __name__ == "__main__":
    main()
