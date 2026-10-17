from pathlib import Path

import numpy as np


def test_first_expert_episode_format():
    paths = sorted(Path("robot_arm/expert_showcase_v4").glob("episode_*.npz"))
    assert paths, "expert showcase is not available"
    data = np.load(paths[0], allow_pickle=True)
    assert data["image"].ndim == 4
    assert data["image"].shape[1] % 9 == 0
    assert data["proprio"].shape[1] == 26
    assert data["action"].shape[1] == 4
    assert len(data["image"]) == len(data["action"]) == len(data["reward"])
