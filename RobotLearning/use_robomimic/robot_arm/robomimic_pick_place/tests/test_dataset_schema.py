from pathlib import Path

import h5py


def test_converted_dataset_schema():
    path = Path("robot_arm/robomimic_pick_place/data/pick_place_expert.hdf5")
    if not path.exists():
        return
    with h5py.File(path, "r") as handle:
        demo = sorted(handle["data"].keys())[0]
        assert set(handle["data"][demo]["obs"].keys()) == {
            "xy_image", "xz_image", "yz_image", "proprio"
        }
        assert set(handle["data"][demo]["obs"].keys()) == set(handle["data"][demo]["next_obs"].keys())
