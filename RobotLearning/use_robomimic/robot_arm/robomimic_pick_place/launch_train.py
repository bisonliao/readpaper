"""Register the custom environment, then delegate to robomimic train.py.

Supports both ``python -m robot_arm.robomimic_pick_place.launch_train`` and
direct execution via ``python robot_arm/robomimic_pick_place/launch_train.py``.
"""

import runpy
import sys
from pathlib import Path

if __package__:
    from .env_registry import register_custom_environment
else:  # direct script execution has no package context
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robot_arm.robomimic_pick_place.env_registry import register_custom_environment


if __name__ == "__main__":
    register_custom_environment()
    runpy.run_module("robomimic.scripts.train", run_name="__main__")
