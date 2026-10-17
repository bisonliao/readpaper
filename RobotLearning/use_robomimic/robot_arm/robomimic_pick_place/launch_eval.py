"""Register the custom environment, then delegate to run_trained_agent.py.

Supports both module execution and direct execution by file path.
"""

import runpy
import sys
from pathlib import Path

if __package__:
    from .env_registry import register_custom_environment
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robot_arm.robomimic_pick_place.env_registry import register_custom_environment


if __name__ == "__main__":
    register_custom_environment()
    runpy.run_module("robomimic.scripts.run_trained_agent", run_name="__main__")
