"""Runtime registration for the custom robomimic environment type."""

from robomimic.utils import env_utils

from .env_adapter import CUSTOM_ENV_TYPE, RobomimicPixelPickPlaceEnv


_ORIGINAL_GET_ENV_CLASS = env_utils.get_env_class


def register_custom_environment():
    """Patch env_utils in-process without modifying robomimic source files."""

    if getattr(env_utils, "_robot_arm_pick_place_registered", False):
        return

    def get_env_class(env_meta=None, env_type=None, env=None):
        resolved_type = env_utils.get_env_type(env_meta=env_meta, env_type=env_type, env=env)
        if resolved_type == CUSTOM_ENV_TYPE:
            return RobomimicPixelPickPlaceEnv
        return _ORIGINAL_GET_ENV_CLASS(env_meta=env_meta, env_type=env_type, env=env)

    env_utils.get_env_class = get_env_class
    env_utils._robot_arm_pick_place_registered = True


__all__ = ["register_custom_environment"]
