import numpy as np

from robot_arm.robomimic_pick_place.env_adapter import RobomimicPixelPickPlaceEnv


def test_adapter_reset_step_and_state():
    env = RobomimicPixelPickPlaceEnv(seed=0)
    try:
        obs = env.reset()
        assert set(obs) == {"xy_image", "xz_image", "yz_image", "proprio"}
        assert obs["xy_image"].shape == (96, 96, 3)
        assert obs["proprio"].shape == (26,)
        state = env.get_state()
        next_obs, reward, done, info = env.step(np.zeros(4, dtype=np.float32))
        assert next_obs["yz_image"].shape == (96, 96, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "success" in info
        restored = env.reset_to(state)
        assert restored["xy_image"].shape == (96, 96, 3)
    finally:
        env.close()
