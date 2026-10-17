"""robomimic EnvBase adapter for RobotEnv.pixel_env.PixelTaskEnv.

The adapter keeps the custom Gymnasium environment unchanged and translates its
five-value reset / step API into the API expected by robomimic rollouts.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from typing import Any, Dict, Optional

import numpy as np
import pybullet as p

import robomimic.envs.env_base as EB

from robot_arm.RobotEnv.pixel_env import (
    CAMERA_VIEW_NAMES,
    DEFAULT_CAMERA_SCALE,
    DEFAULT_IMAGE_SIZE,
    PixelTaskEnv,
)


CUSTOM_ENV_TYPE = 4
ENV_NAME = "custom_pick_place"
ENV_VERSION = "pixel_v1"


class RobomimicPixelPickPlaceEnv(EB.EnvBase):
    """Expose ``PixelTaskEnv`` through robomimic's environment contract."""

    def __init__(
        self,
        env_name: str = ENV_NAME,
        render: bool = False,
        render_offscreen: bool = False,
        use_image_obs: bool = True,
        use_depth_obs: bool = False,
        task: str = "pick_place",
        image_size: int = DEFAULT_IMAGE_SIZE,
        frame_stack: int = 1,
        max_episode_steps: int = 150,
        action_repeat: int = 8,
        camera_scale: float = DEFAULT_CAMERA_SCALE,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        del render_offscreen, use_image_obs, use_depth_obs, kwargs
        if task != "pick_place":
            raise ValueError("RobomimicPixelPickPlaceEnv only supports task='pick_place'")
        if render and render_mode is None:
            render_mode = "human"
        self._env_name = env_name
        self._seed = seed
        self._init_kwargs = dict(
            task=task,
            image_size=int(image_size),
            frame_stack=int(frame_stack),
            max_episode_steps=int(max_episode_steps),
            action_repeat=int(action_repeat),
            camera_scale=float(camera_scale),
            seed=seed,
            render_mode=render_mode,
        )
        self.env = PixelTaskEnv(**self._init_kwargs)
        self._current_obs: Optional[Dict[str, np.ndarray]] = None
        self._current_reward: Optional[float] = None
        self._current_done: Optional[bool] = None
        self._current_info: Dict[str, Any] = {}

    def _split_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = np.asarray(obs["image"], dtype=np.uint8)
        expected_channels = len(CAMERA_VIEW_NAMES) * 3
        if image.ndim != 3 or image.shape[0] != expected_channels:
            raise ValueError(
                f"expected one 3-channel frame with {expected_channels} channels, got {image.shape}"
            )
        views = image.reshape(len(CAMERA_VIEW_NAMES), 3, image.shape[1], image.shape[2])
        result = {
            f"{name}_image": np.transpose(view, (1, 2, 0)).copy()
            for name, view in zip(CAMERA_VIEW_NAMES, views)
        }
        result["proprio"] = np.asarray(obs["proprio"], dtype=np.float32).copy()
        return result

    def _join_render_views(self) -> np.ndarray:
        if self._current_obs is None:
            self.reset()
        assert self._current_obs is not None
        frames = [self._current_obs[f"{name}_image"] for name in CAMERA_VIEW_NAMES]
        return np.concatenate(frames, axis=1)

    def reset(self):
        obs, info = self.env.reset(seed=self._seed)
        self._seed = None
        self._current_obs = self._split_observation(obs)
        self._current_reward = None
        self._current_done = False
        self._current_info = dict(info)
        return deepcopy(self._current_obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._current_obs = self._split_observation(obs)
        self._current_reward = float(reward)
        self._current_done = bool(terminated or truncated)
        self._current_info = dict(info)
        self._current_info["terminated"] = bool(terminated)
        self._current_info["truncated"] = bool(truncated)
        return deepcopy(self._current_obs), float(reward), self._current_done, dict(self._current_info)

    def get_observation(self, obs=None):
        if obs is None:
            if self._current_obs is None:
                self.reset()
            obs = self._current_obs
        return deepcopy(obs)

    def get_reward(self):
        if self._current_reward is None:
            raise RuntimeError("get_reward called before a transition")
        return self._current_reward

    def is_done(self):
        return bool(self._current_done)

    def is_success(self):
        info = self._current_info
        return {
            "task": bool(info.get("success", info.get("is_success", False))),
            "grasped": bool(info.get("ever_grasped", False)),
            "lifted": bool(info.get("ever_lifted", False)),
        }

    def get_goal(self):
        raise NotImplementedError("pick-place BC is not goal-conditioned")

    def set_goal(self, **kwargs):
        raise NotImplementedError("pick-place BC is not goal-conditioned")

    def render(self, mode="human", height=None, width=None, camera_name=None):
        del height, width, camera_name
        if mode == "human":
            return self.env.render()
        if mode == "rgb_array":
            return self._join_render_views()
        raise NotImplementedError(f"unsupported render mode: {mode}")

    def get_state(self):
        """Capture a same-process PyBullet snapshot for rollout reset_to."""
        client = self.env.task_env.base_env._client
        bullet_state = p.saveState(physicsClientId=client)
        base_env = self.env.task_env.base_env
        joint_states = p.getJointStates(
            base_env.robot_id,
            list(range(p.getNumJoints(base_env.robot_id, physicsClientId=client))),
            physicsClientId=client,
        )
        object_position, object_orientation = p.getBasePositionAndOrientation(
            base_env.object_id, physicsClientId=client
        )
        goal_position, goal_orientation = p.getBasePositionAndOrientation(
            base_env.goal_id, physicsClientId=client
        )
        # run_trained_agent.py stores a flattened "states" field in optional
        # rollout datasets. It is diagnostic only; reset_to uses bullet_state.
        flat_states = np.concatenate(
            [
                np.asarray([s[0] for s in joint_states], dtype=np.float32),
                np.asarray([s[1] for s in joint_states], dtype=np.float32),
                np.asarray(object_position, dtype=np.float32),
                np.asarray(object_orientation, dtype=np.float32),
                np.asarray(goal_position, dtype=np.float32),
                np.asarray(goal_orientation, dtype=np.float32),
            ]
        )
        task_env = self.env.task_env
        return {
            "bullet_state": int(bullet_state),
            "states": flat_states,
            "stage": int(task_env.stage),
            "ever_grasped": bool(task_env.ever_grasped),
            "ever_lifted": bool(task_env.ever_lifted),
            "grasp_bonus_given": bool(task_env.grasp_bonus_given),
            "lift_bonus_given": bool(task_env.lift_bonus_given),
            "place_bonus_given": bool(task_env.place_bonus_given),
            "release_bonus_given": bool(task_env.release_bonus_given),
            "stable_steps": int(task_env.stable_steps),
            "stage_steps": int(task_env._stage_steps),
            "contact_steps": int(task_env._contact_steps),
            "violation_steps": int(task_env._violation_steps),
            "premature_release_steps": int(task_env._premature_release_steps),
            "failure_reason": str(task_env._failure_reason),
            "last_reward_terms": deepcopy(task_env._last_reward_terms),
            "last_action": self.env._last_action.copy(),
            "last_gripper_width": float(self.env._last_gripper_width),
            "current_reward": self._current_reward,
            "current_done": self._current_done,
            "current_info": deepcopy(self._current_info),
        }

    def reset_to(self, state):
        client = self.env.task_env.base_env._client
        p.restoreState(stateId=int(state["bullet_state"]), physicsClientId=client)
        task_env = self.env.task_env
        task_env.stage = type(task_env.stage)(int(state["stage"]))
        for key in (
            "ever_grasped", "ever_lifted", "grasp_bonus_given", "lift_bonus_given",
            "place_bonus_given", "release_bonus_given",
        ):
            setattr(task_env, key, bool(state[key]))
        for key in (
            "stable_steps", "stage_steps", "contact_steps", "violation_steps",
            "premature_release_steps",
        ):
            setattr(task_env, f"_{key}" if key.endswith("steps") and key != "stable_steps" else key, int(state[key]))
        task_env._failure_reason = str(state["failure_reason"])
        task_env._last_reward_terms = deepcopy(state["last_reward_terms"])
        self.env._last_action = np.asarray(state["last_action"], dtype=np.float32).copy()
        self.env._last_gripper_width = np.float32(state["last_gripper_width"])
        self._current_reward = state["current_reward"]
        self._current_done = state["current_done"]
        self._current_info = deepcopy(state["current_info"])
        frame = self.env._render_views()
        self.env._frames.clear()
        for _ in range(self.env.frame_stack):
            self.env._frames.append(frame.copy())
        raw_obs = self.env._make_observation()
        self._current_obs = self._split_observation(raw_obs)
        return deepcopy(self._current_obs)

    @property
    def action_dimension(self):
        return int(self.env.action_space.shape[0])

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return CUSTOM_ENV_TYPE

    @property
    def version(self):
        return ENV_VERSION

    def serialize(self):
        return {
            "env_name": self.name,
            "env_version": self.version,
            "type": self.type,
            "env_kwargs": deepcopy(self._init_kwargs),
        }

    @classmethod
    def create_for_data_processing(cls, **kwargs):
        return cls(**kwargs)

    @property
    def rollout_exceptions(self):
        return ()

    @property
    def base_env(self):
        return self.env

    def close(self):
        self.env.close()

    def __repr__(self):
        return self.name + "\n" + json.dumps(self.serialize(), sort_keys=True, indent=2)
