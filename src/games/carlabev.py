import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs import CarlaBEV
from src.utils import rgb_to_semantic_mask


class SemanticMaskWrapper(gym.ObservationWrapper):
    """
    A Gym wrapper to convert RGB observations into semantic masks.

    This wrapper assumes the environment's observation is an RGB image
    and converts it into a 6-channel semantic mask using the rgb_to_semantic_mask function.
    """

    def __init__(self, env):
        super(SemanticMaskWrapper, self).__init__(env)
        # Update the observation space to reflect the semantic mask shape
        obs_shape = self.observation_space.shape
        h, w = obs_shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6, h, w), dtype=np.float32
        )

    def observation(self, observation):
        """
        Convert the RGB observation into a semantic mask.

        Args:
            observation (np.ndarray): (H, W, 3) RGB image.

        Returns:
            np.ndarray: (6, H, W) semantic mask.
        """
        return rgb_to_semantic_mask(observation)


def make_carlabev_env(seed, idx, capture_video, run_name, size):
    def thunk():
        if capture_video and idx == 0:
            env = CarlaBEV(render_mode="rgb_array", size=size)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 10 == 0
            )
        else:
            env = CarlaBEV(render_mode="rgb_array", size=size)

        env = ResizeObservation(env, (64, 64))
        env = SemanticMaskWrapper(env)
        env = FrameStackObservation(env, stack_size=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


def make_env(seed, capture_video, run_name, size):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_env(
                seed=seed + i,
                idx=i,
                capture_video=capture_video,
                run_name=run_name,
                size=size,
            )
            for i in range(1)
        ]
    )

    return envs
