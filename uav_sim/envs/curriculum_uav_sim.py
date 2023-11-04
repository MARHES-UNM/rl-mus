from logging import config
import gymnasium as gym
import random

from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from uav_sim.envs.uav_sim import UavSim


class CurriculumEnv(TaskSettableEnv):
    def __init__(self, config: EnvContext):
        self.config = config
        self.cur_level = self.config.get("start_level", 1)
        self.env = None

        self.env_difficulty_config = self.config["difficulty_config"]
        # self.env_difficulty_config = {
        #     1: {
        #         "beta": 1.0,
        #         "stp_penalty": 0,
        #         "tgt_reward": 100.0,
        #         "obstacle_collision": 0,
        #     },
        #     2: {
        #         "beta": 0.1,
        #         "stp_penalty": 5,
        #         "tgt_reward": 100.0,
        #         "obstacle_collision": 0,
        #     },
        #     3: {
        #         "beta": 0.001,
        #         "stp_penalty": 5,
        #         "tgt_reward": 200.0,
        #         "obstacle_collision": 0.1,
        #     },
        # }
        self.num_tasks = len(self.env_difficulty_config)
        self._make_uav_sim()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False

    def reset(self, *, seed=None, options=None):
        if self.switch_env:
            self._make_uav_sim()
            self.switch_env = False
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        return obs, rew, done, truncated, info

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, self.num_tasks) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
        self.switch_env = True

    def _make_uav_sim(self):
        level_config = self.env_difficulty_config[self.cur_level]
        env_config = self.config.copy()
        env_config.update(level_config)
        self.env = UavSim(env_config)
