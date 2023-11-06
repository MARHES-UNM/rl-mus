from logging import config
import gymnasium as gym
import random

from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from uav_sim.envs.uav_sim import UavSim


class CurriculumEnv(MultiAgentEnv, TaskSettableEnv):
    """
        https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/curriculum_learning.py
        https://www.oreilly.com/library/view/learning-ray/9781098117214/ch04.html
    Args:
        MultiAgentEnv (_type_): _description_
        TaskSettableEnv (_type_): _description_
    """

    def __init__(self, config: EnvContext):
        MultiAgentEnv.__init__(self)
        self.config = config
        self.cur_level = self.config.get("start_level", 1)
        self.env = None

        # self.env_difficulty_config = self.config["difficulty_config"]
        # self.num_tasks = len(self.env_difficulty_config)
        self.num_tasks = self.config["env_max_h"]
        self._make_uav_sim()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = self.env._agent_ids
        self.switch_env = False

    def reset(self, *, seed=None, options=None):
        if self.switch_env:
            self._make_uav_sim()
            self.switch_env = False
        self._agent_ids = self.env._agent_ids
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
        # level_config = self.env_difficulty_config[self.cur_level]
        env_config = self.config.copy()
        env_config["z_high"] = self.cur_level + 1
        # env_config.update(level_config)
        self.env = UavSim(env_config)
