import numpy as np

from gym.utils import seeding
from uav_sim.agents.uav import Quadrotor
from uav_sim.utils.gui import Gui


class UavSim:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}):
        self.dt = env_config.get("dt", 0.01)
        self._seed = env_config.get("seed", None)
        self.render_mode = env_config.get("render_mode", "human")
        self.num_uavs = env_config.get("num_uavs", 4)
        self.fig = None
        self.clock = None
        self.gui = None
        self._time_elapsed = 0

        self.reset()

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def step(self, actions):
        action = np.random.rand(4) * (self.uav.max_f - self.uav.min_f) + self.uav.min_f
        action = np.ones(4) * self.uav.m * self.uav.g / 4

        self.uav.step(action)

        self._time_elapsed += self.dt

    def seed(self, seed=None):
        """Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.seed(seed)

        self.uav = Quadrotor(2, 4, 5)

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = Gui(self.uav)
            else:
                self.gui.update(self.time_elapsed)
