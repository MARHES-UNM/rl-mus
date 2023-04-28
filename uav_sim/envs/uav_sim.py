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
        # self.env_max_x = env_config.get("max_x", )
        self.fig = None
        self.clock = None
        self.gui = None
        self._time_elapsed = 0

        self.reset()

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def step(self, actions):
        for uav in self.uavs:
            action = np.random.rand(4) * (uav.max_f - uav.min_f) + uav.min_f
            action = np.ones(4) * uav.m * uav.g / 4

            uav.step(action)

        self._time_elapsed += self.dt

    def seed(self, seed=None):
        """Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """_summary_

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        """
        if self.gui is not None:
            self.close_gui()

        self.seed(seed)
        self.uavs = []

        uav_pos = [[2, 4, 5], [1, 1, 1], [1, 1, 2], [2, 1, 3]]
        for pos in uav_pos:
            uav = Quadrotor(*pos)
            self.uavs.append(uav)
        # for idx in range(self.num_uavs):
        #     # x = np.random.rand()
        #     # y =
        #     # z =
        #     self.uavs.append(uav)]

        # self.uav = Quadrotor(2, 4, 5)

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = Gui(self.uavs)
            else:
                self.gui.update(self.time_elapsed)

    def close_gui(self):
        if self.gui is None:
            self.gui.close()
        self.gui = None
