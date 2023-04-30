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
        self._agent_ids = set(range(self.num_uavs))

        self.env_max_w = env_config.get("env_max_w", 2.5)
        self.env_max_l = env_config.get("env_max_l", 2.5)
        self.env_max_h = env_config.get("env_max_h", 2.5)
        # self.env_max_w = env_config.get("env_max_w", 10)
        # self.env_max_l = env_config.get("env_max_l", 10)
        # self.env_max_h = env_config.get("env_max_h", 10)

        self.gui = None
        self._time_elapsed = 0

        self.reset()

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def step(self, actions):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

        for i, action in actions.items():
            self.uavs[i].step(action)

        self._time_elapsed += self.dt

        return obs, rew, terminated, truncated, info

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

        for idx in range(self.num_uavs):
            x = np.random.rand() * self.env_max_w
            y = np.random.rand() * self.env_max_l
            z = np.random.rand() * self.env_max_h

            uav = Quadrotor(_id=idx, x=x, y=y, z=z, use_ode=True)
            self.uavs.append(uav)

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = Gui(
                    self.uavs,
                    max_x=self.env_max_w,
                    max_y=self.env_max_l,
                    max_z=self.env_max_h,
                )
            else:
                self.gui.update(self.time_elapsed)

    def close_gui(self):
        if self.gui is None:
            self.gui.close()
        self.gui = None
