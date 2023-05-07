import numpy as np

from gym.utils import seeding
from uav_sim.agents.uav import Quadrotor
from uav_sim.agents.uav import Target
from uav_sim.utils.gui import Gui
import logging

logger = logging.getLogger(__name__)


class UavSim:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}):
        self.dt = env_config.get("dt", 0.1)
        self._seed = env_config.get("seed", None)
        self.render_mode = env_config.get("render_mode", "human")
        self.num_uavs = env_config.get("num_uavs", 4)
        self._agent_ids = set(range(self.num_uavs))

        self.env_max_w = env_config.get("env_max_w", 4)
        self.env_max_l = env_config.get("env_max_l", 4)
        self.env_max_h = env_config.get("env_max_h", 4)
        self.target_v = env_config.get("target_v", 0)
        self.target_w = env_config.get("target_w", 0)
        self.max_time = env_config.get("max_time", 40)
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
        for i, action in actions.items():
            self.uavs[i].step(action)

        self.target.step(np.array([self.target_v, self.target_w]))

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs}
        done = self._get_done()
        info = self._get_info()
        self._time_elapsed += self.dt

        # newer API to return truncated
        # return obs, reward, done, self.time_elapsed >= self.max_time, info
        return obs, reward, done, info

    def _get_info(self):
        pass

    def _get_obs(self):
        pass

    def _get_reward(self):
        pass

    def _get_done(self):
        done = {uav.id: uav.done for uav in self.uavs}

        # Done when Target is reached for all uavs
        done["__all__"] = all(done) or self.time_elapsed >= self.max_time

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
        x = np.random.rand() * self.env_max_w
        y = np.random.rand() * self.env_max_l
        self.target = Target(
            _id=0, x=x, y=y, dt=self.dt, num_landing_pads=self.num_uavs
        )

        # Reset UAVs
        self.uavs = []
        for idx in range(self.num_uavs):
            x = np.random.rand() * self.env_max_w
            y = np.random.rand() * self.env_max_l
            z = np.random.rand() * self.env_max_h

            uav = Quadrotor(_id=idx, x=x, y=y, z=z, dt=self.dt)
            self.uavs.append(uav)

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs}

        return obs

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = Gui(
                    self.uavs,
                    target=self.target,
                    max_x=self.env_max_w,
                    max_y=self.env_max_l,
                    max_z=self.env_max_h,
                )
            else:
                self.gui.update(self.time_elapsed)

    def close_gui(self):
        if self.gui is not None:
            self.gui.close()
        self.gui = None
