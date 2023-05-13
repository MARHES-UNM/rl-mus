import numpy as np

from gym.utils import seeding
from uav_sim.agents.uav import Obstacle, Quad2DInt, Quadrotor
from uav_sim.agents.uav import Target
from uav_sim.utils.gui import Gui
from qpsolvers import solve_qp
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
        self.gamma = env_config.get("gamma", 1)
        self.num_obstacles = env_config.get("num_obstacles", 4)

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
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.reset()

    def _get_action_space(self):
        pass

    def _get_observation_space(self):
        pass

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def get_h(self, uav, entity):
        del_p = uav.pos - entity.pos
        del_v = uav.vel - entity.vel

        h = np.linalg.norm(del_p) - (uav.r + entity.r)
        h = np.sqrt(h)
        h += (del_p.T @ del_v) / np.linalg.norm(del_p)

        return h

    def calc_b(self, uav, entity):
        del_p = uav.pos - entity.pos
        del_v = uav.vel - entity.vel

        h = self.get_h(uav, entity)

        b = self.gamma * h**3 * np.linalg.norm(del_p)
        b -= ((del_v.T @ del_p) ** 2) / ((np.linalg.norm(del_p)) ** 2)
        b += (del_v.T @ del_p) / (np.sqrt(np.linalg.norm(del_p) - (uav.r + entity.r)))
        b += np.linalg.norm(del_v) ** 2
        return b

    def proj_safe_action(self, uav, des_action):
        G = []
        h = []
        P = np.eye(3)
        q = -np.dot(P.T, des_action)

        # other agents
        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                G.append(uav.pos - other_uav.pos)
                b = self.calc_b(uav, other_uav)
                h.append(b)

        for obstacle in self.obstacles:
            G.append(uav.pos - obstacle.pos)
            b = self.calc_b(uav, obstacle)
            h.append(b)

        G = np.array(G)
        h = np.array(h)

        if G.any() and h.any():
            try:
                u_out = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                u_out = des_action
        else:
            print("not running qpsolver")
            return des_action

        if u_out is None:
            print("infeasible sovler")
            return des_action

        return u_out

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

    def _get_obs(self, uav):
        other_uav_states = []

        for other_uav in self.uavs:
            if uav.id != other_uav.id:
                other_uav_states.append(other_uav.state)

        other_uav_states = np.array(other_uav_states)

        landing_pads = []
        for pad in self.target.pads:
            landing_pads.append(pad.state)

        landing_pads = np.array(landing_pads)

        obstacles = np.array([])

        obs_dict = {
            "state": uav.state.astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "landing_pads": landing_pads.astype(np.float32),
            "obstacles": obstacles.astype(np.float32),
        }

        return obs_dict

    def _get_reward(self, uav):
        if uav.done:
            return 0

        reward = 0
        # pos reward if uav lands on any landing pad
        for pad in self.target.pads:
            if uav.get_landed(pad):
                uav.done = True
                reward += 1
                break

        # neg reward if uav collides with other uavs
        # neg reward if uav collides with obstacles
        return reward

    def _get_done(self):
        """Outputs if sim is done based on entity's states.
        Must calculate _get_reward first.
        """
        done = {uav.id: uav.done for uav in self.uavs}

        # Done when Target is reached for all uavs
        done["__all__"] = (
            all(val for val in done.values()) or self.time_elapsed >= self.max_time
        )
        return done

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

        # Reset Target
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

            uav = Quad2DInt(_id=idx, x=x, y=y, z=z, dt=self.dt)
            self.uavs.append(uav)

        # Reset obstacles
        self.obstacles = []
        for idx in range(self.num_obstacles):
            x = np.random.rand() * self.env_max_l
            y = np.random.rand() * self.env_max_l
            z = np.random.rand() * self.env_max_h

            obstacle = Obstacle(_id=idx, x=x, y=y, z=z)
            self.obstacles.append(obstacle)

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs}

        return obs

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.gui is None:
                self.gui = Gui(
                    self.uavs,
                    target=self.target,
                    obstacles=self.obstacles,
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
