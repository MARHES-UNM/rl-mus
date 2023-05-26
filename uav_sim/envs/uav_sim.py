from math import isclose
import numpy as np

from gym.utils import seeding
from _archives.full_uav import ObsType
from uav_sim.agents.uav import Obstacle, Quad2DInt, Quadrotor
from uav_sim.agents.uav import Target
from uav_sim.utils.gui import Gui
from qpsolvers import solve_qp
import logging
import random

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
        self.obstacle_collision_weight = env_config.get("obstacle_collision_weight", 1)
        self.uav_collision_weight = env_config.get("uav_collision_weight", 1)
        self._use_safe_action = env_config.get("use_safe_action", False)

        self._agent_ids = set(range(self.num_uavs))

        self.env_max_w = env_config.get("env_max_w", 4)
        self.env_max_l = env_config.get("env_max_l", 4)
        self.env_max_h = env_config.get("env_max_h", 4)
        self.target_v = env_config.get("target_v", 0)
        self.target_w = env_config.get("target_w", 0)
        self.max_time = env_config.get("max_time", 40)

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

    def get_b(self, uav, entity):
        del_p = uav.pos - entity.pos
        del_v = uav.vel - entity.vel

        h = self.get_h(uav, entity)

        b = self.gamma * h**3 * np.linalg.norm(del_p)
        b -= ((del_v.T @ del_p) ** 2) / ((np.linalg.norm(del_p)) ** 2)
        b += (del_v.T @ del_p) / (np.sqrt(np.linalg.norm(del_p) - (uav.r + entity.r)))
        b += np.linalg.norm(del_v) ** 2
        return b

    def get_safe_action(self, uav, des_action):
        G = []
        h = []
        P = np.eye(3)
        u_in = des_action.copy()
        # u_in[2] = 1 / uav.m * u_in[2] - uav.g
        q = -np.dot(P.T, u_in)

        # other agents
        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                G.append(-(uav.pos - other_uav.pos).T)
                b = self.get_b(uav, other_uav)
                h.append(b)

        for obstacle in self.obstacles:
            G.append(-(uav.pos - obstacle.pos).T)
            b = self.get_b(uav, obstacle)
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

        # if np.isclose(np.linalg.norm(u_out), 0.0) and not np.isclose(np.linalg.norm(des_action), 0):
        # print(f"uav_id: {uav.id} in deadlock")
        # if np.linalg.norm(des_action - u_out) > 1e-3:
        if np.linalg.norm(des_action - u_out) > 0.0001:
            # u_out += np.random.random(3)*.00001
            pass
            # print("safety layer in effect")

        # u_out[2] = uav.m * (uav.g + u_out[2])
        return u_out

    def step(self, actions):
        # step uavs
        for i, action in actions.items():
            if self._use_safe_action:
                action = self.get_safe_action(self.uavs[i], action)
            self.uavs[i].step(action)

        # step target
        self.target.step(np.array([self.target_v, self.target_w]))

        # step obstacles
        for obstacle in self.obstacles:
            obstacle.step(np.array([self.target.vx, self.target.vy]))

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs}
        done = self._get_done()
        info = self._get_info()
        self._time_elapsed += self.dt

        # newer API to return truncated
        # return obs, reward, done, self.time_elapsed >= self.max_time, info
        return obs, reward, done, info

    def _get_info(self):
        info = {}

        for uav in self.uavs:
            uav_collision = 0
            obstacle_collision = 0

            for other_uav in self.uavs:
                if other_uav.id == uav.id:
                    continue
                uav_collision += 1 if uav.in_collision(other_uav) else 0

            for obstacle in self.obstacles:
                obstacle_collision += 1 if uav.in_collision(obstacle) else 0

            info[uav.id] = {
                "time_step": self.time_elapsed,
                "obstacle_collision": obstacle_collision,
                "uav_collision": uav_collision,
                "uav_landed": 1.0 if uav.landed else 0.0,
            }

        return info

    def _get_obs(self, uav):
        other_uav_states = []

        landing_pads = []
        for pad in self.target.pads:
            landing_pads.append(pad.state)

        landing_pads = np.array(landing_pads)

        for other_uav in self.uavs:
            if uav.id != other_uav.id:
                other_uav_states.append(other_uav.state)

        other_uav_states = np.array(other_uav_states)

        obstacle_states = []
        for obstacle in self.obstacles:
            obstacle_states.append(obstacle.state)

        obstacles = np.array(obstacle_states)

        obs_dict = {
            "state": uav.state.astype(np.float32),
            "rel_pad": (uav.state[0:6] - uav.pad.state[0:6]).astype(np.float32),
            # "landing_pads": landing_pads.astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "obstacles": obstacles.astype(np.float32),
        }

        return obs_dict

    def _get_reward(self, uav):
        if uav.done:
            return 0

        reward = 0
        # pos reward if uav lands on any landing pad
        is_reached, dest_dist = uav.check_dest_reached()
        if is_reached:
            uav.done = True
            uav.landed = True
            reward += 1
        else:
            reward -= dest_dist / np.linalg.norm(
                [self.env_max_l, self.env_max_w, self.env_max_h]
            )

        # for pad in self.target.pads:
        #     if uav.get_landed(pad):
        #         uav.done = True
        #         uav.landed = True
        #         reward += 1
        #         break

        # neg reward if uav collides with other uavs
        for other_uav in self.uavs:
            if uav.id == other_uav.id:
                continue
            reward -= (
                self.obstacle_collision_weight if uav.in_collision(other_uav) else 0
            )
        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            reward -= self.uav_collision_weight if uav.in_collision(obstacle) else 0
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
        random.seed(seed)
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

        if seed is None:
            seed = self._seed

        self.seed(seed)

        # TODO ensure we don't start in collision states
        # Reset Target
        x = np.random.rand() * self.env_max_w
        y = np.random.rand() * self.env_max_l
        x = self.env_max_w / 2.0
        y = self.env_max_h / 2.0
        self.target = Target(
            _id=0,
            x=x,
            y=y,
            v=self.target_v,
            w=self.target_w,
            dt=self.dt,
            num_landing_pads=self.num_uavs,
        )

        # Reset UAVs
        self.uavs = []
        for idx in range(self.num_uavs):
            x = np.random.rand() * self.env_max_w
            y = np.random.rand() * self.env_max_l
            z = np.random.rand() * self.env_max_h

            uav = Quad2DInt(
                # uav = Quadrotor(
                _id=idx,
                x=x,
                y=y,
                z=z,
                dt=self.dt,
                pad=self.target.pads[idx],
            )
            self.uavs.append(uav)

        # Reset obstacles
        self.obstacles = []
        for idx in range(self.num_obstacles):
            x = np.random.rand() * self.env_max_l
            y = np.random.rand() * self.env_max_l
            z = np.random.rand() * self.env_max_h
            z = np.random.uniform(low=0.1, high=self.env_max_h)
            _type = random.choice(list(ObsType))

            obstacle = Obstacle(_id=idx, x=x, y=y, z=z, _type=_type)
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
