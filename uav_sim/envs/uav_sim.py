import sys
from gym import spaces
import numpy as np
import gym

from gym.utils import seeding
from uav_sim.agents.uav import Obstacle, UavBase, Uav, ObsType
from uav_sim.agents.uav import Target
from uav_sim.utils.gui import Gui
from qpsolvers import solve_qp
from scipy.integrate import odeint
import logging
import random

logger = logging.getLogger(__name__)


class UavSim(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}):
        super().__init__()
        self.dt = env_config.setdefault("dt", 0.1)
        self._seed = env_config.setdefault("seed", None)
        self.render_mode = env_config.setdefault("render_mode", "human")
        self.num_uavs = env_config.setdefault("num_uavs", 4)
        self.gamma = env_config.setdefault("gamma", 1)
        self.num_obstacles = env_config.setdefault("num_obstacles", 4)
        self.obstacle_radius = env_config.setdefault("obstacle_radius", 1)
        self.max_num_obstacles = env_config.setdefault("max_num_obstacles", 4)
        assert self.max_num_obstacles >= self.num_obstacles, print(
            f"Max number of obstacles {self.max_num_obstacles} is less than number of obstacles {self.num_obstacles}"
        )
        self.obstacle_collision_weight = env_config.setdefault(
            "obstacle_collision_weight", 1
        )
        self.uav_collision_weight = env_config.setdefault("uav_collision_weight", 1)
        self._use_safe_action = env_config.setdefault("use_safe_action", False)
        self.time_final = env_config.setdefault("time_final", 20.0)
        self.t_go_n = env_config.setdefault("t_go_n", 1.0)

        self._agent_ids = set(range(self.num_uavs))
        self._uav_type = getattr(
            sys.modules[__name__], env_config.get("uav_type", "Uav")
        )

        self.env_max_w = env_config.setdefault("env_max_w", 4)
        self.env_max_l = env_config.setdefault("env_max_l", 4)
        self.env_max_h = env_config.setdefault("env_max_h", 4)
        self.target_v = env_config.setdefault("target_v", 0)
        self.target_w = env_config.setdefault("target_w", 0)
        self.max_time = env_config.setdefault("max_time", 40)

        self.env_config = env_config
        self.norm_action_high = np.ones(3)
        self.norm_action_low = np.ones(3) * -1

        self.action_high = np.ones(3) * 5
        self.action_low = np.ones(3) * -5

        self.gui = None
        self._time_elapsed = 0.0
        self.seed(self._seed)
        self.reset()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def _get_action_space(self):
        """The action of the UAV. We don't normalize the action space in this environment.
        It is recommended to normalize using a wrapper function.
        The uav action consist of acceleration in x, y, and z component."""
        return spaces.Dict(
            {
                i: spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
                for i in range(self.num_uavs)
            }
        )

    def _get_observation_space(self):
        if self.num_obstacles == 0:
            num_obstacle_shape = 6
        else:
            num_obstacle_shape = self.obstacles[0].state.shape[0]

        obs_space = spaces.Dict(
            {
                i: spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=self.uavs[0].state.shape,
                            dtype=np.float32,
                        ),
                        "target": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=self.target.state.shape,
                            dtype=np.float32,
                        ),
                        "rel_pad": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=self.uavs[0].pad.state.shape,
                            dtype=np.float32,
                        ),
                        "constraint": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_uavs - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        "other_uav_obs": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_uavs - 1, self.uavs[0].state.shape[0]),
                            dtype=np.float32,
                        ),
                        "obstacles": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                self.num_obstacles,
                                num_obstacle_shape,
                            ),
                            dtype=np.float32,
                        ),
                    }
                )
                for i in range(self.num_uavs)
            }
        )

        return obs_space

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def _get_uav_constraint(self, uav):
        """Return single uav constraint"""
        constraints = []

        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                delta_p = uav.pos - other_uav.pos

                constraints.append(np.linalg.norm(delta_p) - (uav.r + other_uav.r))

        closest_obstacles = self._get_closest_obstacles(uav)

        for obstacle in closest_obstacles:
            delta_p = uav.pos - obstacle.pos

            constraints.append(np.linalg.norm(delta_p) - (uav.r + obstacle.r))

        return np.array(constraints)

    def get_constraints(self):
        return {uav.id: self._get_uav_constraint(uav) for uav in self.uavs}

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

    def get_time_coord_action(self, uav):
        t = self.time_elapsed
        tf = self.time_final
        N = self.t_go_n

        des_pos = np.zeros(12)
        des_pos[0:6] = uav.pad.state[0:6]
        pos_er = des_pos - uav.state

        t0 = min(t, tf - 0.1)
        t_go = (tf - t0) ** N
        p = self.get_p_mat(tf, N, t0)
        B = np.zeros((2, 1))
        B[1, 0] = 1.0

        action = t_go * np.array(
            [
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[0, 3]],
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[1, 4]],
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[2, 5]],
            ]
        )

        return action.squeeze()

    def get_p_mat(self, tf, N=1, t0=0.0):
        A = np.zeros((2, 2))
        A[0, 1] = 1.0

        B = np.zeros((2, 1))
        B[1, 0] = 1.0

        t_go = tf**N

        f1 = 2.0
        f2 = 2.0
        Qf = np.eye(2)
        Qf[0, 0] = f1
        Qf[1, 1] = f2

        Q = np.eye(2) * 0.0

        t = np.arange(tf, t0, -0.1)
        params = (tf, N, A, B, Q)

        g0 = np.array([*Qf.reshape((4,))])

        def dp_dt(time, state, tf, N, A, B, Q):
            t_go = (tf - time) ** N
            P = state[0:4].reshape((2, 2))
            p_dot = -(Q + P @ A + A.T @ P - P @ B * (t_go) @ B.T @ P)
            output = np.array(
                [
                    *p_dot.reshape((4,)),
                ]
            )
            return output

        result = odeint(dp_dt, g0, t, args=params, tfirst=True)
        return result

    def get_col_avoidance(self, uav, des_action):
        min_col_distance = uav.r * 2
        sum_distance = np.zeros(3)

        attractive_f = uav.pad.pos - uav.pos
        attractive_f = (
            self.action_high * attractive_f / (1e-3 + np.linalg.norm(attractive_f) ** 2)
        )

        # other agents
        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                if uav.rel_distance(other_uav) <= (min_col_distance + other_uav.r):
                    dist = other_uav.pos - uav.pos
                    sum_distance += dist

        closest_obstacles = self._get_closest_obstacles(uav)

        for obstacle in closest_obstacles:
            if uav.rel_distance(obstacle) <= (min_col_distance + obstacle.r):
                dist = obstacle.pos - uav.pos
                sum_distance += dist

        dist_vect = np.linalg.norm(sum_distance)
        if dist_vect <= 1e-9:
            u_out = des_action.copy()
        else:
            u_out = -self.action_high * sum_distance / dist_vect
        return u_out

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
            # Done uavs don't move
            if self.uavs[i].done:
                continue

            if self._use_safe_action:
                action = self.get_safe_action(self.uavs[i], action)
            # TODO: this may not be needed
            action = np.clip(action, self.action_low, self.action_high)
            self.uavs[i].step(action)

        # step target
        self.target.step(np.array([self.target_v, self.target_w]))
        # if self.target_v == 0.0:
        #     self.target.step()
        # else:
        # u = self.target.get_target_action(self.time_elapsed, 75.0)
        # self.target.step(u)

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
        """Must be called after _get_reward

        Returns:
            _type_: _description_
        """
        info = {}

        for uav in self.uavs:
            info[uav.id] = {
                "time_step": self.time_elapsed,
                "obstacle_collision": uav.obs_collision,
                "uav_rel_dist": uav.get_rel_pad_dist(),
                "uav_rel_vel": uav.get_rel_pad_vel(),
                "uav_collision": uav.uav_collision,
                "uav_landed": 1.0 if uav.landed else 0.0,
                "uav_done_time": uav.done_time if uav.landed else 0.0,
                "uav_t_go_est": uav.get_t_go_est(),
            }

        return info

    def _get_closest_obstacles(self, uav):
        obstacle_states = np.array([obs.state for obs in self.obstacles])
        dist = np.linalg.norm(obstacle_states[:, :3] - uav.state[:3][None, :], axis=1)
        argsort = np.argsort(dist)[: self.num_obstacles]
        closest_obstacles = [self.obstacles[idx] for idx in argsort]
        return closest_obstacles

    def _get_obs(self, uav):
        other_uav_states = np.array(
            [other_uav.state for other_uav in self.uavs if uav.id != other_uav.id]
        )

        closest_obstacles = self._get_closest_obstacles(uav)
        obstacles_to_add = np.array([obs.state for obs in closest_obstacles])

        obs_dict = {
            "state": uav.state.astype(np.float32),
            "target": self.target.state.astype(np.float32),
            "rel_pad": (uav.state[0:6] - uav.pad.state[0:6]).astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "obstacles": obstacles_to_add.astype(np.float32),
            "constraint": self._get_uav_constraint(uav).astype(np.float32),
        }

        return obs_dict

    def _get_reward(self, uav):
        uav.uav_collision = 0
        uav.obs_collision = 0

        if uav.done:
            # UAV most have finished last time_step, report zero collisions
            return 0

        reward = 0
        # pos reward if uav lands on any landing pad
        is_reached, dest_dist = uav.check_dest_reached()
        if is_reached:
            uav.done = True
            uav.landed = True
            uav.done_time = self.time_elapsed
            reward += 1
        else:
            reward -= dest_dist / np.linalg.norm(
                [self.env_max_l, self.env_max_w, self.env_max_h]
            )

        # neg reward if uav collides with other uavs
        for other_uav in self.uavs:
            if uav.id != other_uav.id and uav.in_collision(other_uav):
                reward -= self.uav_collision_weight
                # uav.done = True
                # uav.landed = False
                uav.uav_collision += 1

        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            if uav.in_collision(obstacle):
                reward -= self.obstacle_collision_weight
                # uav.done = True
                # uav.landed = False
                uav.obs_collision += 1

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
        # not currently compatible with new gym api to pass in seed
        # if seed is None:
        #     seed = self._seed
        # super().reset(seed=seed)

        if self.gui is not None:
            self.close_gui()

        self._time_elapsed = 0.0

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

        def get_random_pos(low_h=0.1):
            x = np.random.rand() * self.env_max_w
            y = np.random.rand() * self.env_max_l
            z = np.random.uniform(low=low_h, high=self.env_max_h)
            return (x, y, z)

        def is_in_collision(uav):
            for pad in self.target.pads:
                if uav.get_landed(pad):
                    return True

            for obstacle in self.obstacles:
                if uav.in_collision(obstacle):
                    return True

            for other_uav in self.uavs:
                if uav.in_collision(other_uav):
                    return True

            return False

        # Reset obstacles, obstacles should not be in collision with target. Obstacles can be in collision with each other.
        self.obstacles = []
        for idx in range(self.max_num_obstacles):
            in_collision = True

            while in_collision:
                x, y, z = get_random_pos(low_h=self.obstacle_radius * 1.5)
                _type = ObsType.S
                obstacle = Obstacle(
                    _id=idx,
                    x=x,
                    y=y,
                    z=z,
                    r=self.obstacle_radius,
                    dt=self.dt,
                    _type=_type,
                )

                in_collision = any(
                    [
                        obstacle.in_collision(other_obstacle)
                        for other_obstacle in self.obstacles
                        if obstacle.id != other_obstacle.id
                    ]
                )

            self.obstacles.append(obstacle)

        # Reset UAVs
        self.uavs = []
        for agent_id in self._agent_ids:
            in_collision = True

            # make sure the agent is not in collision with other agents or obstacles
            # the lowest height needs to be the uav radius x2
            while in_collision:
                x, y, z = get_random_pos(low_h=0.2)
                uav = self._uav_type(
                    _id=agent_id,
                    x=x,
                    y=y,
                    z=z,
                    dt=self.dt,
                    pad=self.target.pads[agent_id],
                )
                in_collision = is_in_collision(uav)

            self.uavs.append(uav)

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs}

        return obs

    def unscale_action(self, action):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert np.all(np.greater_equal(action, self.norm_action_low)), (
            action,
            self.norm_action_low,
        )
        assert np.all(np.less_equal(action, self.norm_action_high)), (
            action,
            self.norm_action_high,
        )
        action = self.action_low + (self.action_high - self.action_low) * (
            (action - self.norm_action_low)
            / (self.norm_action_high - self.norm_action_low)
        )
        # # TODO: this is not needed
        # action = np.clip(action, self.action_low, self.action_high)

        return action

    def scale_action(self, action):
        """Scale agent action between default norm action values"""
        # assert np.all(np.greater_equal(action, self.action_low)), (action, self.action_low)
        # assert np.all(np.less_equal(action, self.action_high)), (action, self.action_high)
        action = (self.norm_action_high - self.norm_action_low) * (
            (action - self.action_low) / (self.action_high - self.action_low)
        ) + self.norm_action_low

        return action

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

    def close(self):
        self.close_gui()
