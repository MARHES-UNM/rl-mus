import sys
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from uav_sim.agents.uav import Obstacle, UavBase, Uav, ObsType
from uav_sim.agents.uav import Target
from uav_sim.utils.gui import Gui
from qpsolvers import solve_qp
from scipy.integrate import odeint
import logging
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import io


logger = logging.getLogger(__name__)


class UavSim(MultiAgentEnv):
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
        self.nom_num_uavs = env_config.setdefault("nom_num_uavs", 4)
        self.gamma = env_config.setdefault("gamma", 1)
        self.num_obstacles = env_config.setdefault("num_obstacles", 4)
        self.nom_num_obstacles = env_config.setdefault("nom_num_obstacles", 4)
        self.obstacle_radius = env_config.setdefault("obstacle_radius", 0.1)
        self.max_num_obstacles = env_config.setdefault("max_num_obstacles", 4)
        # self.num_obstacles = min(self.max_num_obstacles, self.num_obstacles)
        if self.max_num_obstacles < self.num_obstacles:
            self.num_obstacles = self.max_num_obstacles

        assert self.max_num_obstacles >= self.num_obstacles, print(
            f"Max number of obstacles {self.max_num_obstacles} is less than number of obstacles {self.num_obstacles}"
        )
        self.obstacle_collision_weight = env_config.setdefault(
            "obstacle_collision_weight", 0.1
        )
        self.uav_collision_weight = env_config.setdefault("uav_collision_weight", 0.1)
        self._use_safe_action = env_config.setdefault("use_safe_action", False)
        self._use_virtual_leader = env_config.setdefault("use_virtual_leader", False)
        self.time_final = env_config.setdefault("time_final", 8.0)
        self.t_go_max = env_config.setdefault("t_go_max", 2.0)
        self.t_go_n = env_config.setdefault("t_go_n", 1.0)
        self._beta = env_config.setdefault("beta", 0.01)
        self._beta_vel = env_config.setdefault("beta_vel", 0.00)
        self._d_thresh = env_config.setdefault("d_thresh", 0.01)  # uav.rad + pad.rad
        self._tgt_reward = env_config.setdefault("tgt_reward", 0.0)
        self._sa_reward = env_config.setdefault("sa_reward", self._tgt_reward)
        self._crash_penalty = env_config.setdefault("crash_penalty", 10.0)
        self._dt_go_penalty = env_config.setdefault("dt_go_penalty", 10.0)
        self._stp_penalty = env_config.setdefault("stp_penalty", 0.0)
        self._max_time_penalty = env_config.setdefault("max_time_penalty", 5.0)
        self._dt_reward = env_config.setdefault("dt_reward", 0.0)
        self._dt_weight = env_config.setdefault("dt_weight", 0.0)
        self.num_state_shape = 6

        self._agent_ids = set(range(self.num_uavs))
        self._uav_type = getattr(
            sys.modules[__name__], env_config.get("uav_type", "Uav")
        )

        self.env_max_w = env_config.setdefault("env_max_w", 1.25)
        self.env_max_l = env_config.setdefault("env_max_l", 1.25)
        self.env_max_h = env_config.setdefault("env_max_h", 1.75)
        self.max_rel_dist = np.linalg.norm(
            [2 * self.env_max_w, 2 * self.env_max_l, self.env_max_h]
        )
        self.max_start_dist = env_config.setdefault("max_start_dist", self.max_rel_dist)

        # check max start distance not outside environment
        if self.max_start_dist > self.max_rel_dist:
            logger.warning(
                f"max_start_dist {self.max_start_dist} is greater than max_rel_dist {self.max_rel_dist}"
            )
            self.max_start_dist = self.max_rel_dist

        self._z_high = env_config.setdefault("z_high", self.env_max_h)
        self._z_high = min(self.env_max_h, self._z_high)
        self._z_low = env_config.setdefault("z_low", 0.2)
        self._z_low = max(0, self._z_low)
        self.pad_r = env_config.setdefault("pad_r", 0.1)

        # check min start distance
        if self.max_start_dist < 0.5:
            logger.warning(
                f"max_start_dist {self.max_start_dist} is less than min pad dist: {0.5}. Setting to {0.5}"
            )
            self.max_start_dist = 0.5

        self.target_v = env_config.setdefault("target_v", 0)
        self.target_w = env_config.setdefault("target_w", 0)
        self.target_r = env_config.setdefault("target_r", 0.50)
        self.max_time = self.time_final + self.t_go_max
        self.max_dt_std = env_config.setdefault("max_dt_std", 0.1)
        self.max_dt_go_error = env_config.setdefault("max_dt_go_error", 0.2)
        self._t_go_error_func = env_config.setdefault("t_go_error_func", "mean")
        self._early_done = env_config.setdefault("early_done", False)
        env_config["max_time"] = self.max_time

        self._env_config = env_config
        self.norm_action_high = np.ones(3)
        self.norm_action_low = np.ones(3) * -1

        self.action_high = np.ones(3) * 5.0
        self.action_low = -self.action_high

        self.gui = None
        self.alive_agents = set()
        self._time_elapsed = 0.0

        self.all_landed = []
        self.first_landing_time = None
        self._scaled_t_go_error = []
        self._all_t_go_errors = []
        self.seed(self._seed)
        self.reset()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    @property
    def env_config(self):
        return self._env_config

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def agent_ids(self):
        return self._agent_ids

    def _get_action_space(self):
        """The action of the UAV. We don't normalize the action space in this environment.
        It is recommended to normalize using a wrapper function.
        The uav action consist of acceleration in x, y, and z component."""
        return spaces.Dict(
            {
                i: spaces.Box(
                    low=self.action_low,
                    high=self.action_high,
                    shape=(3,),
                    dtype=np.float32,
                )
                for i in range(self.num_uavs)
            }
        )

    def _get_observation_space(self):
        num_state_shape = 6
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
                            shape=(num_state_shape,),
                            dtype=np.float32,
                        ),
                        "done_dt": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        # "dt_go": spaces.Box(
                        #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        # ),
                        # "target": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=self.target.state.shape,
                        #     dtype=np.float32,
                        # ),
                        "rel_pad": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(num_state_shape,),
                            dtype=np.float32,
                        ),
                        # "constraint": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=(self.num_uavs - 1 + self.num_obstacles,),
                        #     dtype=np.float32,
                        # ),
                        "other_uav_obs": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.nom_num_uavs - 1, num_state_shape),
                            dtype=np.float32,
                        ),
                        "obstacles": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                self.nom_num_obstacles,
                                num_state_shape,
                            ),
                            dtype=np.float32,
                        ),
                    }
                )
                for i in range(self.num_uavs)
            }
        )

        return obs_space

    def _get_uav_constraint(self, uav):
        """Return single uav constraint"""
        constraints = []

        for other_uav in self.uavs.values():
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

    def get_tc_controller(self, uav):
        """https://onlinelibrary.wiley.com/doi/abs/10.1002/asjc.2685

        Args:
            uav (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.num_uavs > 1:
            mean_tg_error = np.array(
                [
                    # x.get_t_go_est() - (self.time_final - self.time_elapsed)
                    x.get_t_go_est()  # - (self.time_final - self.time_elapsed)
                    for x in self.uavs.values()
                    if x.id != uav.id
                ]
            ).mean()

            cum_tg_error = (self.num_uavs / (self.num_uavs - 1)) * (
                # mean_tg_error - (uav.get_t_go_est() - (self.time_final - self.time_elapsed))
                mean_tg_error
                - (uav.get_t_go_est())  # - (self.time_final - self.time_elapsed)
            )

        # else:
        # cum_tg_error = mean_tg_error - uav.get_t_go_est()
        # cum_tg_error = 0

        # for other_uav in self.uavs.values():
        #     if other_uav.id != uav.id:
        #         cum_tg_error += other_uav.get_t_go_est() - uav.get_t_go_est()

        des_pos = np.zeros(12)
        des_pos[0:6] = uav.pad.state[0:6]
        pos_er = des_pos - uav.state

        action = np.zeros(3)
        # if uav.id == 0:

        #     action = -1 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        # else:
        #     action = (
        #         # -0.5 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #         -0.05
        #         * cum_tg_error
        #         * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #     )
        # action = (
        #     # -0.5 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #     # -0.05
        #     # -5 / (uav.init_r * uav.init_tg)
        #     -0.5
        #     * cum_tg_error
        #     * np.array([pos_er[0], pos_er[1], pos_er[2]])
        # )
        # cum_tg_error = self.time_final - self.time_elapsed
        # action += (
        #     3 * np.array([pos_er[0], pos_er[1], pos_er[2]]) #* (-0.3 * cum_tg_error)
        # )

        # action +=  3 * np.linalg.norm([pos_er[0], pos_er[1], pos_er[2]]) * cum_tg_error
        # action += 2 * (1 - 2 * np.linalg.norm(pos_er[:3])*cum_tg_error)
        # action +=  2* cum_tg_error

        # action += 2 * cum_tg_error * np.array([pos_er[3], pos_er[4], pos_er[5]])

        # K = 1 * (1 - 1 * np.linalg.norm(pos_er[:3]) * cum_tg_error)
        # action += 1 * ( 0.3 * np.linalg.norm(pos_er[:3]) * cum_tg_error)
        # K = 1
        # action += .2 * np.array(
        action += 1 * np.array(
            [pos_er[0], pos_er[1], pos_er[2]]
        )  # * (-0.3 * cum_tg_error)

        # action += 3 * np.array([pos_er[3], pos_er[4], pos_er[5]])

        return action

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
        for other_uav in self.uavs.values():
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
        for other_uav in self.uavs.values():
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

    def get_scaled_t_go_error(self):
        # get all t_go_errors
        t_go_errors = []
        for uav in self.uavs.values():
            if self._use_virtual_leader:
                (
                    t_go_errors.append(
                        (self.time_final - self._time_elapsed) - uav.get_t_go_est()
                    )
                )
            else:
                t_go_error = np.array(
                    [
                        other_uav.get_t_go_est() - uav.get_t_go_est()
                        for other_uav in self.uavs.values()
                        if other_uav.id
                        != uav.id  # and other_uav in self.alive_agents and not other_uav.done
                    ]
                )
                t_go_errors.append(t_go_error.sum())

        # t_go_errors = [
        #     uav.get_t_go_est() for uav in self.uavs.values()
        # ]

        # self._all_t_go_errors.extend(t_go_errors)

        # running_t_go_errors = np.array(self._all_t_go_errors)
        # normalize t_go_errors
        t_go_errors = np.array(t_go_errors)
        # if self._use_virtual_leader:
        #     return t_go_errors
        # t_go_errors = (t_go_errors - np.mean(t_go_errors)) / (np.std(t_go_errors) + 1e-9)
        t_go_errors = (t_go_errors) / (15 + 1e-9)

        # t_go_errors = (t_go_errors - np.mean(running_t_go_errors)) / (np.std(running_t_go_errors) + 1e-9)

        return t_go_errors

    def step(self, actions):
        # step uavs
        self.alive_agents = set()
        for i, action in actions.items():
            self.alive_agents.add(i)
            # Done uavs don't move
            if self.uavs[i].done or self.uavs[i].landed:
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

        self._scaled_t_go_error = self.get_scaled_t_go_error()

        obs = {
            uav.id: self._get_obs(uav)
            for uav in self.uavs.values()
            if uav.id in self.alive_agents
        }
        reward = {
            uav.id: self._get_reward(uav)
            for uav in self.uavs.values()
            if uav.id in self.alive_agents
        }

        # get global reward
        glob_reward = self._get_global_reward()
        reward = {k: v + glob_reward for k, v in reward.items()}

        # IMPORTANT: this must be called after _get_reward or get_global_reward. Not before.
        info = {
            uav.id: self._get_info(uav)
            for uav in self.uavs.values()
            if uav.id in self.alive_agents
        }

        # Done only if all landed
        # all_landed = all([uav.landed for uav in self.uavs.values()])

        landed = [uav.landed for uav in self.uavs.values()]
        all_landed = all(landed)

        # calculate done for each agent
        done = {
            self.uavs[uav_id].id: self.uavs[uav_id].done  # or all_landed
            for uav_id in self.alive_agents
        }

        done_all = []
        for uav in self.uavs.values():
            uav_done = uav.done or uav.landed
            done_all.append(uav_done)

        # # if all(done_all):
        # # get global reward
        # glob_reward = self._get_global_reward()
        # reward = {k: v + glob_reward for k, v in reward.items()}

        done["__all__"] = (
            all(done_all) or self.time_elapsed >= self.max_time or all_landed
        )
        self._time_elapsed += self.dt

        # newwer api gymnasium > 0.28
        # return obs, reward, terminated, terminated, info

        # old api gym < 0.26.1
        # return obs, reward, done, info
        return obs, reward, done, done, info

    def _get_info(self, uav):
        """Must be called after _get_reward

        Returns:
            _type_: _description_
        """

        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        info = {
            "time_step": self.time_elapsed,
            "obstacle_collision": uav.obs_collision,
            "uav_rel_dist": rel_dist,
            "uav_rel_vel": rel_vel,
            "uav_collision": uav.uav_collision,
            "uav_landed": 1.0 if uav.done else 0.0,
            "uav_done_dt": uav.done_dt,
            "uav_crashed": 1.0 if uav.crashed else 0.0,
            "uav_dt_go": uav.dt_go,
        }

        return info

    def _get_cum_dt_go_est(self, uav):
        pass

    def _get_closest_obstacles(self, uav):
        obstacle_states = np.array([obs.state for obs in self.obstacles])

        # there must be 0 obstacles return empty list
        if len(obstacle_states) == 0:
            return []

        dist = np.linalg.norm(obstacle_states[:, :3] - uav.state[:3][None, :], axis=1)
        argsort = np.argsort(dist)[: self.num_obstacles]
        closest_obstacles = [self.obstacles[idx] for idx in argsort]
        return closest_obstacles

    def _get_obs(self, uav):
        other_uav_state_list = [
            other_uav.state[0:6]
            for other_uav in self.uavs.values()
            if uav.id != other_uav.id
        ]

        num_active_other_agents = len(other_uav_state_list)
        if num_active_other_agents < self.nom_num_uavs - 1:
            fake_uav = [0.0] * 6
            for _ in range(self.nom_num_uavs - 1 - num_active_other_agents):
                other_uav_state_list.append(fake_uav.copy())

        other_uav_states = np.array(other_uav_state_list, dtype=np.float32)

        closest_obstacles = self._get_closest_obstacles(uav)

        obstacle_state_list = [obs.state[0:6] for obs in closest_obstacles]
        num_active_obstacles = len(obstacle_state_list)
        if num_active_obstacles < self.nom_num_obstacles:
            fake_obs = [0.0] * 6
            for _ in range(self.nom_num_obstacles - num_active_obstacles):
                obstacle_state_list.append(fake_obs)

        obstacles_to_add = np.array(obstacle_state_list, dtype=np.float32)

        obs_dict = {
            "state": uav.state[0:6].astype(np.float32),
            # "target": self.target.state.astype(np.float32),
            # https://farama.org/Gymnasium-Terminated-Truncated-Step-API
            # time is part of the observation
            "done_dt": np.array(
                [self.time_final - self._time_elapsed], dtype=np.float32
            ),
            # "dt_go": np.array(
            #     [uav.get_t_go_est() - (self.time_final - self._time_elapsed)],
            #     dtype=np.float32,
            # ),
            "rel_pad": (uav.state[0:6] - uav.pad.state[0:6]).astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "obstacles": obstacles_to_add.astype(np.float32),
            # "constraint": self._get_uav_constraint(uav).astype(np.float32),
        }

        return obs_dict

    def _get_global_reward(self):
        return 0

    def _get_reward(self, uav):
        # reward_info = {
        #     "stp_penalty": 0,
        #     "tgt_reward": 0,
        #     "dt_reward": 0,
        #     "r_tgt_reward": 0,
        #     "r_tgt_reward": 0,
        #     "dt_go_penalty": 0,
        # }
        reward = 0.0
        t_remaining = self.time_final - self.time_elapsed
        uav.uav_collision = 0.0
        uav.obs_collision = 0.0

        if uav.done:
            # UAV most have finished last time_step, report zero collisions
            return reward

        # give penalty for reaching the time limit
        elif self.time_elapsed >= self.max_time:
            reward -= self._stp_penalty
            return reward

        # pos reward if uav lands on any landing pad
        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        # TODO: fix getting dt_go
        # if rel_vel == 0:
        #     _rel_vel = self.action_high * self.dt
        #     uav.dt_go = uav.get_t_go_est(_rel_vel) - t_remaining
        # else:

        uav.dt_go = uav.get_t_go_est() - t_remaining

        uav.done_dt = t_remaining

        if is_reached:
            uav.done = True
            uav.landed = True
            uav.done_time = self.time_elapsed

            reward += -(abs(uav.done_dt) / self.time_final) * self._stp_penalty
            # get reward for reaching destination in time
            # if abs(uav.done_dt) < self.t_go_max:
            #     reward += self._tgt_reward

            # else:

            # reward += (
            #     -(1 - (self.time_elapsed / self.time_final)) * self._stp_penalty
            # )

            # No need to check for other reward, UAV is done.
            return reward

        elif rel_dist >= np.linalg.norm(
            [2 * self.env_max_l, 2 * self.env_max_w, self.env_max_h]
        ):
            uav.crashed = True
            reward += -self._crash_penalty
        else:
            reward -= self._beta * (
                rel_dist
                / self.max_rel_dist
                # / np.linalg.norm(
                # [2 * self.env_max_l, 2 * self.env_max_w, self.env_max_h]
                # )
            )

        # give small penalty for having large relative velocity
        reward += -self._beta * rel_vel

        # uav.done_dt = self.time_final
        # get reward if uav maintains time difference
        # if t_remaining >= 0 and (uav.dt_go < self.t_go_max):
        # TODO: consider adding a consensus reward here that is the sum of the error time difference between UAVs
        # this of this as a undirected communication graph
        # if abs(uav.dt_go) > self.t_go_max:
        #     reward += -self._dt_go_penalty

        cum_dt_penalty = 0

        # neg reward if uav collides with other uavs
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                if uav.in_collision(other_uav):
                    reward -= self.uav_collision_weight
                    uav.uav_collision += 1

                # cum_dt_penalty += other_uav.get_t_go_est() - uav.get_t_go_est()
        # reward -= self._dt_weight * abs(cum_dt_penalty)

        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            dist_col = uav.rel_distance(obstacle)

            if uav.in_collision(obstacle):
                reward -= self.obstacle_collision_weight
                uav.obs_collision += 1

        return reward

    def seed(self, seed=None):
        """Random value to seed"""
        random.seed(seed)
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def get_random_pos(
        self,
        low_h=0.1,
        x_high=None,
        y_high=None,
        z_high=None,
    ):
        if x_high is None:
            x_high = self.env_max_w
        if y_high is None:
            y_high = self.env_max_l
        if z_high is None:
            z_high = self.env_max_h

        x = np.random.uniform(low=-x_high, high=x_high)
        y = np.random.uniform(low=-y_high, high=y_high)
        z = np.random.uniform(low=low_h, high=z_high)
        return np.array([x, y, z])

    def is_in_collision(self, entity, pos, rad):
        for target in self.targets.values():
            if target.in_collision(entity, pos, rad):
                return True

        for obstacle in self.obstacles:
            if obstacle.in_collision(entity, pos, rad):
                return True

        for other_uav in self.uavs.values():
            if other_uav.in_collision(entity, pos, rad):
                return True

        return False

    def reset(self, *, seed=None, options=None):
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
        self._agent_ids = set(range(self.num_uavs))

        def get_random_pos(
            low_h=0.1,
            x_high=self.env_max_w,
            y_high=self.env_max_l,
            z_high=self.env_max_h,
        ):
            x = np.random.uniform(low=-x_high, high=x_high)
            y = np.random.uniform(low=-y_high, high=y_high)
            z = np.random.uniform(low=low_h, high=z_high)
            return (x, y, z)

        (x, y, z) = get_random_pos(
            low_h=0,
            x_high=self.env_max_w - self.target_r,
            y_high=self.env_max_l - self.target_r,
            z_high=self.env_max_h - self.target_r,
        )
        # (x, y, z) = (0.0, 0.0, 0.75)

        self.target = Target(
            _id=0,
            x=x,
            y=y,
            z=z,
            v=self.target_v,
            w=self.target_w,
            dt=self.dt,
            r=self.target_r,
            num_landing_pads=self.num_uavs,
        )

        def is_in_collision(uav):
            # for pad in self.target.pads:
            #     pad_landed, _, _ = uav.check_dest_reached(pad)
            #     if pad_landed:
            #         return True

            pad_landed, rel_dist, _ = uav.check_dest_reached()
            if pad_landed:
                return True
            if rel_dist > self.max_start_dist:
                return True

            for obstacle in self.obstacles:
                if uav.in_collision(obstacle):
                    return True

            for other_uav in self.uavs.values():
                if uav.in_collision(other_uav):
                    return True

            return False

        # Reset obstacles, obstacles should not be in collision with target. Obstacles can be in collision with each other.
        self.obstacles = []
        for idx in range(self.max_num_obstacles):
            in_collision = True

            while in_collision:
                x, y, z = get_random_pos(
                    low_h=self.obstacle_radius * 2.0, z_high=self.env_max_h - 0.25
                )
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
                ) or obstacle.in_collision(self.target)

            self.obstacles.append(obstacle)

        # Reset UAVs
        self.uavs = {}
        for agent_id in self._agent_ids:
            in_collision = True

            # make sure the agent is not in collision with other agents or obstacles
            # the lowest height needs to be the uav radius x2
            while in_collision:
                x, y, z = get_random_pos(low_h=self._z_low, z_high=self._z_high)
                uav = self._uav_type(
                    _id=agent_id,
                    x=x,
                    y=y,
                    z=z,
                    dt=self.dt,
                    pad=self.target.pads[agent_id],
                    d_thresh=self._d_thresh,
                )
                in_collision = is_in_collision(uav)

            uav.last_rel_dist = np.linalg.norm([x, y, z])
            self.uavs[agent_id] = uav

        self.all_landed = []
        self.first_landing_time = None
        self.alive_agents = set([uav.id for uav in self.uavs.values()])
        self._all_t_go_errors = []
        self._scaled_t_go_error = self.get_scaled_t_go_error()
        obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values()}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values()}

        # get global reward
        glob_reward = self._get_global_reward()
        reward = {k: v + glob_reward for k, v in reward.items()}
        info = {uav.id: self._get_info(uav) for uav in self.uavs.values()}

        return obs, info

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

    def render(
        self,
        mode="human",
        done=False,
        obs=None,
        rew=None,
        info=None,
        plot_results=False,
    ):
        """
        See this example for converting python figs to images:
        https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array

        Args:
            mode (str, optional): _description_. Defaults to "human".
            done (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if self.gui is None:
            self.gui = Gui(
                self.uavs,
                target=self.target,
                obstacles=self.obstacles,
                max_x=self.env_max_w,
                max_y=self.env_max_l,
                max_z=self.env_max_h,
            )

        if mode == "human":
            self.gui.update(self._time_elapsed, done, obs, rew, info, plot_results)

        elif mode == "rgb_array":
            fig = self.gui.update(
                self._time_elapsed, done, obs, rew, info, plot_results
            )

            with io.BytesIO() as buff:
                fig.savefig(buff, format="raw")
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                im = data.reshape((int(h), int(w), -1))
            return im

    def close_gui(self):
        if self.gui is not None:
            self.gui.close()
        self.gui = None

    def close(self):
        self.close_gui()
