from gymnasium import spaces
import numpy as np
from uav_sim.envs.uav_sim import UavSim
import logging


logger = logging.getLogger(__name__)


class UavRlRen(UavSim):

    def __init__(self, env_config={}):
        super().__init__(env_config=env_config)

    def _get_observation_space(self):
        num_state_shape = 6

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
                        "dt_go_error": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "rel_pad": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(num_state_shape,),
                            dtype=np.float32,
                        ),
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
            "uav_landed": 1.0 if uav.landed else 0.0,
            "uav_done_dt": uav.done_dt,
            "uav_crashed": 1.0 if uav.crashed else 0.0,
            "uav_dt_go": self.get_uav_t_go_error(uav),
            "uav_t_go": uav.get_t_go_est(),
            "uav_done_time": uav.done_time,
            "uav_sa_sat": 1.0 if uav.sa_sat else 0.0,
        }

        return info

    def _get_obs(self, uav):
        # TODO: handle case with more than the number of nominal uavs
        other_uav_state_list = []
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                temp_list = other_uav.state[:6].tolist()
                # temp_list.append(self.get_uav_t_go_error(other_uav))
                other_uav_state_list.append(temp_list)

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
            "dt_go_error": np.array(
                [self.get_uav_t_go_error(uav) / self.get_mean_tg_error()],
                dtype=np.float32,
            ),
            "rel_pad": (uav.state[0:6] - uav.pad.state[0:6]).astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "obstacles": obstacles_to_add.astype(np.float32),
        }

        return obs_dict

    def get_mean_tg_error(self):
        if self.num_uavs == 1:
            return 0

        mean_tg_error = np.array([x.get_t_go_est() for x in self.uavs.values()]).mean()

        return mean_tg_error + 1e-6

    def get_uav_t_go_error(self, uav):

        if self._use_virtual_leader:
            return (self.time_final - self._time_elapsed) - uav.get_t_go_est()

        uav_tg_error = np.array(
            [
                other_uav.get_t_go_est() - uav.get_t_go_est()
                for other_uav in self.uavs.values()
                if other_uav.id != uav.id
            ]
        )

        if self._t_go_error_func == "mean":
            uav_tg_error = uav_tg_error.mean()
        else:
            uav_tg_error = uav_tg_error.sum()

        return uav_tg_error

    def get_cum_dt_go_error(self, uav):
        # TODO: fix this so we use sum instead of mean
        # TODO: use max instead of min here
        """calculates cumulative t_go error between agents
        https://onlinelibrary.wiley.com/doi/abs/10.1002/asjc.2685

        Args:
            uav (_type_): _description_

        Returns:
            _type_: _description_
        """

        # t_remaining = self.time_final - self.time_elapsed

        # return t_remaining - uav.get_t_go_est()

        if self.num_uavs == 1:
            return 0

        mean_tg_error = np.array([x.get_t_go_est() for x in self.uavs.values()]).mean()

        uav_tg_error = np.array(
            [
                other_uav.get_t_go_est() - uav.get_t_go_est()
                for other_uav in self.uavs.values()
                if other_uav.id != uav.id
            ]
        ).sum()

        if mean_tg_error < 1e-4:
            return 0.0

        return uav_tg_error / mean_tg_error

    def _get_global_reward(self):
        all_landed = all([uav.landed for uav in self.uavs.values()])

        if all_landed:
            done_time = np.array([uav.done_time for uav in self.uavs.values()]).std()
            if done_time <= self.max_dt_std:
                for uav in self.uavs.values():
                    uav.sa_sat = True
                return self._sa_reward

        return 0.0

    def get_tc_controller(self, uav):
        pos_er = uav.pad.state[0:6] - uav.state[0:6]

        action = np.zeros(3)

        action += 20 * np.array(pos_er[:3])

        action += 5 * np.array(pos_er[3:])

        uav_tg_error = self.get_uav_t_go_error(uav)
        # uav_tg_error = self.get_cum_dt_go_error(uav)
        #
        # uav_tg_error = (self.time_final - self._time_elapsed) - uav.get_t_go_est()

        # action *= (1 -  2 * uav_tg_error / ( uav.get_t_go_est()))
        action *= 1 - 1 * np.abs(uav_tg_error) ** (0.5) * np.sign(uav_tg_error)

        return action

    def _get_reward(self, uav):

        reward = 0.0
        t_remaining = self.time_final - self._time_elapsed
        uav.uav_collision = 0.0
        uav.obs_collision = 0.0

        if uav.done:
            # UAV most have finished last time_step, return 0 for reward
            return reward

        # give penaly for reaching the time limit
        elif self._time_elapsed >= self.max_time:
            reward -= self._max_time_penalty
            uav.done_time = self.max_time
            return reward

        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        uav.t_go = uav.get_t_go_est()

        uav_dt_go_error = self.get_uav_t_go_error(uav)

        uav.dt_go = uav_dt_go_error
        uav.done_dt = t_remaining

        if is_reached:
            uav.done = True
            uav.landed = True

            if uav.done_time == 0:
                uav.done_time = self._time_elapsed

            reward += self._tgt_reward

            return reward
        elif (
            uav.state[0] < -self.env_max_w
            or uav.state[0] > self.env_max_w
            or uav.state[1] < -self.env_max_l
            or uav.state[1] > self.env_max_l
            or uav.state[2] > self.env_max_h
        ):
            uav.crashed = True
            if self._early_done:
                uav.done = True
            reward += -self._crash_penalty
        # elif rel_dist >= np.linalg.norm(
        #     [self.env_max_l, self.env_max_w, self.env_max_h]
        # ):
        #     uav.crashed = True
        #     reward += -self._crash_penalty
        else:
            reward += self._beta * np.sign(uav.last_rel_dist - rel_dist)

            uav.last_rel_dist = rel_dist

        # give small penalty for having large relative velocity
        reward += -self._beta_vel * rel_vel

        # reward += max(0, self._stp_penalty - abs(uav_dt_go_error))
        if abs(uav_dt_go_error) <= self.max_dt_go_error:
            reward += self._stp_penalty

        # if abs(uav_dt_go_error) > self.max_dt_std:
        # reward += -self._stp_penalty

        # neg reward if uav collides with other uavs
        other_uav_list = []

        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                other_uav_list.append(other_uav)
                if uav.in_collision(other_uav):
                    # reward -= self.uav_collision_weight
                    if self._early_done:
                        uav.done = True
                    uav.uav_collision += 1

        closest_uavs = uav.get_closest_entities(other_uav_list, num_to_return=1)

        if closest_uavs:
            dist_to_uav = uav.rel_distance(closest_uavs[0])
            if uav.in_collision(closest_uavs[0]):
                reward -= self.uav_collision_weight
            elif dist_to_uav <= (uav.r + 0.15):
                reward += -np.exp(-dist_to_uav / 0.1)

        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            if uav.in_collision(obstacle):
                # reward -= self.obstacle_collision_weight
                if self._early_done:
                    uav.done = True
                uav.obs_collision += 1

        closest_obstacles = uav.get_closest_entities(self.obstacles, num_to_return=1)

        if closest_obstacles:
            dist_to_obstacle = uav.rel_distance(closest_obstacles[0])
            if uav.in_collision(closest_obstacles[0]):
                reward -= self.obstacle_collision_weight
            elif dist_to_obstacle <= (obstacle.r + 0.25):
                reward += -np.exp(-dist_to_obstacle / 0.1)

        return reward
