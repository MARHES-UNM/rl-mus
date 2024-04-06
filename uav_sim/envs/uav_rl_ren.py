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
                            shape=(self.nom_num_uavs - 1, num_state_shape + 1),
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
            "uav_landed": 1.0 if uav.done else 0.0,
            "uav_done_dt": uav.done_dt,
            "uav_crashed": 1.0 if uav.crashed else 0.0,
            "uav_dt_go": uav.dt_go,
            "uav_done_time": uav.done_time
        }

        return info

    def _get_obs(self, uav):
        other_uav_state_list = []
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                temp_list = other_uav.state[:6].tolist()
                temp_list.append(self.get_cum_dt_go_error(other_uav))
                other_uav_state_list.append(temp_list)

        # other_uav_state_list = [
        #     other_uav.state[0:6].tolist()
        #     for other_uav in self.uavs.values()
        #     if uav.id != other_uav.id
        # ]

        num_active_other_agents = len(other_uav_state_list)
        if num_active_other_agents < self.nom_num_uavs - 1:
            fake_uav = [0.0] * 7
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
            "dt_go_error": np.array([self.get_cum_dt_go_error(uav)], dtype=np.float32),
            "rel_pad": (uav.state[0:6] - uav.pad.state[0:6]).astype(np.float32),
            "other_uav_obs": other_uav_states.astype(np.float32),
            "obstacles": obstacles_to_add.astype(np.float32),
        }

        return obs_dict

    def get_cum_dt_go_error(self, uav):
        """calculates cumulative t_go error between agents
        https://onlinelibrary.wiley.com/doi/abs/10.1002/asjc.2685

        Args:
            uav (_type_): _description_

        Returns:
            _type_: _description_
        """
        mean_tg_error = np.array(
            [
                x.get_t_go_est()
                for x in self.uavs.values()
                #  if x.id != uav.id
            ]
        ).mean()

        # cum_tg_error = abs(
        #     (self.num_uavs / (self.num_uavs - 1)) * (mean_tg_error - uav.get_t_go_est())
        # )
        uav_tg_error = np.array(
            [
                abs(other_uav.get_t_go_est() - uav.get_t_go_est())
                for other_uav in self.uavs.values()
                if other_uav.id != uav.id
            ]
        ).mean()

        return uav_tg_error
        # return uav_tg_error / mean_tg_error

        # cum_tg_error = abs(mean_tg_error - uav.get_t_go_est())
        # return cum_tg_error / mean_tg_error

    def _get_reward(self, uav):

        reward = 0.0
        t_remaining = self.time_final - self.time_elapsed
        uav.uav_collision = 0.0
        uav.obs_collision = 0.0

        if uav.done:
            # UAV most have finished last time_step, report zero collisions
            return reward

        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        uav.dt_go = uav.get_t_go_est()

        uav.done_dt = t_remaining
        uav.done_time = self.time_elapsed

        if is_reached:
            uav.done = True
            uav.landed = True

            reward += self._tgt_reward

            return reward

        elif rel_dist >= np.linalg.norm(
            [2 * self.env_max_l, 2 * self.env_max_w, self.env_max_h]
        ):
            uav.crashed = True
            reward += -self._crash_penalty
        else:
            reward -= self._beta * (
                rel_dist
                / np.linalg.norm(
                    [2 * self.env_max_l, 2 * self.env_max_w, self.env_max_h]
                )
            )

        # give small penalty for having large relative velocity
        reward += -self._beta * rel_vel

        reward += -self._stp_penalty * self.get_cum_dt_go_error(uav)

        # neg reward if uav collides with other uavs
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                if uav.in_collision(other_uav):
                    reward -= self.uav_collision_weight
                    uav.uav_collision += 1

        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            if uav.in_collision(obstacle):
                reward -= self.obstacle_collision_weight
                uav.obs_collision += 1

        return reward
