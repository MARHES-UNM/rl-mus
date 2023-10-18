from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class TrainCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length <= 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        # TODO: add callback for dt_go: https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/custom_metrics_and_callbacks.py
        episode.user_data["obstacle_collisions"] = []
        episode.user_data["uav_collisions"] = []
        episode.user_data["uav_dt_go"] = []
        # episode.user_data["num_uav_landed"] = []
        # episode.user_data["uav_done_dt"] = []
        # episode.user_data["uav_rel_dist"] = []

        # episode.user_data["uav_dt_go"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        agent_ids = episode.get_agents()
        cum_uav_collisions = 0
        cum_obstacle_collisions = 0
        cum_uav_dt_go = 0

        for agent_id in agent_ids:
            # last_info = episode.last_info_for(agent_id)
            last_info = episode._last_infos[agent_id]
            cum_uav_collisions += last_info["uav_collision"]
            cum_obstacle_collisions += last_info["obstacle_collision"]
            cum_uav_dt_go += last_info["uav_dt_go"]

        episode.user_data["uav_collisions"].append(cum_uav_collisions)
        episode.user_data["obstacle_collisions"].append(cum_obstacle_collisions)
        episode.user_data["uav_dt_go"].append(cum_uav_dt_go)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        agent_ids = episode.get_agents()
        num_agents = len(agent_ids)
        cum_uav_landed = 0.0
        cum_uav_done_dt = 0.0
        cum_uav_rel_dist = 0.0

        for agent_id in agent_ids:
            # last_info = episode.last_info_for(agent_id)
            last_info = episode._last_infos[agent_id]
            cum_uav_rel_dist += last_info["uav_rel_dist"]
            cum_uav_landed += last_info["uav_landed"]
            cum_uav_done_dt += last_info["uav_done_dt"]

        obstacle_collisions = (
            np.sum(episode.user_data["obstacle_collisions"]) / num_agents
        )
        episode.custom_metrics["obstacle_collisions"] = obstacle_collisions
        uav_collisions = np.sum(episode.user_data["uav_collisions"]) / num_agents
        episode.custom_metrics["uav_collisions"] = uav_collisions
        uav_dt_go = np.mean(episode.user_data["uav_dt_go"]) / num_agents
        episode.custom_metrics["uav_dt_go"] = uav_dt_go
        uav_landed = cum_uav_landed / num_agents
        episode.custom_metrics["num_uav_landed"] = uav_landed
        uav_done_dt = cum_uav_done_dt / num_agents
        episode.custom_metrics["uav_done_dt"] = uav_done_dt
        uav_rel_dist = cum_uav_rel_dist / num_agents
        episode.custom_metrics["uav_rel_dist"] = uav_rel_dist
