#!/bin/bash

CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-14-08-19_8f46de9_4u_4o/rebaseline_85/rebaseline_85/PPO_multi-uav-ren-v0_330d5_00000_0_beta=0.1000,stp_penalty=0.8500,tgt_reward=50,uav_type=UavBase,use_safe_action=False_2024-04-14_08-19-11/checkpoint_000457"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-18-06-00_66c4c18_4u_4o/dt_go_fix_rew_long/dt_go_fix_rew_long/PPO_multi-uav-ren-v0_7d314_00000_0_beta=0.1000,obstacle_collision_weight=1.0000,sa_reward=50,stp_penalty=1.5000,tgt_reward=50,uav__2024-04-18_06-00-30/checkpoint_000457"

cp -rf ${CHECKPOINT}/* checkpoints/uav_same_time