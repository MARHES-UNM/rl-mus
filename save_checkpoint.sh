#!/bin/bash

CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-14-08-19_8f46de9_4u_4o/rebaseline_85/rebaseline_85/PPO_multi-uav-ren-v0_330d5_00000_0_beta=0.1000,stp_penalty=0.8500,tgt_reward=50,uav_type=UavBase,use_safe_action=False_2024-04-14_08-19-11/checkpoint_000457"

cp -rf ${CHECKPOINT}/* checkpoints/uav_same_time