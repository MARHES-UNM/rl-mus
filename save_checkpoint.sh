#!/bin/bash

CHECKPOINT_SAME_TIME="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-27-08-59_47fb018_4u_4o/replicate_dc8b9b2_02_05/replicate_dc8b9b2_02_05/PPO_multi-uav-ren-v0_0cf50_00000_0_early_done=False,max_dt_go_error=0.2000,max_dt_std=0.0500,stp_penalty=1.5000,t_go_error_func=su_2024-04-27_08-59-58/checkpoint_000457"

cp -rf ${CHECKPOINT_SAME_TIME}/* checkpoints/uav_same_time
cp -rf ${CHECKPOINT_SAME_TIME}/../params.json checkpoints/uav_same_time/.
cp -rf ${CHECKPOINT_SAME_TIME}/../progress.csv checkpoints/uav_same_time/.
cp -rf ${CHECKPOINT_SAME_TIME}/../result.json checkpoints/uav_same_time/.

CHECKPOINT_GOAL="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-27-08-59_47fb018_4u_4o/replicate_dc8b9b2_02_05/replicate_dc8b9b2_02_05/PPO_multi-uav-ren-v0_0cf50_00002_2_early_done=False,max_dt_go_error=0.2000,max_dt_std=0.0500,stp_penalty=0.0000,t_go_error_func=su_2024-04-27_08-59-58/checkpoint_000457"
cp -rf ${CHECKPOINT_GOAL}/* checkpoints/uav_goal
cp -rf ${CHECKPOINT_GOAL}/../params.json checkpoints/uav_goal/.
cp -rf ${CHECKPOINT_GOAL}/../progress.csv checkpoints/uav_goal/.
cp -rf ${CHECKPOINT_GOAL}/../result.json checkpoints/uav_goal/.


