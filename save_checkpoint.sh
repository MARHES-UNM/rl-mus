#!/bin/bash

CHECKPOINT_SAME_TIME="/workspaces/multi-uav-sim/ray_results/train/PPO/multi-uav-ren-v0_2025-05-03-13-31_cc94735_4u_4o/final_train/final_train/PPO_multi-uav-ren-v0_6b51f_00000_0_early_done=False,max_dt_go_error=0.1000,max_dt_std=0.0500,sa_reward=50,stp_penalty=1,t_go_error_2025-05-03_13-31-17/checkpoint_000275"

cp -rf ${CHECKPOINT_SAME_TIME}/* checkpoints/uav_same_time
cp -rf ${CHECKPOINT_SAME_TIME}/../params.json checkpoints/uav_same_time/.
cp -rf ${CHECKPOINT_SAME_TIME}/../progress.csv checkpoints/uav_same_time/.
cp -rf ${CHECKPOINT_SAME_TIME}/../result.json checkpoints/uav_same_time/.

CHECKPOINT_CUR="/workspaces/multi-uav-sim/ray_results/train/PPO/multi-uav-ren-curriculum-v0_2025-05-03-19-43_cc94735_4u_4o/final_train_cur/final_train_cur/PPO_multi-uav-ren-curriculum-v0_60238_00000_0_early_done=False,max_dt_go_error=0.1000,max_dt_std=0.0500,sa_reward=50,stp_penalty=1_2025-05-03_19-43-12/checkpoint_000305"
cp -rf ${CHECKPOINT_CUR}/* checkpoints/uav_cur
cp -rf ${CHECKPOINT_CUR}/../params.json checkpoints/uav_cur/.
cp -rf ${CHECKPOINT_CUR}/../progress.csv checkpoints/uav_cur/.
cp -rf ${CHECKPOINT_CUR}/../result.json checkpoints/uav_cur/.


