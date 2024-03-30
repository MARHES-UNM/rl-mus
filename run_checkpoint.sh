#!/bin/bash

CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-sim-curriculum-v0_2024-03-30-16-26_0c0629b_4u_4o/uav_base_cur/uav_base_cur/PPO_multi-uav-sim-curriculum-v0_bebaa_00000_0_t_go_max=2.0000,time_final=8.0000,uav_type=UavBase,use_safe_action=False_2024-03-30_16-26-10/checkpoint_000180/policies/shared_policy"

python run_experiment.py --run PPO --tf 4 test \
    --checkpoint $CHECKPOINT \
    --render \
    --plot_results \
    --seed None