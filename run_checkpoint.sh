#!/bin/bash

CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-sim-v0_2024-03-29-15-17_c32edbf_4u_4o/uav_base_no_cur/uav_base_no_cur/PPO_multi-uav-sim-v0_06011_00000_0_t_go_max=2.0000,time_final=20.0000,uav_type=Uav,use_safe_action=False_2024-03-29_15-17-45/checkpoint_000302/policies/shared_policy"

python run_experiment.py test --checkpoint $CHECKPOINT --render --plot_results