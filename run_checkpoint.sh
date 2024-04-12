#!/bin/bash

CHECKPOINT="checkpoints/cur_learning/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-14-02_b176a4f_4u_4o/sim_tf/sim_tf/PPO_multi-uav-ren-v0_c9998_00001_1_stp_penalty=1.0000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_14-02-12/checkpoint_000070/policies/shared_policy"
# CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-14-02_b176a4f_4u_4o/sim_tf/sim_tf/PPO_multi-uav-ren-v0_c9998_00000_0_stp_penalty=0.1000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_14-02-12/checkpoint_000085/policies/shared_policy"

python run_experiment.py --run PPO --tf 4 test \
    --checkpoint $CHECKPOINT \
    --render \
    --plot_results \
    --seed None
