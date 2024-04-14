#!/bin/bash

CHECKPOINT="checkpoints/cur_learning/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-14-02_b176a4f_4u_4o/sim_tf/sim_tf/PPO_multi-uav-ren-v0_c9998_00001_1_stp_penalty=1.0000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_14-02-12/checkpoint_000070/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-17-33_c70c262_4u_4o/raw_tg_go/raw_tg_go/PPO_multi-uav-ren-v0_453d3_00001_1_stp_penalty=0.5000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_17-33-15/checkpoint_000245/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-17-33_c70c262_4u_4o/raw_tg_go/raw_tg_go/PPO_multi-uav-ren-v0_453d3_00000_0_stp_penalty=1.0000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_17-33-15/checkpoint_000265/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-19-42_15a8f1d_4u_4o/raw_tg_go/raw_tg_go/PPO_multi-uav-ren-v0_5ea04_00003_3_stp_penalty=0.8500,tgt_reward=50,uav_type=UavBase,use_safe_action=False_2024-04-12_19-42-48/checkpoint_000305/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-14-00-35_a927c0e_4u_4o/safe_vary_beta/safe_vary_beta/PPO_multi-uav-ren-v0_707e3_00000_0_beta=0.1000,stp_penalty=0.9000,tgt_reward=50,uav_type=UavBase,use_safe_action=False_2024-04-14_00-35-36/checkpoint_000457/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-14-08-19_8f46de9_4u_4o/rebaseline_85/rebaseline_85/PPO_multi-uav-ren-v0_330d5_00000_0_beta=0.1000,stp_penalty=0.8500,tgt_reward=50,uav_type=UavBase,use_safe_action=False_2024-04-14_08-19-11/checkpoint_000340/policies/shared_policy"

# CHECKPOINT="/home/prime/Documents/workspace/rl_multi_uav_sim/ray_results/train/PPO/multi-uav-ren-v0_2024-04-12-14-02_b176a4f_4u_4o/sim_tf/sim_tf/PPO_multi-uav-ren-v0_c9998_00000_0_stp_penalty=0.1000,tgt_reward=20,uav_type=UavBase,use_safe_action=False_2024-04-12_14-02-12/checkpoint_000085/policies/shared_policy"

python run_experiment.py --run PPO --tf 8 test \
    --checkpoint $CHECKPOINT \
    --render \
    --seed None \
    --max_num_episodes 1 \
    --plot_results \
