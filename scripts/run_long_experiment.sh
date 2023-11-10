#!/bin/bash
# NUM_EPS=200
NUM_EPS=100
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-30-14-50_2e22ac1/obstacle_4/train_safety_layer_2e31d_00000_0_target_v=1.0000,batch_size=256,eps=0.0500,eps_deriv=0.0150,loss_action_weight=0.1000,lr=0.0005,nu_2023-09-30_14-50-10/checkpoint_000319/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-17-02-37_efe0f87/full_dynamics/train_safety_layer_a2e58_00001_1_target_v=1.0000,batch_size=256,eps=0.0500,eps_deriv=0.0150,loss_action_weight=0.1000,lr=0.0005,nu_2023-09-17_02-37-15/checkpoint_000499/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-03-16-15_50dfe4b/test_refactor/train_safety_layer_a85ae_00001_1_target_v=1.0000,batch_size=256,eps=0.1000,eps_deriv=0.0300,loss_action_weight=0.1000,lr=0.0005,nu_2023-10-03_16-15-57/checkpoint_000199/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-04-05-55_81ac3a1/inc_h_loss/train_safety_layer_29a84_00004_4_max_num_obstacles=4,num_obstacles=4,obstacle_radius=1.0000,target_v=0.0000,batch_size=256,eps=0.1_2023-10-04_13-27-51/checkpoint_000269/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-06-13-14_0f2a865/unscale_u/train_safety_layer_c57ce_00001_1_obstacle_radius=0.1000,loss_action_weight=0.1000_2023-10-06_13-14-15/checkpoint_000059/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-06-13-14_0f2a865/unscale_u/train_safety_layer_c57ce_00000_0_obstacle_radius=1.0000,loss_action_weight=0.1000_2023-10-06_13-14-13/checkpoint_000064/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-07-00-47_4700c4b/best_acc_weight/train_safety_layer_a744f_00000_0_obstacle_radius=1.0000,batch_size=1024,loss_action_weight=0.1000_2023-10-07_00-47-43/checkpoint_000384/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-07-08-48_dfd9449/best_acc_weight/train_safety_layer_c92f4_00001_1_obstacle_radius=0.1000,target_v=0.0000,batch_size=1024,eps_action=0.0002,loss_action_weight=0.100_2023-10-07_08-48-18/checkpoint_000499/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-13-01-02_ba7765a/no_deepset/train_safety_layer_a7e97_00000_0_obstacle_radius=1.0000,target_v=0.0000,batch_size=1024,eps_action=0.0000,eps_dang=0.0500,eps_deri_2023-10-13_01-02-06/checkpoint_000499/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-10-13-01-00_b7e0467/deepset/train_safety_layer_7708f_00000_0_obstacle_radius=1.0000,target_v=0.0000,batch_size=1024,eps_action=0.0000,eps_dang=0.0500,eps_deri_2023-10-13_01-00-44/checkpoint_000499/checkpoint"
NN_CBF_DIR="/home/prime/Documents/workspace/rl_multi_uav_sim/results/safety_layer/safety_layer2023-11-03-07-18_b178d34/test_rl/checkpoint_499/checkpoint"
EXP_CONFIG="configs/exp_long_cfg.json"
# EXP_CONFIG="configs/exp_long_cfg_sca.json"
# EXP_CONFIG="configs/exp_basic_cfg.json"

# python run_multi_experiment.py --exp_config ${EXP_CONFIG} --nn_cbf_dir ${NN_CBF_DIR} --num_eps ${NUM_EPS}
python run_multi_experiment.py --exp_config ${EXP_CONFIG} --num_eps ${NUM_EPS}
