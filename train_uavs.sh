#!/bin/bash

# python multi_agent_shared_parameter.py --name mulagen_par --stop-timesteps 20000000
# python train_agent.py --name cur_l --stop-timesteps 20000000
# python run_experiment.py --name run_exp --stop-timesteps 20000000
# python run_experiment.py --name baseline --run PPO \
#     train \
#     --cpu 8 \
#     --gpu 0.5 \
#     --stop_timesteps 20000000

# comparing curriculum learning
python run_experiment.py --name uav_base_cur --run PPO \
    --env_name "multi-uav-sim-curriculum-v0" \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000

# python run_experiment.py --name uav_base_no_cur --run PPO \
#     train \
#     --cpu 8 \
#     --gpu 0.5 \
#     --stop_timesteps 20000000
