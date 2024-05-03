#!/bin/bash

python run_experiment.py --name get_other_uav_td_er --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
