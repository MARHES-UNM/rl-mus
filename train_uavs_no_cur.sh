#!/bin/bash

python run_experiment.py --name no_other_uav_td_er_64 --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
