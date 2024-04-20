#!/bin/bash

python run_experiment.py --name mean_t_g_error --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
