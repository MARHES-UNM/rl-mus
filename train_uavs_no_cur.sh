#!/bin/bash

python run_experiment.py --name stp_pen_sum --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
