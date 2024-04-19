#!/bin/bash

python run_experiment.py --name t_go_done_zero --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
