#!/bin/bash

python run_experiment.py --name large_obs_pen_sa_rew --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
