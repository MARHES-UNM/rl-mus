#!/bin/bash/env bash
python plot_results.py --exp_folder "/home/prime/Documents/workspace/rl_multi_uav_sim/results/test_results/exp_2023-11-21-11-15_0f2a08c" --exp_config configs/exp_long_cfg_rl_paper.json --skip_legend
python plot_results.py --exp_folder "/home/prime/Documents/workspace/rl_multi_uav_sim/results/test_results/exp_2023-11-21-11-15_0f2a08c" --exp_config configs/exp_long_cfg_ppo_cur_comp.json --skip_legend --img_folder "ppo_cur_comparsion"
# path to paper: /home/prime/Documents/workspace/uav_sim/results/test_results/exp_2023-10-13-23-26_e1293db
