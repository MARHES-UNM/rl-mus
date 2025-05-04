#!/bin/bash

python run_multi_experiment.py --exp_config configs/exp_long_cfg_rl_ren_full_no_vl.json --num_eps 50
python run_multi_experiment.py --exp_config configs/exp_long_cfg_rl_ren_full_vl.json --num_eps 50
