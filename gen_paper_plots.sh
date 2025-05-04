#!/bin/bash

exp_no_vl="results/test_results/ppo_ren_comp_no_vl/exp_2025-05-04-11-43_fd8177f_vl_0_ed_0"
exp_vl="results/test_results/ppo_ren_comp_vl/exp_2025-05-04-11-59_fd8177f_vl_1_ed_0"
output_folder="sim_results"

python plot_results.py --exp_config configs/exp_ren_paper_no_vl_concensus_comp.json \
    --exp_folder ${exp_no_vl} \
    --out_folder ${output_folder} \
    --img_folder no_vl_cur_comp \
    --skip_legend --plot_type bar

python plot_results.py --exp_config configs/exp_ren_paper_no_vl_ppo_ren_comp.json \
    --exp_folder ${exp_no_vl} \
    --out_folder ${output_folder} \
    --img_folder no_vl_ren_comparison \
    --skip_legend --plot_type bar

python plot_results.py --exp_config configs/exp_ren_paper_vl_ppo_ttc_comp.json \
    --exp_folder ${exp_vl} \
    --out_folder ${output_folder} \
    --img_folder vl_ren_comparison \
    --skip_legend --plot_type bar