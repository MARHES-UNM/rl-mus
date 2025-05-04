#!/bin/bash

exp_no_vl="results/test_results/ppo_ren_comp_no_vl/exp_2025-05-02-00-51_ee8d35d_vl_0_ed_0"
exp_vl="results/test_results/ppo_ren_comp_vl/exp_2025-05-02-00-54_ee8d35d_vl_1_ed_0"
output_folder="sim_figs"

python plot_results.py --exp_config configs/exp_ren_paper_no_vl_concensus_comp.json \
    --exp_folder ${exp_no_vl} \
    --out_folder ${output_folder} \
    --img_folder no_vl_concensus_comp \
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