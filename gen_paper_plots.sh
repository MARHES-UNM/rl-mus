#!/bin/bash

python plot_results.py --exp_config configs/exp_ren_paper_no_vl_concensus_comp.json \
    --exp_folder paper_test_results/exp_2024-04-27-20-44_9fab435_vl_0 \
    --out_folder paper_figs \
    --img_folder no_vl_concensus_comp \
    --skip_legend --plot_type bar

python plot_results.py --exp_config configs/exp_ren_paper_no_vl_ppo_ren_comp.json \
    --exp_folder paper_test_results/exp_2024-04-27-20-44_9fab435_vl_0 \
    --out_folder paper_figs \
    --img_folder no_vl_ren_comparison \
    --skip_legend --plot_type bar

python plot_results.py --exp_config configs/exp_ren_paper_vl_ppo_ttc_comp.json \
    --exp_folder paper_test_results/exp_2024-04-28-00-50_df2ae32_vl_1 \
    --out_folder paper_figs \
    --img_folder vl_ren_comparison \
    --skip_legend --plot_type bar