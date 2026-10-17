#!/bin/zsh
# Re-run the whole pipeline (all-cause neonatal deaths). Writes to 2-model/allcause/ and 3-results/allcause/.
# Prerequisites: data/raw/ and data/processed/ built by 1-data/01_build_panel.py, 02_build_adjacency.py,
# 03_build_covariates.py from the public SIM and SINASC records (see README).
export NNM2_OUTCOME=deaths_total
cd "$(dirname "$0")"
mkdir -p 3-results/allcause 2-model/allcause
for s in 2-model/02_canonical_fit.py 2-model/07_state_model.py 2-model/09_aggregation_ladder.py \
         2-model/04_spatial_robustness.py 2-model/28_scaled_bym2.py 2-model/08_ppc_and_dispersion.py \
         2-model/12_seed_stability.py 2-model/13_flexible_trend.py 2-model/14_covariates.py \
         2-model/15_state_dispersion.py 2-model/19_health_region.py \
         3-results/16_design_two_arm.py 3-results/23_power_threshold.py 3-results/29_matched_arms.py \
         3-results/30_matched_arms_summary.py 3-results/31_threshold_sensitivity.py \
         3-results/17_orphan_numbers.py 3-results/18_transferability.py 3-results/03_precision_report.py \
         3-results/07_vs_national.py 3-results/11_supporting_checks.py 3-results/20_state_contrast_ci.py \
         3-results/25_count_reconciliation.py 3-results/26_audit_aggregate_trend_regions.py \
         3-results/27_information_figure.py 3-results/32_supplementary_data_and_pdecline_map.py \
         3-results/33_power_figure_v3.py 3-results/05_tables_figures.py 3-results/06_map.py; do
  echo "=== $s  $(date)"
  .venv/bin/python "$s" || echo "=== $s exited with $?"
done
