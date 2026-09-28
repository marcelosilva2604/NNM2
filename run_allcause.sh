#!/bin/zsh
# Re-run the whole pipeline with all-cause neonatal deaths as the outcome.
# Writes to 2-model/allcause/ and 3-results/allcause/; touches nothing else.
export NNM2_OUTCOME=deaths_total
cd "$(dirname "$0")"
LOG=3-results/allcause/run_all.log
mkdir -p 3-results/allcause 2-model/allcause
echo "START $(date)" >> $LOG
for s in 2-model/02_canonical_fit.py 2-model/07_state_model.py 2-model/09_aggregation_ladder.py \
         2-model/04_spatial_robustness.py 2-model/08_ppc_and_dispersion.py 2-model/12_seed_stability.py \
         2-model/13_flexible_trend.py 2-model/14_covariates.py 2-model/15_state_dispersion.py \
         2-model/19_health_region.py 3-results/16_design_two_arm.py 3-results/17_orphan_numbers.py \
         3-results/18_transferability.py 3-results/03_precision_report.py 3-results/07_vs_national.py \
         3-results/11_supporting_checks.py 3-results/20_state_contrast_ci.py \
         3-results/05_tables_figures.py 3-results/06_map.py; do
  echo "=== $s  $(date)" >> $LOG
  .venv/bin/python "$s" > "3-results/allcause/$(basename $s .py).out" 2>&1
  echo "=== exit $?  $(date)" >> $LOG
done
echo "END $(date)" >> $LOG
