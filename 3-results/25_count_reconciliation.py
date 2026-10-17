"""One table that says where every count of resolved units in the manuscript comes from.

The main text reports several counts of resolved states (8, 9, 10, 13) and of resolved
immediate regions (19, 37, 42, 43, 44, 46, 53, 62, 69). None is a typo: each belongs to a
different fit, hierarchy, chain length or criterion. An audit found the state counts
unreconciled in the text and the 13 untraced. This script reads every one of them back
from the committed artefact that holds it and writes a single table, so the supplement
can print the reconciliation and a reader can see which number is comparable to which.

Nothing is refitted. Every row names the script that produced the artefact it reads.

Run:
    NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/25_count_reconciliation.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import model_dir, results_dir  # noqa: E402
MODEL = model_dir(ROOT)
OUT = results_dir(ROOT)
TAB = OUT / "tables"


def main():
    lad = pd.read_csv(TAB / "aggregation_ladder.csv").set_index("level")
    orph = json.loads((OUT / "ORPHAN_NUMBERS.json").read_text())
    sd = pd.read_csv(TAB / "state_dispersion.csv")
    rob = pd.read_csv(TAB / "spatial_robustness.csv").set_index("prior")
    disp = pd.read_csv(TAB / "dispersion_comparison.csv", index_col=0)
    cov = pd.read_csv(TAB / "covariate_models.csv").set_index("variant")
    shape = pd.read_csv(TAB / "trend_shape.csv").set_index("trend")
    seeds = json.loads((OUT / "SEED_STABILITY.json").read_text())
    hr = json.loads((OUT / "HEALTH_REGION.json").read_text())
    srate = pd.read_csv(MODEL / "slopes_rate.csv")
    sstate = pd.read_csv(MODEL / "slopes_state.csv")
    sladder_state = pd.read_csv(MODEL / "slopes_ladder_state.csv")
    bh = pd.read_csv(TAB / "bh_rate.csv")

    rows = [
        # ---- states -------------------------------------------------------------
        dict(level="State", count=int(sladder_state.classified_95.sum()), of=27,
             model="Two-level hierarchical (national + state), negative binomial, single dispersion",
             chain="2,500 tuning / 1,500 sampling, target acceptance 0.95 (aggregation-ladder settings)",
             criterion="95% CrI of the state slope excludes zero",
             script="2-model/09_aggregation_ladder.py", artefact="tables/aggregation_ladder.csv; 2-model/<outcome>/slopes_ladder_state.csv",
             where="Table 1; eTable 10; Comment"),
        dict(level="State", count=int(sstate.classified_95.sum()), of=27,
             model="Two-level hierarchical (national + state), negative binomial, single dispersion",
             chain="3,000 tuning / 2,000 sampling, target acceptance 0.99 (primary settings)",
             criterion="95% CrI of the state slope excludes zero",
             script="2-model/07_state_model.py", artefact="2-model/<outcome>/slopes_state.csv",
             where="Figure 1A legend"),
        dict(level="State", count=int(sd.classified_exposure.astype(bool).sum()), of=27,
             model="Two-level hierarchical (national + state), negative binomial, exposure-dependent dispersion",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI of the state slope excludes zero",
             script="2-model/15_state_dispersion.py", artefact="tables/state_dispersion.csv (classified_exposure)",
             where="not reported in the main text; eTable 7 shows the slopes"),
        dict(level="State", count=int(orph["states_resolved_from_region_model"]), of=27,
             model="Three-level hierarchical (national + state + immediate region), negative binomial, single dispersion: the primary region model read at its state level (mu_b + b_state)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI of mu_b + b_state excludes zero",
             script="3-results/17_orphan_numbers.py (reads 2-model/02_canonical_fit.py output)",
             artefact="ORPHAN_NUMBERS.json (states_resolved_from_region_model)",
             where="Results, 'Resolution by geographic scale'"),
        dict(level="State", count=int(lad.loc["state", "classified_unpooled"]), of=27,
             model="Unpooled quasi-Poisson fit of each state's own 11-year series",
             chain="none (maximum likelihood)",
             criterion="|slope| > t(0.975, 9 df) x SE, Pearson dispersion",
             script="2-model/09_aggregation_ladder.py (unpooled_ladder)", artefact="tables/aggregation_ladder.csv (classified_unpooled)",
             where="Table 1; Results"),
        # ---- immediate regions --------------------------------------------------
        dict(level="Immediate region", count=int(lad.loc["immediate region", "classified"]), of=510,
             model="Two-level hierarchical (national + region), negative binomial, single dispersion",
             chain="2,500 / 1,500, target acceptance 0.95 (aggregation-ladder settings)",
             criterion="95% CrI excludes zero",
             script="2-model/09_aggregation_ladder.py", artefact="tables/aggregation_ladder.csv",
             where="Table 1; eTable 10; eTable 11"),
        dict(level="Immediate region", count=int(hr["comparison_immediate_region"]["classified"]), of=510,
             model="Same two-level ladder fit, reported beside the 433 health regions",
             chain="2,500 / 1,500, target acceptance 0.95",
             criterion="95% CrI excludes zero",
             script="2-model/19_health_region.py", artefact="HEALTH_REGION.json",
             where="eTable 11; Comment"),
        dict(level="Immediate region", count=int(srate.classified_95.sum()), of=510,
             model="Three-level hierarchical (national + state + region), negative binomial, single dispersion: the primary (canonical) fit",
             chain="3,000 / 2,000, target acceptance 0.99, seed 20260803",
             criterion="95% CrI excludes zero",
             script="2-model/02_canonical_fit.py", artefact="2-model/<outcome>/slopes_rate.csv; NUMBERS.json",
             where="Abstract; Synopsis; Results; Figure 1B; Supplementary Data File 1"),
        dict(level="Immediate region", count=int(bh.bh_fdr05.sum()), of=510,
             model="Same primary fit, Benjamini-Hochberg at 5% false discovery rate over the 510 posterior z-scores",
             chain="as above",
             criterion="BH-adjusted two-sided normal p-value <= 0.05",
             script="3-results/03_precision_report.py", artefact="tables/bh_rate.csv; NUMBERS.json (bh_rate)",
             where="Abstract; Results, 'Design analysis'; Conclusions"),
        dict(level="Immediate region", count=int(disp.loc["global", "classified"]), of=510,
             model="Three-level hierarchical, single dispersion, refitted by the predictive-check script (same model as the primary fit, separate run)",
             chain="3,000 / 2,000, target acceptance 0.99, seed 20260803",
             criterion="95% CrI excludes zero",
             script="2-model/08_ppc_and_dispersion.py", artefact="tables/dispersion_comparison.csv (global)",
             where="Results, 'Sensitivity analyses' (dispersion sentence)"),
        dict(level="Immediate region", count=int(disp.loc["exposure", "classified"]), of=510,
             model="Three-level hierarchical, exposure-dependent dispersion",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/08_ppc_and_dispersion.py", artefact="tables/dispersion_comparison.csv (exposure)",
             where="Results, 'Sensitivity analyses'; eTable 12"),
        dict(level="Immediate region", count=int(cov.loc["base", "classified_slope"]), of=510,
             model="Three-level hierarchical, single dispersion, base row of the covariate comparison (same model as the primary fit, separate run)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/14_covariates.py", artefact="tables/covariate_models.csv (base)",
             where="Results, 'Sensitivity analyses' ('reference model'); eTable 3"),
        dict(level="Immediate region", count=int(shape.loc["linear", "classified"]), of=510,
             model="Three-level hierarchical, single dispersion, linear row of the trend-shape comparison (same model as the primary fit, separate run)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/13_flexible_trend.py", artefact="tables/trend_shape.csv (linear)",
             where="eTable 12 note; S4.4"),
        dict(level="Immediate region", count=int(rob.loc["exch", "classified"]), of=510,
             model="Three-level hierarchical, exchangeable slope prior, refitted by the spatial-robustness script (same prior as the primary fit, separate run)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/04_spatial_robustness.py", artefact="tables/spatial_robustness.csv (exch)",
             where="Table 3"),
        dict(level="Immediate region", count=int(rob.loc["nopool", "classified"]), of=510,
             model="Bayesian, no pooling on the slope: independent Normal(0, 0.25) slope deviations, hierarchical intercepts, single dispersion",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/04_spatial_robustness.py", artefact="tables/spatial_robustness.csv (nopool)",
             where="Table 3"),
        dict(level="Immediate region", count=int(rob.loc["bym2", "classified"]), of=510,
             model="Three-level hierarchical, BYM2 spatial slope prior (queen contiguity)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/04_spatial_robustness.py", artefact="tables/spatial_robustness.csv (bym2)",
             where="Table 3"),
        dict(level="Immediate region", count=int(lad.loc["immediate region", "classified_unpooled"]), of=510,
             model="Unpooled quasi-Poisson fit of each region's own 11-year series (unit-specific Pearson dispersion)",
             chain="none (maximum likelihood)",
             criterion="|slope| > t(0.975, 9 df) x SE",
             script="2-model/09_aggregation_ladder.py (unpooled_ladder); 3-results/17_orphan_numbers.py",
             artefact="tables/aggregation_ladder.csv; ORPHAN_NUMBERS.json (unpooled_sign_split)",
             where="Table 1; Results"),
        dict(level="Immediate region", count=int(cov.loc["slope", "classified_slope"]), of=510,
             model="Three-level hierarchical with the development index on the slope (covariate-assisted; excluded from the 19 to 62 range)",
             chain="3,000 / 2,000, target acceptance 0.99",
             criterion="95% CrI excludes zero",
             script="2-model/14_covariates.py", artefact="tables/covariate_models.csv (slope)",
             where="Results, 'Sensitivity analyses'; eTable 3"),
        dict(level="Immediate region", count=f"{seeds['min']} to {seeds['max']}", of=510,
             model="Three-level hierarchical, single dispersion, eight sampling seeds",
             chain="as the primary fit, seeds 11 to 88",
             criterion="95% CrI excludes zero",
             script="2-model/12_seed_stability.py", artefact="SEED_STABILITY.json",
             where="Results, 'Sensitivity analyses'"),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "count_reconciliation.csv", index=False)
    print(table[["level", "count", "of", "model", "where"]].to_string(index=False))

    note = {
        "states": {"ladder_fit": int(sladder_state.classified_95.sum()),
                   "state_model": int(sstate.classified_95.sum()),
                   "state_model_exposure_dispersion": int(sd.classified_exposure.astype(bool).sum()),
                   "state_components_of_region_model": int(orph["states_resolved_from_region_model"]),
                   "unpooled": int(lad.loc["state", "classified_unpooled"])},
        "regions": {"ladder_fit": int(lad.loc["immediate region", "classified"]),
                    "primary_fit": int(srate.classified_95.sum()),
                    "primary_fit_bh_fdr05": int(bh.bh_fdr05.sum()),
                    "ppc_refit_global": int(disp.loc["global", "classified"]),
                    "exposure_dispersion": int(disp.loc["exposure", "classified"]),
                    "covariate_base": int(cov.loc["base", "classified_slope"]),
                    "trend_linear": int(shape.loc["linear", "classified"]),
                    "prior_exch": int(rob.loc["exch", "classified"]),
                    "prior_nopool_bayesian": int(rob.loc["nopool", "classified"]),
                    "prior_bym2": int(rob.loc["bym2", "classified"]),
                    "unpooled_quasipoisson": int(lad.loc["immediate region", "classified_unpooled"]),
                    "covariate_slope": int(cov.loc["slope", "classified_slope"]),
                    "seeds_min": seeds["min"], "seeds_max": seeds["max"]},
    }
    (OUT / "COUNT_RECONCILIATION.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
