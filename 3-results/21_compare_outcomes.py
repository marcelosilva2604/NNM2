"""Side-by-side comparison of every headline quantity under the two outcomes.

Reads the artefacts written by the pipeline under 3-results/ (avoidable, the submitted
version) and 3-results/allcause/ (all-cause neonatal deaths) and prints one table. It is
a reading aid for the decision to switch the primary outcome; it fits nothing and it does
not feed the manuscript. Missing artefacts are reported as "pending" so the script can be
run while the all-cause pipeline is still writing.

Run:
    .venv/bin/python 3-results/21_compare_outcomes.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"avoidable": (ROOT / "2-model", ROOT / "3-results"),
        "all-cause": (ROOT / "2-model" / "allcause", ROOT / "3-results" / "allcause")}


def safe(fn):
    try:
        v = fn()
        return v
    except FileNotFoundError:
        return "pending"
    except Exception as exc:  # noqa: BLE001 - a broken artefact must be visible, not hidden
        return f"ERR {type(exc).__name__}"


def rows_for(model, res):
    T = res / "tables"
    out = {}

    def slopes():
        s = pd.read_csv(model / "slopes_rate.csv")
        return s
    out["regions resolved, three-level (of 510)"] = safe(lambda: int(slopes().classified_95.sum()))
    out["regions resolved, %"] = safe(lambda: round(100 * slopes().classified_95.mean(), 1))
    out["regions resolved AND rising"] = safe(lambda: int(((slopes().b_mean > 0) & slopes().classified_95).sum()))
    out["regions with positive point estimate"] = safe(lambda: int((slopes().b_mean > 0).sum()))
    out["median posterior SD of slope"] = safe(lambda: round(float(slopes().b_sd.median()), 4))
    out["median MDE (1.96 SD)"] = safe(lambda: round(float(slopes().mde.median()), 4))
    out["regions surviving BH-FDR 5%"] = safe(lambda: int(pd.read_csv(T / "bh_rate.csv").bh_fdr05.sum()))
    out["naive sign error % (1 - prob_direction > ...)"] = safe(
        lambda: round(100 * float((slopes().prob_direction < 0.5).mean()), 1))

    def tr():
        return json.loads((res / "TRANSFERABILITY.json").read_text())
    out["national slope mu_b (per year)"] = safe(lambda: tr()["completeness_bound"]["national_drift_mu_b"])
    out["national change per decade, %"] = safe(
        lambda: round(100 * (np.exp(10 * tr()["completeness_bound"]["national_drift_mu_b"]) - 1), 1))
    out["capture +1%/yr displaces slope by"] = safe(
        lambda: tr()["completeness_bound"]["displacement_if_capture_improves_1pct_per_year"])
    out["displacement as % of national slope"] = safe(
        lambda: round(100 * tr()["completeness_bound"]["displacement_as_share_of_national_drift"], 1))
    out["capture growth erasing the drift, %/yr"] = safe(
        lambda: tr()["completeness_bound"]["capture_growth_that_would_erase_the_drift_pct_per_year"])
    for key in ("lowest", "median", "highest"):
        out[f"threshold rule, {key}: deaths/unit-year"] = safe(lambda k=key: tr()["threshold_rule"][k]["deaths_per_unit_year"])
        out[f"threshold rule, {key}: power %"] = safe(lambda k=key: round(100 * tr()["threshold_rule"][k]["power"], 1))

    def lad():
        return pd.read_csv(T / "aggregation_ladder.csv").set_index("level")
    for lvl in ("municipality", "immediate region", "state", "macro-region"):
        out[f"ladder {lvl}: median deaths/unit"] = safe(lambda l=lvl: int(lad().loc[l, "median_deaths_per_unit"]))
        out[f"ladder {lvl}: hierarchical"] = safe(lambda l=lvl: f"{int(lad().loc[l, 'classified'])} ({lad().loc[l, 'pct_classified']}%)")
        out[f"ladder {lvl}: unpooled"] = safe(lambda l=lvl: f"{int(lad().loc[l, 'classified_unpooled'])} of {int(lad().loc[l, 'units_fitted'])} ({lad().loc[l, 'pct_unpooled']}%)")
    out["ladder worst R-hat"] = safe(lambda: float(lad().worst_rhat.max()))

    def st():
        return pd.read_csv(model / "slopes_state.csv")
    out["states resolved (state model, of 27)"] = safe(lambda: int(st().classified_95.sum()))
    out["states resolved AND rising"] = safe(lambda: int(((st().b_mean > 0) & st().classified_95).sum()))

    def sd():
        return pd.read_csv(T / "state_dispersion.csv")
    out["states departing, single dispersion"] = safe(lambda: int(sd().departs_global.sum()))
    out["states departing, exposure dispersion"] = safe(
        lambda: f"{int(sd().departs_exposure.sum())}: {', '.join(sd()[sd().departs_exposure].UF)}")
    out["states departing, unpooled"] = safe(
        lambda: f"{int(sd().departs_unpooled.sum())}: {', '.join(sd()[sd().departs_unpooled].UF)}")
    out["RJ own slope / rest / contrast"] = safe(
        lambda: f"{sd().set_index('UF').loc['RJ', 'slope_own']:.4f} / {sd().set_index('UF').loc['RJ', 'slope_rest']:.4f} / {sd().set_index('UF').loc['RJ', 'contrast']:.4f}")

    def het():
        return json.loads((res / "SUPPORTING_CHECKS.json").read_text())
    out["Cochran Q (26 df) / I2 % / tau"] = safe(
        lambda: f"{het()['between_state_heterogeneity']['Q']} / {het()['between_state_heterogeneity']['I2_pct']} / {het()['between_state_heterogeneity']['tau']}")
    out["common national shock share %"] = safe(lambda: round(100 * het()["common_national_shock"]["common_variance_share"], 2))
    out["linear trend rejected overall %"] = safe(lambda: het()["linearity"]["reject_linear_pct"])
    out["linear trend rejected, top quintile %"] = safe(lambda: het()["linearity"]["by_quintile"][-1]["reject_pct"])
    out["winner's curse ratio"] = safe(lambda: het()["winners_curse"]["inflation_ratio"])
    out["spearman illdef trend vs slope"] = safe(lambda: het()["coding_quality"]["spearman_illdef_trend_vs_slope"])

    def vs(level):
        return pd.read_csv(T / f"vs_national_{level}.csv")
    out["states separable from national drift"] = safe(lambda: int((vs("state").lagging | vs("state").leading).sum()))
    out["regions separable (leading/lagging)"] = safe(
        lambda: f"{int((vs('region').lagging | vs('region').leading).sum())} ({int(vs('region').leading.sum())}/{int(vs('region').lagging.sum())})")

    def d2():
        return json.loads((res / "DESIGN_TWO_ARM.json").read_text())
    out["arm A mean power at 1x national, %"] = safe(lambda: round(100 * d2()["arm_A_mean_power_at_1x"], 1))
    out["arm A power by quintile, %"] = safe(lambda: [round(100 * p, 1) for p in d2()["arm_A_power_at_1x_by_quintile"]])
    out["arm A null declaration rate, %"] = safe(lambda: round(100 * d2()["arm_A_null_declaration_rate"], 1))
    out["arm B mean power, %"] = safe(lambda: round(100 * d2()["arm_B_mean_power"], 1))
    out["arm B power by quintile, %"] = safe(lambda: [round(100 * p, 1) for p in d2()["arm_B_power_by_quintile"]])
    out["arm B classification count mean (min-max)"] = safe(
        lambda: f"{d2()['arm_B_classification_count']['mean']} ({d2()['arm_B_classification_count']['min']}-{d2()['arm_B_classification_count']['max']})")
    out["tau (between-region slope SD)"] = safe(lambda: d2()["tau"])

    def dt():
        return pd.read_csv(T / "design_two_arm.csv")
    def arm(a, x, col):
        s = dt()
        s = s[s.arm.str.startswith(a) & (s.true_slope_x_national == x)].sort_values("exposure_quintile")
        return s[col].tolist()
    out["arm A type S at 1x by quintile"] = safe(lambda: arm("A", 1.0, "type_S"))
    out["arm A type M at 1x by quintile"] = safe(lambda: arm("A", 1.0, "type_M"))
    out["arm B type S by quintile"] = safe(lambda: dt()[dt().arm.str.startswith("B")].sort_values("exposure_quintile").type_S.tolist())
    out["arm B type M by quintile"] = safe(lambda: dt()[dt().arm.str.startswith("B")].sort_values("exposure_quintile").type_M.tolist())

    def rope(kind):
        return pd.read_csv(T / f"rope_curve_{kind}.csv").set_index("tolerance_x_national")
    out["ROPE rate at 1x: changing / stable / indeterminate %"] = safe(
        lambda: f"{int(rope('rate').loc[1.0, 'drifting'])} / {int(rope('rate').loc[1.0, 'credibly_flat'])} / {round(100 * rope('rate').loc[1.0, 'indeterminate_share'], 1)}")
    out["ROPE rate at 2x: changing / stable"] = safe(
        lambda: f"{int(rope('rate').loc[2.0, 'drifting'])} / {int(rope('rate').loc[2.0, 'credibly_flat'])}")
    out["regions precise enough to be certified stable at 1x"] = safe(
        lambda: int((slopes().b_sd < abs(tr()["completeness_bound"]["national_drift_mu_b"]) / 1.96).sum()))

    def rob():
        return pd.read_csv(T / "spatial_robustness.csv").set_index("prior")
    for p in ("nopool", "exch", "bym2"):
        out[f"prior {p}: classified / median MDE"] = safe(
            lambda q=p: f"{int(rob().loc[q, 'classified'])} / {rob().loc[q].get('median_mde', rob().loc[q].iloc[-1])}")

    def shape():
        return pd.read_csv(T / "trend_shape.csv").set_index("trend")
    for s_ in ("linear", "quadratic", "rw"):
        out[f"trend {s_}: classified"] = safe(lambda q=s_: int(shape().loc[q, "classified"]))

    def seeds():
        return json.loads((res / "SEED_STABILITY.json").read_text())
    out["seeds: counts"] = safe(lambda: seeds()["counts"])
    out["seeds: always / never / flip"] = safe(
        lambda: f"{seeds()['regions_classified_in_every_run']} / {seeds()['regions_classified_in_no_run']} / {seeds()['regions_that_flip_between_runs']}")

    def disp():
        return pd.read_csv(T / "dispersion_comparison.csv", index_col=0)
    out["dispersion global -> exposure: classified"] = safe(
        lambda: f"{int(disp().loc['global', 'classified'])} -> {int(disp().loc['exposure', 'classified'])}")
    out["dispersion: SD ratio max/min global -> exposure"] = safe(
        lambda: f"{disp().loc['global', 'sd_ratio_max_min']} -> {disp().loc['exposure', 'sd_ratio_max_min']}")

    def cov():
        return pd.read_csv(T / "covariate_models.csv").set_index("variant")
    out["covariates: base / level / slope / timevar / mundlak"] = safe(
        lambda: " / ".join(str(int(cov().loc[v, "classified_slope"])) for v in ("base", "level", "slope", "timevar", "mundlak")))

    def hr():
        return json.loads((res / "HEALTH_REGION.json").read_text())
    out["health regions: median deaths / classified / unpooled"] = safe(
        lambda: f"{hr()['median_deaths_per_unit']} / {hr()['classified']} ({hr()['pct_classified']}%) / {hr()['classified_unpooled']} ({hr()['pct_unpooled']}%)")
    out["health regions: rising hierarchical / unpooled"] = safe(
        lambda: f"{hr()['rising_hierarchical']} / {hr()['rising_unpooled']}")

    def num():
        return json.loads((res / "NUMBERS.json").read_text())
    out["share outcome: classified"] = safe(lambda: num()["classified_share"])
    out["joint: both / rate only / share only"] = safe(
        lambda: f"{num()['joint_both']} / {num()['joint_rate_only']} / {num()['joint_share_only']}")
    out["exposure-dep. dispersion coefficient"] = safe(
        lambda: [p for p in num()["parameters"] if "exposure" in str(p).lower()][:1] or "see 08 log")
    return out


def main():
    cols = {name: rows_for(*paths) for name, paths in RUNS.items()}
    keys = list(cols["avoidable"])
    width = max(len(k) for k in keys)
    print(f"{'quantity':<{width}}  {'avoidable':<34}  all-cause")
    print("-" * (width + 72))
    for k in keys:
        a, b = cols["avoidable"].get(k), cols["all-cause"].get(k)
        flag = "" if a == b or "pending" in (str(a), str(b)) else "  *"
        print(f"{k:<{width}}  {str(a):<34}  {b}{flag}")


if __name__ == "__main__":
    main()
