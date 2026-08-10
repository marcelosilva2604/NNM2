"""Prospective design analysis: what this surveillance system could detect, and how badly.

The classification rate and the posterior-SD detection floor both depend on Brazil's own
distribution of true slopes and on the prior, so neither is a transportable property of
the registration system. A design analysis is: fix a set of true slopes, simulate each
unit's counts at its real births and its estimated dispersion, refit, and report how
often the truth is recovered.

Three quantities per exposure stratum and per assumed true slope:

    power    the probability of declaring the unit changed, at the study's own criterion
    type-S   among declarations, the probability the sign is wrong
    type-M   among declarations, the ratio of the estimated magnitude to the truth,
             that is, the exaggeration a manager would act on

Units are refitted unpooled, so the reported operating characteristics belong to the data
a unit generates and not to the hierarchical prior. This is deliberately the pessimistic
end; the pooled model classifies slightly more.

Run:
    .venv/bin/python 3-results/10_design_analysis.py
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

RNG = np.random.default_rng(20260803)
N_SIM = 200
# True slopes expressed as multiples of the estimated national drift, both directions.
MULTIPLES = (0.0, 0.5, 1.0, 2.0)


def setup():
    panel = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    panel["t"] = panel.year - panel.year.mean()

    idata = az.from_netcdf(MODEL / "idata_rate.nc")
    mu_b = float(idata.posterior["mu_b"].mean())
    alpha = float(idata.posterior["alpha"].mean())

    # Each unit keeps its own births and its own fitted level, so simulated counts have
    # the exposure profile the country actually has.
    levels = (
        panel.groupby("rgi_id")
        .apply(lambda g: np.log(g.avoidable.sum() / g.births.sum()), include_groups=False)
        .rename("level")
        .reset_index()
    )
    panel = panel.merge(levels, on="rgi_id")
    return panel, mu_b, alpha


def simulate_unit(g, true_b, alpha):
    """Draw one decade of counts for a unit under a known slope."""
    mu = g.births.values * np.exp(g.level.values[0] + true_b * g.t.values)
    # Negative binomial parameterised by mean and the fitted dispersion.
    p = alpha / (alpha + mu)
    return RNG.negative_binomial(alpha, p)


def fit_unit(g, y):
    """Unpooled quasi-Poisson slope and standard error from simulated counts."""
    if y.sum() == 0:
        return None
    x = sm.add_constant(g[["t"]])
    try:
        fit = sm.GLM(y, x, family=sm.families.Poisson(),
                     offset=np.log(g.births.values)).fit(scale="X2")
    except Exception:  # noqa: BLE001
        return None
    b, se = float(fit.params["t"]), float(fit.bse["t"])
    if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
        return None
    return b, se


def main():
    panel, mu_b, alpha = setup()
    pooled = panel.groupby("rgi_id").avoidable.sum()
    quintile = pd.qcut(pooled, 5, labels=False) + 1

    groups = {rgi: g for rgi, g in panel.groupby("rgi_id")}

    rows = []
    for mult in MULTIPLES:
        true_b = mult * mu_b  # mu_b is negative, so this is a decline of that size
        records = []
        for rgi, g in groups.items():
            for _ in range(N_SIM // 10 if len(groups) > 200 else N_SIM):
                y = simulate_unit(g, true_b, alpha)
                res = fit_unit(g, y)
                if res is None:
                    continue
                b, se = res
                declared = abs(b) > stats.t.ppf(0.975, len(g) - 2) * se
                records.append(
                    {
                        "rgi_id": rgi,
                        "q": int(quintile.loc[rgi]),
                        "declared": declared,
                        "wrong_sign": declared and (np.sign(b) != np.sign(true_b))
                        if true_b != 0
                        else False,
                        "ratio": abs(b) / abs(true_b) if (declared and true_b != 0)
                        else np.nan,
                    }
                )
        rec = pd.DataFrame(records)
        for q, sub in rec.groupby("q"):
            decl = sub[sub.declared]
            rows.append(
                {
                    "true_slope_x_national": mult,
                    "exposure_quintile": q,
                    "median_deaths": int(pooled[quintile == q].median()),
                    "power": round(float(sub.declared.mean()), 3),
                    "type_S": round(float(decl.wrong_sign.mean()), 4) if len(decl) else np.nan,
                    "type_M": round(float(decl.ratio.median()), 2) if len(decl) else np.nan,
                }
            )
        print(f"done: true slope = {mult}x national")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "design_analysis.csv", index=False)

    # The headline a surveillance designer needs: exposure required for 80% power at a
    # change the size of the national one.
    at_1x = table[table.true_slope_x_national == 1.0]
    summary = {
        "n_simulations_per_unit_per_scenario": N_SIM // 10 if len(groups) > 200 else N_SIM,
        "national_drift": round(mu_b, 5),
        "dispersion_alpha": round(alpha, 1),
        "power_at_national_size_change_by_quintile": at_1x[
            ["exposure_quintile", "median_deaths", "power", "type_S", "type_M"]
        ].to_dict("records"),
        "overall_power_at_national_size_change": round(float(at_1x.power.mean()), 3),
        "false_positive_rate_when_truth_is_zero": round(
            float(table[table.true_slope_x_national == 0.0].power.mean()), 4
        ),
    }
    (OUT / "DESIGN_ANALYSIS.json").write_text(json.dumps(summary, indent=2))

    text = "\n".join(
        ["Design analysis by simulation", "=" * 70, "",
         table.to_string(index=False), "", json.dumps(summary, indent=2)]
    )
    (OUT / "10_design_analysis.log").write_text(text + "\n")
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
