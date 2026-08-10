"""Design analysis in two arms: what a local reader can detect, and what the model can.

The classification rate is a property of Brazil's own distribution of true slopes, its
exposures and the prior, so it is not a transportable statement about the registration
system. A design analysis is: fix true slopes, simulate at each unit's real exposure,
refit, and measure how often the truth is recovered and how badly it is distorted when
it is.

Two defects in the first version of this analysis are repaired here.

    Dispersion. Counts were simulated under a single dispersion (alpha 56.4) that the
    project's own posterior predictive check rejects, most severely in the largest units,
    which is exactly where the headline sentence lives. Both arms now simulate under the
    exposure-dependent dispersion, drawing each region's alpha from the fitted
    log alpha_i = c0 + c1 log(mean births).

    Degenerate generating process. Every unit shared one true slope. Under that process a
    hierarchical model would estimate the between-unit spread near zero and classify
    almost everything, so evaluating it would be vacuous, and the unpooled arm was the
    only honest choice. That is a property of the simulation, not a virtue of the
    procedure, and the first version did not say so.

    ARM A  fixed true slope, unpooled refit. Characterises what a reader of one unit's
           own series can detect. Comparable to the published Table 2.
    ARM B  true slopes drawn from the fitted heterogeneity, hierarchical refit. The only
           setting in which the procedure the paper actually uses can be evaluated
           without the answer being built into the generating process.

Arm B uses a shorter chain than the main analysis (2,000 tuning and 1,500 sampling
iterations against 3,000 and 2,000) because it is run many times; the classification
criterion is otherwise identical.

Run:
    .venv/bin/python 3-results/16_design_two_arm.py
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

RNG = np.random.default_rng(20260806)
SEED = 20260806

ARM_A_SIMS = 20          # per region, per scenario
ARM_B_REPLICATES = 20    # each yields 510 unit-level outcomes
MULTIPLES = (0.0, 0.5, 1.0, 2.0)

DRAWS_B, TUNE_B, CHAINS_B = 1500, 2000, 4


def setup():
    """Panel, per-region level, per-region dispersion, and the fitted heterogeneity."""
    panel = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    panel = panel.reset_index(drop=True)
    panel["t"] = panel.year - panel.year.mean()

    regions = np.sort(panel.rgi_id.unique())
    panel["region_idx"] = panel.rgi_id.map({r: i for i, r in enumerate(regions)})
    rs = panel[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    sor = rs.map({s: i for i, s in enumerate(states)}).values

    canonical = az.from_netcdf(MODEL / "idata_rate.nc")
    mu_b = float(canonical.posterior["mu_b"].mean())
    # Total between-unit spread of the true slope, state and region components combined.
    tau = float(
        np.sqrt(
            canonical.posterior["sigma_b_state"].mean() ** 2
            + canonical.posterior["sigma_b_region"].mean() ** 2
        )
    )

    # Per-region dispersion from the model the paper prefers, not the global one.
    disp = az.from_netcdf(MODEL / "idata_disp_exposure.nc")
    alpha = disp.posterior["alpha_region"].mean(dim=("chain", "draw")).values

    level = (
        panel.groupby("rgi_id")
        .apply(lambda g: np.log(g.avoidable.sum() / g.births.sum()), include_groups=False)
        .loc[regions]
        .values
    )
    return panel, regions, states, sor, mu_b, tau, alpha, level


def simulate(panel, level, alpha, true_b):
    """Draw counts for every region-year under known slopes and per-region dispersion."""
    idx = panel.region_idx.values
    mu = panel.births.values * np.exp(level[idx] + true_b[idx] * panel.t.values)
    a = alpha[idx]
    return RNG.negative_binomial(a, a / (a + mu))


def fit_unpooled(panel, y):
    """One quasi-Poisson slope per region, referred to t on n-2 residual df."""
    out = np.full((panel.region_idx.max() + 1, 3), np.nan)  # slope, se, declared
    frame = panel.assign(y=y)
    for idx, g in frame.groupby("region_idx"):
        if g.y.sum() == 0:
            continue
        try:
            f = sm.GLM(
                g.y, sm.add_constant(g[["t"]]),
                family=sm.families.Poisson(), offset=np.log(g.births),
            ).fit(scale="X2")
            b, se = float(f.params["t"]), float(f.bse["t"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
            continue
        crit = stats.t.ppf(0.975, len(g) - 2)
        out[idx] = (b, se, float(abs(b) > crit * se))
    return out


def fit_hierarchical(panel, y, n_regions, n_states, sor, alpha, seed):
    """The paper's own three-level model, refitted on simulated counts."""
    idx = panel.region_idx.values
    with pm.Model():
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa_s = pm.HalfNormal("sigma_a_state", 0.5)
        sa_r = pm.HalfNormal("sigma_a_region", 0.5)
        sb_s = pm.HalfNormal("sigma_b_state", 0.1)
        sb_r = pm.HalfNormal("sigma_b_region", 0.1)
        a = pm.Deterministic(
            "a", mu_a + (sa_s * pm.Normal("z_a_state", 0, 1, shape=n_states))[sor]
            + sa_r * pm.Normal("z_a_region", 0, 1, shape=n_regions))
        b = pm.Deterministic(
            "b", mu_b + (sb_s * pm.Normal("z_b_state", 0, 1, shape=n_states))[sor]
            + sb_r * pm.Normal("z_b_region", 0, 1, shape=n_regions))
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[idx] + b[idx] * panel.t.values + np.log(panel.births.values)),
            alpha=alpha[idx],
            observed=y,
        )
        idata = pm.sample(
            draws=DRAWS_B, tune=TUNE_B, chains=CHAINS_B, random_seed=seed,
            target_accept=0.95, progressbar=False,
        )
    draws = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(draws, 0.025, axis=1), np.quantile(draws, 0.975, axis=1)
    return draws.mean(axis=1), (lo > 0) | (hi < 0), int(idata.sample_stats.diverging.sum())


def summarise(records, label):
    """Power, type S and type M by exposure quintile."""
    df = pd.DataFrame(records)
    rows = []
    for q, sub in df.groupby("quintile"):
        decl = sub[sub.declared]
        rows.append({
            "arm": label,
            "exposure_quintile": int(q),
            "n": len(sub),
            "power": round(float(sub.declared.mean()), 3),
            "type_S": round(float(decl.wrong_sign.mean()), 4) if len(decl) else np.nan,
            "type_M": round(float(decl.ratio.median()), 2) if len(decl) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    panel, regions, states, sor, mu_b, tau, alpha, level = setup()
    pooled = panel.groupby("rgi_id").avoidable.sum().loc[regions].values
    quint = pd.qcut(pooled, 5, labels=False) + 1
    n_regions = len(regions)

    log = ["Design analysis, two arms", "=" * 70, "",
           f"national drift mu_b        : {mu_b:.5f}",
           f"between-unit slope SD (tau): {tau:.5f}",
           f"per-region alpha           : median {np.median(alpha):.1f}, "
           f"range {alpha.min():.1f} to {alpha.max():.1f}", ""]

    # ---------------- ARM A ----------------
    tables = []
    for mult in MULTIPLES:
        true_b = np.full(n_regions, mult * mu_b)
        records = []
        for _ in range(ARM_A_SIMS):
            y = simulate(panel, level, alpha, true_b)
            res = fit_unpooled(panel, y)
            for i in range(n_regions):
                b, se, decl = res[i]
                if not np.isfinite(b):
                    continue
                records.append({
                    "quintile": quint[i],
                    "declared": bool(decl),
                    "wrong_sign": bool(decl) and mult != 0
                                  and np.sign(b) != np.sign(true_b[i]),
                    "ratio": abs(b) / abs(true_b[i]) if (decl and mult != 0) else np.nan,
                })
        t = summarise(records, f"A: fixed {mult}x, unpooled")
        t["true_slope_x_national"] = mult
        tables.append(t)
        print(f"arm A done: {mult}x")

    # ---------------- ARM B ----------------
    records, divergences = [], 0
    for rep in range(ARM_B_REPLICATES):
        true_b = mu_b + RNG.normal(0.0, tau, size=n_regions)
        y = simulate(panel, level, alpha, true_b)
        bhat, decl, div = fit_hierarchical(
            panel, y, n_regions, len(states), sor, alpha, SEED + rep
        )
        divergences += div
        for i in range(n_regions):
            records.append({
                "quintile": quint[i],
                "declared": bool(decl[i]),
                "wrong_sign": bool(decl[i]) and np.sign(bhat[i]) != np.sign(true_b[i]),
                "ratio": abs(bhat[i]) / abs(true_b[i]) if decl[i] else np.nan,
                "count_flag": bool(decl[i]),
                "replicate": rep,
            })
        print(f"arm B replicate {rep + 1}/{ARM_B_REPLICATES}")

    b_tab = summarise(records, "B: drawn slopes, hierarchical")
    b_tab["true_slope_x_national"] = np.nan
    tables.append(b_tab)

    counts = pd.DataFrame(records).groupby("replicate").count_flag.sum()

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / "tables" / "design_two_arm.csv", index=False)

    a1 = table[(table.arm == "A: fixed 1.0x, unpooled")]
    summary = {
        "arm_A_power_at_1x_by_quintile": a1[["exposure_quintile", "power", "type_S", "type_M"]]
            .to_dict("records"),
        "arm_A_mean_power_at_1x": round(float(a1.power.mean()), 3),
        "arm_A_null_declaration_rate": round(
            float(table[table.arm == "A: fixed 0.0x, unpooled"].power.mean()), 4),
        "arm_B_power_by_quintile": b_tab[["exposure_quintile", "power", "type_S", "type_M"]]
            .to_dict("records"),
        "arm_B_mean_power": round(float(b_tab.power.mean()), 3),
        "arm_B_classification_count": {
            "mean": round(float(counts.mean()), 1),
            "sd": round(float(counts.std(ddof=1)), 1),
            "min": int(counts.min()), "max": int(counts.max()),
        },
        "arm_B_divergences_total": divergences,
        "tau": round(tau, 5),
        "alpha_median": round(float(np.median(alpha)), 1),
    }
    (OUT / "DESIGN_TWO_ARM.json").write_text(json.dumps(summary, indent=2))

    log += [table.to_string(index=False), "", json.dumps(summary, indent=2)]
    (OUT / "16_design_two_arm.log").write_text("\n".join(log) + "\n")
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
