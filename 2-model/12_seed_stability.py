"""How stable is the headline count, and why is it unstable?

Three runs of the same specification gave 84, 86 and 89 classified regions. Two distinct
things produce that spread and they call for different remedies.

    Monte Carlo error. Each region's posterior tail probability is estimated from a
    finite number of draws. This shrinks with more draws and can be quantified from a
    single fit; it does not require refitting the model many times.

    Boundary sensitivity. A binary rule applied to a continuous quantity is unstable for
    any unit whose posterior probability sits near the threshold. No amount of sampling
    removes this, because it is a property of the criterion rather than of the sampler.

This script measures both: it refits under several seeds to observe the spread directly,
and it counts the regions close enough to the decision boundary to account for it.

Run:
    .venv/bin/python 2-model/12_seed_stability.py
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "2-model"
RES = ROOT / "3-results"

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    df = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    df = df.reset_index(drop=True)
    df["t"] = df.year - df.year.mean()
    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})
    rs = df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    return df, regions, states, rs.map({s: i for i, s in enumerate(states)}).values


def build(df, n_regions, n_states, sor):
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa_s = pm.HalfNormal("sigma_a_state", 0.5)
        sa_r = pm.HalfNormal("sigma_a_region", 0.5)
        sb_s = pm.HalfNormal("sigma_b_state", 0.1)
        sb_r = pm.HalfNormal("sigma_b_region", 0.1)
        a = pm.Deterministic(
            "a",
            mu_a + (sa_s * pm.Normal("z_a_state", 0, 1, shape=n_states))[sor]
            + sa_r * pm.Normal("z_a_region", 0, 1, shape=n_regions),
        )
        b = pm.Deterministic(
            "b",
            mu_b + (sb_s * pm.Normal("z_b_state", 0, 1, shape=n_states))[sor]
            + sb_r * pm.Normal("z_b_region", 0, 1, shape=n_regions),
        )
        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[df.region_idx.values] + b[df.region_idx.values] * df.t.values
                           + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def count_classified(idata):
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)
    cls = (lo > 0) | (hi < 0)
    pdir = np.maximum((b > 0).mean(axis=1), (b < 0).mean(axis=1))
    return int(cls.sum()), cls, pdir


def boundary_population(pdir, band=0.01):
    """Regions whose probability of direction sits within `band` of the 0.975 threshold."""
    return int(((pdir > 0.975 - band) & (pdir < 0.975 + band)).sum())


def main():
    df, regions, states, sor = load()

    counts, per_region = [], []
    for seed in SEEDS:
        cached = OUT / f"idata_seed_{seed}.nc"
        if cached.exists():
            idata = az.from_netcdf(cached)
        else:
            with build(df, len(regions), len(states), sor):
                idata = pm.sample(
                    draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                    target_accept=TARGET_ACCEPT, progressbar=False,
                )
            idata.to_netcdf(cached, engine="h5netcdf")

        n, cls, pdir = count_classified(idata)
        counts.append(n)
        per_region.append(cls)
        print(f"seed {seed}: {n} classified")

    counts = np.array(counts)
    flags = np.vstack(per_region)  # seeds x regions
    always = int((flags.all(axis=0)).sum())
    never = int((~flags.any(axis=0)).sum())
    sometimes = int(len(regions) - always - never)

    # Boundary population from the canonical fit.
    canonical = az.from_netcdf(OUT / "idata_rate.nc")
    _, _, pdir_canon = count_classified(canonical)

    summary = {
        "seeds": list(SEEDS),
        "counts": counts.tolist(),
        "mean": round(float(counts.mean()), 1),
        "sd": round(float(counts.std(ddof=1)), 2),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "range_pct_of_510": round(100 * float(np.ptp(counts)) / len(regions), 2),
        "regions_classified_in_every_run": always,
        "regions_classified_in_no_run": never,
        "regions_that_flip_between_runs": sometimes,
        "boundary_population_canonical_band_0.01": boundary_population(pdir_canon, 0.01),
        "boundary_population_canonical_band_0.02": boundary_population(pdir_canon, 0.02),
    }
    (RES / "SEED_STABILITY.json").write_text(json.dumps(summary, indent=2))

    text = "\n".join(
        [
            "Stability of the classification count across seeds",
            "=" * 60,
            json.dumps(summary, indent=2),
            "",
            f"Interpretation: {always} regions are classified under every seed and "
            f"{never} under none; only {sometimes} change status between runs, and "
            f"{summary['boundary_population_canonical_band_0.01']} regions sit within "
            "0.01 of the decision threshold in the canonical fit. The instability is "
            "concentrated at the boundary and is a property of the binary rule, not of "
            "the estimate.",
        ]
    )
    (OUT / "12_seed_stability.log").write_text(text + "\n")
    print()
    print(text)


if __name__ == "__main__":
    main()
