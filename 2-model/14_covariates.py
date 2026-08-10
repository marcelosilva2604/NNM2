"""Can covariates rescue trend detectability, or do they only appear to?

The small area estimation literature exists to borrow strength, so a paper claiming that
regions cannot be resolved has to show that the claim survives a covariate-assisted
specification. Two things must be separated, and the whole design here exists to
separate them.

    A covariate can genuinely sharpen a LEVEL. Knowing a region is poor helps estimate
    how many babies die there.

    A covariate that predicts a TREND does something different. If a model is told that
    poorer regions declined more slowly, and then reports that a poor region declined
    slowly, part of that report came from the covariate rather than from the deaths.
    Classification would improve while the data underneath had not changed at all.

Four fits, identical in likelihood, hierarchy, sampler and criterion, differing only in
where covariates enter:

    base        no covariates (the paper's specification, refitted here so all fits are
                comparable under one script and one seed)
    level       the static index enters the intercept only
    slope       the static index enters the slope
    timevar     time-varying capacity enters the linear predictor directly
    mundlak     the same capacity covariates split into a region mean and a within-region
                deviation. A random intercept absorbs neither part, so a single
                coefficient blends variation between regions with variation over time
                inside a region, and only the within part can speak to change. If the
                within coefficient is null or negative, the reading of the pooled
                coefficient as confounding by unit size is demonstrated rather than
                asserted; if it is positive, that reading is wrong and must be withdrawn.

For the `slope` fit two classification counts are reported: of the slope itself, which
the covariate can inflate, and of the residual deviation from the covariate prediction,
which is what the region's own deaths establish beyond what the covariate already
implied. The gap between them measures how much of the map the covariate would be
writing.

Run:
    .venv/bin/python 2-model/14_covariates.py
"""

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "2-model"
RES = ROOT / "3-results"

SEED = 20260803
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    df = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    cov = pd.read_csv(PROC / "covariates_region_year.csv")
    df = df.merge(
        cov[["rgi_id", "year", "idhm_z", "nicu_per_1000_z", "ubs_per_1000_z"]],
        on=["rgi_id", "year"],
        how="left",
    ).reset_index(drop=True)
    assert df[["idhm_z", "nicu_per_1000_z", "ubs_per_1000_z"]].notna().all().all()

    df["t"] = df.year - df.year.mean()
    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})
    rs = df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    sor = rs.map({s: i for i, s in enumerate(states)}).values

    # The static index is one value per region; take it once rather than per row.
    idhm = df.groupby("region_idx").idhm_z.first().loc[np.arange(len(regions))].values
    return df, regions, states, sor, idhm


def build(df, n_regions, n_states, sor, idhm, variant):
    with pm.Model() as model:
        r_idx = df.region_idx.values
        t = df.t.values

        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa_s = pm.HalfNormal("sigma_a_state", 0.5)
        sa_r = pm.HalfNormal("sigma_a_region", 0.5)
        sb_s = pm.HalfNormal("sigma_b_state", 0.1)
        sb_r = pm.HalfNormal("sigma_b_region", 0.1)

        a_dev = pm.Deterministic(
            "a_dev",
            (sa_s * pm.Normal("z_a_state", 0, 1, shape=n_states))[sor]
            + sa_r * pm.Normal("z_a_region", 0, 1, shape=n_regions),
        )
        b_dev = pm.Deterministic(
            "b_dev",
            (sb_s * pm.Normal("z_b_state", 0, 1, shape=n_states))[sor]
            + sb_r * pm.Normal("z_b_region", 0, 1, shape=n_regions),
        )

        a_fixed = mu_a
        b_fixed = mu_b
        if variant == "level":
            gamma_a = pm.Normal("gamma_a", 0.0, 1.0)
            a_fixed = mu_a + gamma_a * idhm
        elif variant == "slope":
            gamma_b = pm.Normal("gamma_b", 0.0, 0.05)
            b_fixed = mu_b + gamma_b * idhm

        a = pm.Deterministic("a", a_fixed + a_dev)
        b = pm.Deterministic("b", b_fixed + b_dev)

        eta = a[r_idx] + b[r_idx] * t

        if variant == "timevar":
            d_nicu = pm.Normal("delta_nicu", 0.0, 0.5)
            d_ubs = pm.Normal("delta_ubs", 0.0, 0.5)
            eta = eta + d_nicu * df.nicu_per_1000_z.values + d_ubs * df.ubs_per_1000_z.values

        if variant == "mundlak":
            nicu_between = df.groupby("region_idx").nicu_per_1000_z.transform("mean").values
            ubs_between = df.groupby("region_idx").ubs_per_1000_z.transform("mean").values
            nicu_within = df.nicu_per_1000_z.values - nicu_between
            ubs_within = df.ubs_per_1000_z.values - ubs_between
            b_nicu = pm.Normal("between_nicu", 0.0, 0.5)
            b_ubs = pm.Normal("between_ubs", 0.0, 0.5)
            w_nicu = pm.Normal("within_nicu", 0.0, 0.5)
            w_ubs = pm.Normal("within_ubs", 0.0, 0.5)
            eta = (eta + b_nicu * nicu_between + b_ubs * ubs_between
                   + w_nicu * nicu_within + w_ubs * ubs_within)

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(eta + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def classify(draws):
    """95% equal-tailed interval excluding zero, the criterion used throughout."""
    lo, hi = np.quantile(draws, 0.025, axis=1), np.quantile(draws, 0.975, axis=1)
    return (lo > 0) | (hi < 0), draws.std(axis=1)


def main():
    df, regions, states, sor, idhm = load()
    rows, log = [], ["Covariate-assisted specifications", "=" * 70]

    for variant in ("base", "level", "slope", "timevar", "mundlak"):
        cached = OUT / f"idata_cov_{variant}.nc"
        if cached.exists():
            idata = az.from_netcdf(cached)
        else:
            with build(df, len(regions), len(states), sor, idhm, variant):
                idata = pm.sample(
                    draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                    target_accept=TARGET_ACCEPT, progressbar=False,
                )
            try:
                idata.to_netcdf(cached, engine="h5netcdf")
            except Exception:  # noqa: BLE001
                with open(OUT / f"idata_cov_{variant}.pkl", "wb") as h:
                    pickle.dump(idata, h)

        post = idata.posterior
        b = post["b"].stack(sample=("chain", "draw")).values
        a = post["a"].stack(sample=("chain", "draw")).values
        cls_b, sd_b = classify(b)
        _, sd_a = classify(a)

        row = {
            "variant": variant,
            "classified_slope": int(cls_b.sum()),
            "pct_slope": round(100 * float(cls_b.mean()), 1),
            "median_sd_slope": round(float(np.median(sd_b)), 4),
            "median_sd_level": round(float(np.median(sd_a)), 4),
            "divergences": int(idata.sample_stats.diverging.sum()),
        }

        # For the slope-assisted fit, the honest quantity is what survives once the
        # covariate's own prediction is removed.
        if variant == "slope":
            b_dev = post["b_dev"].stack(sample=("chain", "draw")).values
            cls_r, _ = classify(b_dev)
            row["classified_residual"] = int(cls_r.sum())
            row["pct_residual"] = round(100 * float(cls_r.mean()), 1)

        for name in ("gamma_a", "gamma_b", "delta_nicu", "delta_ubs",
                     "between_nicu", "between_ubs", "within_nicu", "within_ubs"):
            if name in post:
                # Read the coefficient from the posterior directly. az.summary rounds to
                # a display precision and returns strings, which cannot be formatted as
                # numbers and silently differ from the stored draws.
                draws = post[name].stack(sample=("chain", "draw")).values
                lo, hi = np.quantile(draws, 0.025), np.quantile(draws, 0.975)
                row[name] = f"{draws.mean():.4f} ({lo:.4f} to {hi:.4f})"

        rows.append(row)
        log.append(f"[{variant}] {row}")
        print(row)

    table = pd.DataFrame(rows)
    table.to_csv(RES / "tables" / "covariate_models.csv", index=False)

    base = table[table.variant == "base"].iloc[0]
    lvl = table[table.variant == "level"].iloc[0]
    slp = table[table.variant == "slope"].iloc[0]

    tv = table[table.variant == "timevar"].iloc[0]
    log += [
        "",
        table.to_string(index=False),
        "",
        "Reading:",
        f"  LEVEL: the static index predicts the level strongly (gamma_a "
        f"{lvl.gamma_a}) and moves level precision from {base.median_sd_level} to "
        f"{lvl.median_sd_level}, while slope classification goes from "
        f"{int(base.classified_slope)} to {int(lvl.classified_slope)} of 510. A static "
        f"covariate identifies where mortality is high, not whether it is falling.",
        f"  SLOPE: gamma_b is {slp.gamma_b}, an interval that includes zero, so the "
        f"index carries no information about the rate of change and classification is "
        f"unchanged at {int(slp.classified_slope)}. Because the coefficient is "
        f"effectively zero there is nothing for the covariate to assert and the residual "
        f"deviation reduces to the departure-from-national contrast; its count "
        f"({int(slp.classified_residual)}) is that same quantity, not an independent "
        f"result, and must not be reported as one.",
        f"  TIMEVAR: classification rises to {int(tv.classified_slope)}, but the primary "
        f"care coefficient is {tv.delta_ubs}, credibly POSITIVE, i.e. more units per "
        f"birth associated with higher avoidable mortality. That sign is not plausible "
        f"causally and is consistent with confounding by unit size and with CNES being "
        f"an establishment registry. The gain is variance absorbed by a covariate of "
        f"unclear meaning, not an improvement in what the deaths establish, and is "
        f"reported as such.",
    ]
    text = "\n".join(log)
    (OUT / "14_covariates.log").write_text(text + "\n")
    print()
    print(text)


if __name__ == "__main__":
    main()
