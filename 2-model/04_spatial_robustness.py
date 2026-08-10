"""Does the indeterminacy survive borrowing strength from neighbours?

This is the objection the paper must answer before submission: the small-area estimation
literature exists to borrow strength, and an exchangeable prior borrows only from the
national mean. If a spatially structured prior resolves most regions, the paper's central
claim is an artefact of the model class rather than a property of the data.

Three slope priors on the same negative binomial rate likelihood:

    nopool  independent slopes, weak prior. No strength borrowed at all.
    exch    exchangeable (the canonical fit's prior), refitted here so the three numbers
            come from one script under identical settings.
    bym2    BYM2: a mixture of a spatially structured ICAR component and an unstructured
            component, with rho the share of slope variation that is spatial.

The comparison the manuscript reports is the classification rate across the three, plus
rho, which measures how much of the map a spatial prior would be writing.

Run:
    .venv/bin/python 2-model/04_spatial_robustness.py
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

HDI_PROB = 0.95
SEED = 20260803
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    """Panel, region ordering, state index and the adjacency edge list."""
    df = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    df = df.reset_index(drop=True)
    df["t"] = df["year"] - df["year"].mean()

    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})

    region_state = (
        df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    )
    states = np.sort(region_state.unique())
    state_of_region = region_state.map(
        {s: i for i, s in enumerate(states)}
    ).values

    pairs = pd.read_csv(PROC / "adjacency_pairs.csv")
    # ICAR takes the symmetric binary adjacency matrix.
    n = len(regions)
    W = np.zeros((n, n))
    W[pairs.i.values, pairs.j.values] = 1.0
    W[pairs.j.values, pairs.i.values] = 1.0
    return df, regions, states, state_of_region, W


def build(df, n_regions, n_states, state_of_region, W, prior):
    """Same likelihood throughout; only the slope prior changes."""
    with pm.Model() as model:
        t = df.t.values
        r_idx = df.region_idx.values

        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)

        sigma_a_state = pm.HalfNormal("sigma_a_state", 0.5)
        sigma_a_region = pm.HalfNormal("sigma_a_region", 0.5)
        z_a_state = pm.Normal("z_a_state", 0.0, 1.0, shape=n_states)
        z_a_region = pm.Normal("z_a_region", 0.0, 1.0, shape=n_regions)
        a = pm.Deterministic(
            "a",
            mu_a
            + (sigma_a_state * z_a_state)[state_of_region]
            + sigma_a_region * z_a_region,
        )

        if prior == "nopool":
            # Deliberately no hierarchy on the slope: each region's slope is informed
            # only by its own deaths, under a prior wide enough to be uninformative at
            # this scale.
            # Scale 0.25 is still an order of magnitude wider than the fitted between-
            # region slope spread (~0.02), so nothing is pooled in practice, but it
            # stops chains from wandering in the regions whose likelihood is flat.
            # At scale 1.0 this model did not converge (worst r-hat 1.61).
            b = pm.Deterministic(
                "b", mu_b + pm.Normal("b_raw", 0.0, 0.25, shape=n_regions)
            )

        elif prior == "exch":
            sigma_b_state = pm.HalfNormal("sigma_b_state", 0.1)
            sigma_b_region = pm.HalfNormal("sigma_b_region", 0.1)
            z_b_state = pm.Normal("z_b_state", 0.0, 1.0, shape=n_states)
            z_b_region = pm.Normal("z_b_region", 0.0, 1.0, shape=n_regions)
            b = pm.Deterministic(
                "b",
                mu_b
                + (sigma_b_state * z_b_state)[state_of_region]
                + sigma_b_region * z_b_region,
            )

        elif prior == "bym2":
            sigma_b = pm.HalfNormal("sigma_b", 0.1)
            rho = pm.Beta("rho", 1.0, 1.0)
            phi = pm.ICAR("phi", W=W, sigma=1.0)
            theta = pm.Normal("theta", 0.0, 1.0, shape=n_regions)
            # BYM2 convolution: rho is the share of slope variance that is spatial.
            b = pm.Deterministic(
                "b",
                mu_b
                + sigma_b
                * (pm.math.sqrt(rho) * phi + pm.math.sqrt(1.0 - rho) * theta),
            )
        else:
            raise ValueError(prior)

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[r_idx] + b[r_idx] * t + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def summarise(idata, regions, prior):
    """Classification rate and precision under this prior."""
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo = np.quantile(b, 0.025, axis=1)
    hi = np.quantile(b, 0.975, axis=1)
    sd = b.std(axis=1)
    classified = (lo > 0) | (hi < 0)

    pd.DataFrame(
        {
            "rgi_id": regions,
            "b_mean": b.mean(axis=1),
            "b_sd": sd,
            "b_lo95": lo,
            "b_hi95": hi,
            "classified_95": classified,
        }
    ).to_csv(OUT / f"slopes_spatial_{prior}.csv", index=False)

    return {
        "prior": prior,
        "classified": int(classified.sum()),
        "share": float(classified.mean()),
        "median_sd": float(np.median(sd)),
        "median_mde": float(np.median(1.96 * sd)),
        "divergences": int(idata.sample_stats.diverging.sum()),
    }


def main():
    df, regions, states, state_of_region, W = load()
    rows, log = [], ["Spatial robustness of the classification rate", "=" * 60]

    for prior in ("nopool", "exch", "bym2"):
        cached = OUT / f"idata_spatial_{prior}.nc"
        if cached.exists():
            idata = az.from_netcdf(cached)
            log.append(f"[{prior}] loaded cached posterior")
        else:
            model = build(
                df, len(regions), len(states), state_of_region, W, prior
            )
            with model:
                idata = pm.sample(
                    draws=DRAWS,
                    tune=TUNE,
                    chains=CHAINS,
                    random_seed=SEED,
                    target_accept=TARGET_ACCEPT,
                    progressbar=False,
                )
            try:
                idata.to_netcdf(cached, engine="h5netcdf")
            except Exception as exc:  # noqa: BLE001
                with open(OUT / f"idata_spatial_{prior}.pkl", "wb") as handle:
                    pickle.dump(idata, handle)
                log.append(f"[{prior}] netCDF write failed ({exc}); pickled instead")

        row = summarise(idata, regions, prior)
        rows.append(row)

        checked = ["mu_b", "a", "b"]
        rhat = az.rhat(idata, var_names=checked)
        worst = max(float(rhat[v].max()) for v in rhat.data_vars)
        log.append(
            f"[{prior}] classified {row['classified']}/{len(regions)}"
            f" ({row['share']:.1%})  median SD {row['median_sd']:.4f}"
            f"  divergences {row['divergences']}  worst r-hat {worst:.4f}"
        )

        if prior == "bym2":
            rho = az.summary(idata, var_names=["rho", "sigma_b"], ci_prob=HDI_PROB)
            log.append("")
            log.append("BYM2 spatial share of slope variation (rho):")
            log.append(rho.to_string())

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "3-results" / "tables" / "spatial_robustness.csv", index=False)

    log.append("")
    log.append(table.to_string(index=False))
    text = "\n".join(log)
    (OUT / "04_spatial_robustness.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
