"""Scaled BYM2 slope prior with the state slope term retained (repair of 04_spatial_robustness).

The earlier "bym2" branch of 04_spatial_robustness.py used pm.ICAR(sigma=1) on binary adjacency
without graph scaling and dropped the state-level slope component that the exchangeable model
keeps. Two things changed at once, so the 42 versus 62 comparison could not be attributed to
spatial borrowing, and rho was not a variance share.

Here, with everything else identical to the exchangeable (primary) model:
  * intercept hierarchy unchanged (national + state + region);
  * slope = mu_b + state component (kept) + sigma_b * ( sqrt(rho) * phi_scaled + sqrt(1-rho) * theta );
  * phi_scaled = ICAR / sqrt(s), s = geometric mean of the marginal variances of the ICAR
    generalised inverse (Riebler et al. 2016), so that sigma_b and rho have their intended meaning;
  * the graph is a single connected component (six queen components welded by five added edges in 02_build_adjacency.py).
Priors as in the exchangeable model (HalfNormal(0.1) on scales); rho ~ Beta(1, 1).

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 2-model/28_scaled_bym2.py
Writes idata_spatial_bym2_scaled.nc and slopes_spatial_bym2_scaled.csv in 2-model/allcause.
"""
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, model_dir  # noqa: E402

assert OUTCOME == "deaths_total"
PROC = ROOT / "data" / "processed"
OUT = model_dir(ROOT)
import os
SEED, TARGET_ACCEPT = 20260803, 0.99
DRAWS = int(os.environ.get("NNM2_DRAWS", 2000))
TUNE = int(os.environ.get("NNM2_TUNE", 3000))
CHAINS = int(os.environ.get("NNM2_CHAINS", 4))
TAG = os.environ.get("NNM2_TAG", "")


def main():
    df = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"]).reset_index(drop=True)
    df["t"] = df["year"] - df["year"].mean()
    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})
    rs = df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    state_of_region = rs.map({s: i for i, s in enumerate(states)}).values
    n_regions, n_states = len(regions), len(states)

    pairs = pd.read_csv(PROC / "adjacency_pairs.csv")
    W = np.zeros((n_regions, n_regions))
    W[pairs.i.values, pairs.j.values] = 1.0
    W[pairs.j.values, pairs.i.values] = 1.0
    ncomp, _ = cg.connected_components(sp.csr_matrix(W))
    assert ncomp == 1, f"graph has {ncomp} components"
    Q = np.diag(W.sum(1)) - W
    scale = float(np.exp(np.mean(np.log(np.diag(np.linalg.pinv(Q))))))
    print(f"ICAR scaling factor s = {scale:.4f}")

    with pm.Model() as model:
        t, r_idx = df.t.values, df.region_idx.values
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sigma_a_state = pm.HalfNormal("sigma_a_state", 0.5)
        sigma_a_region = pm.HalfNormal("sigma_a_region", 0.5)
        z_a_state = pm.Normal("z_a_state", 0.0, 1.0, shape=n_states)
        z_a_region = pm.Normal("z_a_region", 0.0, 1.0, shape=n_regions)
        a = pm.Deterministic("a", mu_a + (sigma_a_state * z_a_state)[state_of_region] + sigma_a_region * z_a_region)

        sigma_b_state = pm.HalfNormal("sigma_b_state", 0.1)
        z_b_state = pm.Normal("z_b_state", 0.0, 1.0, shape=n_states)
        sigma_b = pm.HalfNormal("sigma_b", 0.1)
        rho = pm.Beta("rho", 1.0, 1.0)
        phi = pm.ICAR("phi", W=W, sigma=1.0)
        theta = pm.Normal("theta", 0.0, 1.0, shape=n_regions)
        u = sigma_b * (pm.math.sqrt(rho) * phi / np.sqrt(scale) + pm.math.sqrt(1.0 - rho) * theta)
        b = pm.Deterministic("b", mu_b + (sigma_b_state * z_b_state)[state_of_region] + u)

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial("y", mu=pm.math.exp(a[r_idx] + b[r_idx] * t + np.log(df.births.values)),
                            alpha=alpha, observed=df["deaths_total"].values)
        t0 = time.time()
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                          target_accept=TARGET_ACCEPT, progressbar=False)
        print(f"sampling seconds: {time.time() - t0:.0f}")
        pm.compute_log_likelihood(idata, progressbar=False)
    idata.to_netcdf(OUT / f"idata_spatial_bym2_scaled{TAG}.nc", engine="h5netcdf")

    b_s = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(b_s, 0.025, axis=1), np.quantile(b_s, 0.975, axis=1)
    classified = (lo > 0) | (hi < 0)
    pd.DataFrame({"rgi_id": regions, "b_mean": b_s.mean(1), "b_sd": b_s.std(1), "b_lo95": lo,
                  "b_hi95": hi, "classified_95": classified}).to_csv(OUT / f"slopes_spatial_bym2_scaled{TAG}.csv", index=False)
    summ = az.summary(idata, var_names=["mu_b", "sigma_b", "sigma_b_state", "rho", "alpha"])
    print(summ.to_string())
    print("classified:", int(classified.sum()), " declines:", int((hi < 0).sum()), " increases:", int((lo > 0).sum()))
    rh = az.rhat(idata, var_names=["b", "sigma_b", "rho", "sigma_b_state", "mu_b"])
    es = az.ess(idata, var_names=["b", "sigma_b", "rho", "sigma_b_state", "mu_b"])
    print("divergences:", int(idata.sample_stats.diverging.sum()))
    for v in ("b", "sigma_b", "rho", "sigma_b_state", "mu_b"):
        print(f"{v}: max r-hat {float(np.nanmax(rh[v].values)):.4f}  min ESS bulk {float(np.nanmin(es[v].values)):.0f}")
    loo = az.loo(idata)
    print("LOO elpd:", float(loo.elpd), " p_loo:", float(loo.p), " pareto k > 0.7:", int((loo.pareto_k.values > 0.7).sum()))


if __name__ == "__main__":
    main()
