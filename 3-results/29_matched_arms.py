"""Matched design analysis: the SAME simulated datasets fitted by three procedures.

The earlier two-arm analysis (16_design_two_arm.py) compared an unpooled quasi-Poisson fit on a
common true slope (arm A) with the hierarchical fit on slopes drawn from the fitted
heterogeneity (arm B). The truths, the dispersion information and the decision rules differed, so
the statement that pooling reduces sign and magnitude errors "at the cost of power" was not
supported. Here, in each replicate and scenario, one dataset is simulated and fitted by:

    QP   unpooled quasi-Poisson per region, t criterion on n-2 df (arm A procedure);
    NB   unpooled negative binomial per region with the region's dispersion KNOWN (the same
         dispersion information the hierarchical fit receives), Wald z criterion;
    HIER the paper's three-level hierarchical model (arm B procedure), 95% interval excludes 0.

Scenarios (true slopes):
    null0    every region's true slope is 0 (false declarations);
    hetnull  true slopes ~ N(0, tau): heterogeneity without a mean trend (sign errors);
    hetagg   true slopes ~ N(m_agg, tau), m_agg = -0.00717 = slope of the aggregate national rate
             (audit 26), tau the fitted between-region SD as in arm B.

Records per region, procedure and replicate go to 3-results/allcause/matched/<scenario>_<rep>.csv.
Aggregation (power, type S, type M, false declarations by exposure quintile, with Monte Carlo
SE across replicates) is done by 30_matched_arms_summary.py.

Run one block of replicates:
    NNM2_OUTCOME=deaths_total NNM2_SCENARIO=hetagg NNM2_REP_START=0 NNM2_REP_END=10 \
        .venv/bin/python 3-results/29_matched_arms.py
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, results_dir  # noqa: E402

assert OUTCOME == "deaths_total"
spec = importlib.util.spec_from_file_location("two_arm", ROOT / "3-results" / "16_design_two_arm.py")
two = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two)

OUTDIR = results_dir(ROOT) / "matched"
OUTDIR.mkdir(exist_ok=True)
SCENARIO = os.environ["NNM2_SCENARIO"]
REP0, REP1 = int(os.environ["NNM2_REP_START"]), int(os.environ["NNM2_REP_END"])
M_AGG = -0.00717
BASE_SEED = {"null0": 710000, "hetnull": 720000, "hetagg": 730000}[SCENARIO]


def fit_nb_known(panel, y, alpha):
    """Unpooled NB per region with known dispersion; slope, se, declared (|z| > 1.96)."""
    n = panel.region_idx.max() + 1
    out = np.full((n, 3), np.nan)
    frame = panel.assign(y=y)
    for idx, g in frame.groupby("region_idx"):
        if g.y.sum() == 0:
            continue
        try:
            f = sm.GLM(g.y, sm.add_constant(g[["t"]]),
                       family=sm.families.NegativeBinomial(alpha=1.0 / alpha[idx]),
                       offset=np.log(g.births)).fit()
            b, se = float(f.params["t"]), float(f.bse["t"])
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            out[idx] = (b, se, float(abs(b) > 1.959964 * se))
    return out


def main():
    panel, regions, states, sor, mu_b, tau, alpha, level = two.setup()
    n_regions = len(regions)
    pooled = panel.groupby("rgi_id")[OUTCOME].sum().loc[regions].values
    quint = pd.qcut(pooled, 5, labels=False) + 1
    # dispersion convention of the original arms: alpha is the NB size parameter; statsmodels
    # NegativeBinomial uses variance mu + a*mu^2 with a = 1/size.
    for rep in range(REP0, REP1):
        f = OUTDIR / f"{SCENARIO}_{rep:03d}.csv"
        if f.exists():
            continue
        rng = np.random.default_rng(BASE_SEED + rep)
        two.RNG = rng  # simulate() draws from the module-level generator
        if SCENARIO == "null0":
            true_b = np.zeros(n_regions)
        elif SCENARIO == "hetnull":
            true_b = rng.normal(0.0, tau, n_regions)
        else:
            true_b = M_AGG + rng.normal(0.0, tau, n_regions)
        y = two.simulate(panel, level, alpha, true_b)
        t0 = time.time()
        qp = two.fit_unpooled(panel, y)
        nb = fit_nb_known(panel, y, alpha)
        bhat, decl, div = two.fit_hierarchical(panel, y, n_regions, len(states), sor, alpha, BASE_SEED + rep)
        rows = []
        for i in range(n_regions):
            for proc, bh, dc in (("QP", qp[i, 0], qp[i, 2]), ("NB", nb[i, 0], nb[i, 2]), ("HIER", bhat[i], float(decl[i]))):
                if not np.isfinite(bh):
                    continue
                rows.append((SCENARIO, rep, regions[i], int(quint[i]), proc, true_b[i], float(bh), bool(dc), div))
        pd.DataFrame(rows, columns=["scenario", "rep", "rgi_id", "quintile", "procedure", "true_b",
                                    "b_hat", "declared", "divergences"]).to_csv(f, index=False)
        print(f"{SCENARIO} rep {rep}: {time.time() - t0:.0f}s, divergences {div}", flush=True)


if __name__ == "__main__":
    main()
