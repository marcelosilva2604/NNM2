"""Summarise the matched design analysis (29_matched_arms.py).

For each scenario, procedure and exposure quintile (pooled over all replicates and regions):
  declared_rate  proportion of regions declared (power when the truth is non-null, false-declaration
                 rate under null0);
  wrong_sign     proportion of declarations with the wrong sign (type S error);
  type_M         MEDIAN over all declarations of |declared slope| / |true slope| (type M error);
  power_decline_ge_8pct   declared proportion among regions whose true slope is a decline of at least 8% per decade.
Monte Carlo standard errors are bootstrap standard errors that resample whole replicates (200 resamples),
so the replicate (one simulated dataset) is the unit. Paired differences HIER minus QP and HIER minus NB in the
declaration rate are computed within replicate.

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/30_matched_arms_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import results_dir  # noqa: E402

RES = results_dir(ROOT)
files = sorted((RES / "matched").glob("*_[0-9][0-9][0-9].csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
DECLINE8 = np.log(0.92) / 10  # about -0.00834 per year
RNG = np.random.default_rng(20261017)
B = 200


def pooled(parts):
    """parts: list per replicate of dicts with n, k, wrong, ratios, n8, k8."""
    n = sum(p["n"] for p in parts); k = sum(p["k"] for p in parts)
    wrong = sum(p["wrong"] for p in parts); n8 = sum(p["n8"] for p in parts); k8 = sum(p["k8"] for p in parts)
    ratios = np.concatenate([p["ratios"] for p in parts]) if parts else np.array([])
    return (k / n if n else np.nan, wrong / k if k else np.nan, float(np.median(ratios)) if len(ratios) else np.nan, k8 / n8 if n8 else np.nan)


rows = []
for (sc, pr, q), g in df.groupby(["scenario", "procedure", "quintile"]):
    parts = []
    for rep, h in g.groupby("rep"):
        d = h[h.declared]
        nz = (h.true_b != 0).any()
        low = h.true_b <= DECLINE8
        parts.append({"n": len(h), "k": int(h.declared.sum()),
                      "wrong": int((np.sign(d.b_hat) != np.sign(d.true_b)).sum()) if nz else 0,
                      "ratios": (d.b_hat.abs() / d.true_b.abs()).values if nz else np.array([]),
                      "n8": int(low.sum()), "k8": int((h.declared & low).sum())})
    est = pooled(parts)
    boots = np.array([pooled([parts[i] for i in RNG.integers(0, len(parts), len(parts))]) for _ in range(B)])
    se = np.nanstd(boots, axis=0, ddof=1)
    ncontrib = sum(1 for p in parts if p["k"] > 0)
    nzs = sc != "null0" and ncontrib >= 10  # error rates and their bootstrap errors are withheld when fewer than 10 replicates contribute declarations
    rows.append({"scenario": sc, "procedure": pr, "quintile": q, "replicates": len(parts), "replicates_with_declarations": ncontrib,
                 "declared_rate": round(est[0], 6), "declared_rate_mcse": round(se[0], 6),
                 "wrong_sign": round(est[1], 6) if nzs else np.nan, "wrong_sign_mcse": round(se[1], 4) if nzs else np.nan,
                 "type_M": round(est[2], 2) if nzs else np.nan, "type_M_mcse": round(se[2], 2) if nzs else np.nan,
                 "power_decline_ge_8pct": round(est[3], 6) if nzs else np.nan, "power_decline_ge_8pct_mcse": round(se[3], 4) if nzs else np.nan})
by_q = pd.DataFrame(rows)
by_q.to_csv(RES / "tables" / "matched_arms_by_quintile.csv", index=False)

overall = (df.groupby(["scenario", "procedure", "rep"]).agg(declared=("declared", "sum")).reset_index()
             .groupby(["scenario", "procedure"]).declared.agg(["mean", "std", "min", "max", "count"]).round(2))
overall.to_csv(RES / "tables" / "matched_arms_counts_declared.csv")
wide = df.groupby(["scenario", "procedure", "rep"]).declared.mean().unstack("procedure")
pairs = []
for sc, w in wide.groupby(level=0):
    for a, b in (("HIER", "QP"), ("HIER", "NB")):
        d = (w[a] - w[b]).values
        pairs.append({"scenario": sc, "comparison": f"{a} minus {b}", "mean_difference_in_declared_rate": round(float(d.mean()), 4),
                      "sd_across_replicates": round(float(d.std(ddof=1)), 4), "replicates": len(d)})
pd.DataFrame(pairs).to_csv(RES / "tables" / "matched_arms_paired_differences.csv", index=False)
print("replicates per scenario:", df.drop_duplicates(["scenario", "rep"]).groupby("scenario").size().to_dict())
print("hierarchical divergences:", df.drop_duplicates(["scenario", "rep"]).groupby("scenario").divergences.sum().to_dict())
print(by_q[by_q.scenario == "hetagg"][["procedure", "quintile", "declared_rate", "wrong_sign", "type_M", "type_M_mcse"]].to_string(index=False))
