"""How fragile is the 2,000-death threshold? Sensitivity to dispersion and to temporal dependence.

The threshold of about 2,000 neonatal deaths per unit-year for 80% power against an 8% decade
decline (23_power_threshold.py) rests on the fitted exposure-dependent dispersion and on
independent annual counts. Here the same unpooled procedure is re-run with
  * dispersion: the negative binomial size parameter halved (more overdispersion) and doubled;
  * temporal dependence: a stationary AR(1) latent term on the log mean (rho = 0.5, SD 0.03 and
    0.06) that the quasi-Poisson fit, which assumes independent years, does not model.
Each variant reports the exposure at which power reaches 80% (log-linear interpolation of the
binned curve) and the false-declaration rate under the null. Analytic thresholds are given for the
dispersion variants (no simulation).

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/31_threshold_sensitivity.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, results_dir  # noqa: E402

assert OUTCOME == "deaths_total"
spec = importlib.util.spec_from_file_location("p23", ROOT / "3-results" / "23_power_threshold.py")
p23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p23)
OUT = results_dir(ROOT)
RNG = np.random.default_rng(20261017)
SIMS = 30
SCALES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

s = p23.setup()
panel = s["panel"]
n_regions = len(s["regions"])
t = np.sort(panel.t.unique())
births = panel.pivot(index="region_idx", columns="t", values="births").values
pooled = s["pooled"]
deaths_per_year = pooled / p23.N_YEARS
b8 = s["mu_b"]  # the 8% per decade design scenario (hierarchical mean slope)
kw = dict(c0=s["c0"], c1=s["c1"], centre=s["log_exp_centre"], national_rate=s["national_rate"])

VARIANTS = [
    ("baseline", 1.0, 0.0),
    ("dispersion size x0.5 (more overdispersion)", 0.5, 0.0),
    ("dispersion size x2 (less overdispersion)", 2.0, 0.0),
    ("AR(1) latent SD 0.03, rho 0.5", 1.0, 0.03),
    ("AR(1) latent SD 0.06, rho 0.5", 1.0, 0.06),
]


def latent(shape, sd, rho=0.5):
    if sd == 0:
        return np.zeros(shape)
    eps = np.zeros(shape)
    eps[..., 0] = RNG.normal(0, sd, shape[:-1])
    for j in range(1, shape[-1]):
        eps[..., j] = rho * eps[..., j - 1] + np.sqrt(1 - rho ** 2) * RNG.normal(0, sd, shape[:-1])
    return eps


rows, summary = [], []
for name, amult, ar_sd in VARIANTS:
    curves = {"b8": [], "null": []}
    for k in SCALES:
        B = births * k
        logB = np.log(B)
        alpha_k = (s["alpha"] * k ** s["c1"] * amult)[:, None]
        for eff, b_true in (("b8", b8), ("null", 0.0)):
            mu = B * np.exp(s["level"][:, None] + b_true * t[None, :])
            mu = mu[None, :, :] * np.exp(latent((SIMS, n_regions, p23.N_YEARS), ar_sd))
            Y = RNG.negative_binomial(alpha_k[None], alpha_k[None] / (alpha_k[None] + mu)).reshape(-1, p23.N_YEARS)
            O = np.tile(logB, (SIMS, 1))
            bh, se, valid = p23.fit_quasipoisson(Y, O, t)
            decl = valid & (np.abs(bh) > p23.T_CRIT * se)
            region = np.tile(np.arange(n_regions), SIMS)
            d = pd.DataFrame({"x": k * deaths_per_year[region], "declared": decl, "valid": valid})
            curves[eff].append(d[d.valid])
    thr, fp = {}, {}
    for eff in ("b8", "null"):
        allc = pd.concat(curves[eff], ignore_index=True)
        allc["bin"] = pd.cut(allc.x, p23.BIN_EDGES)
        c = (allc.groupby("bin", observed=True)
             .apply(lambda g: pd.Series({"deaths_per_unit_year": float(np.exp(np.log(g.x).mean())),
                                         "power": float(g.declared.mean()), "n": len(g)}), include_groups=False)
             .reset_index(drop=True))
        if eff == "b8":
            thr = {"power50": p23.interpolate_threshold(c, 0.5), "power80": p23.interpolate_threshold(c, 0.8),
                   "max_power": float(c.power.max()), "max_power_at": float(c.loc[c.power.idxmax(), "deaths_per_unit_year"])}
        else:
            fp = float(allc.declared.mean())
    an = None
    if ar_sd == 0:
        k2 = dict(kw); k2["c0"] = kw["c0"] + np.log(amult)
        an = {"power50": p23.analytic_threshold(0.5, b8, **k2), "power80": p23.analytic_threshold(0.8, b8, **k2)}
    summary.append({"variant": name, "simulated_deaths_per_unit_year_50pct": thr["power50"],
                    "simulated_deaths_per_unit_year_80pct": thr["power80"], "false_declaration_rate_null": round(fp, 4),
                    "max_power_reached": round(thr["max_power"], 3), "max_power_at_deaths_per_unit_year": round(thr["max_power_at"], 0),
                    "analytic_50pct": an and round(an["power50"], 1), "analytic_80pct": an and round(an["power80"], 1)})
    print(summary[-1], flush=True)

pd.DataFrame(summary).to_csv(OUT / "tables" / "threshold_sensitivity.csv", index=False)
json.dump(summary, open(OUT / "THRESHOLD_SENSITIVITY.json", "w"), indent=2, default=float)
