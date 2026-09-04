"""Confidence intervals for the unpooled state contrasts in state_dispersion.csv.

`2-model/15_state_dispersion.py` writes, for each state, the unpooled quasi-Poisson slope
of the state's own series (`slope_own`), the slope of the rest of the country
(`slope_rest`), their difference (`contrast`), the t statistic and its p-value on n-2 = 9
residual degrees of freedom. The supplement previously printed t and p. The journal asks
for a difference with a 95% confidence interval instead, so this script adds the interval
implied by the same t: se = contrast / t, and the bounds are contrast ± t_{0.975, 9} · se.
Nothing is re-estimated; the columns are a re-expression of what is already there.

Adds `contrast_se`, `contrast_lo`, `contrast_hi` to `3-results/tables/state_dispersion.csv`.

Run:
    .venv/bin/python 3-results/20_state_contrast_ci.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "3-results" / "tables" / "state_dispersion.csv"
DF = 9  # 11 annual points, two parameters


def main() -> None:
    sd = pd.read_csv(PATH)
    tcrit = stats.t.ppf(0.975, DF)
    sd["contrast_se"] = sd["contrast"] / sd["t"]
    sd["contrast_lo"] = sd["contrast"] - tcrit * sd["contrast_se"]
    sd["contrast_hi"] = sd["contrast"] + tcrit * sd["contrast_se"]
    # The interval excludes zero exactly when |t| exceeds the critical value, which is the
    # existing `departs_unpooled` flag; assert the two agree before writing.
    excl = (sd["contrast_lo"] > 0) | (sd["contrast_hi"] < 0)
    assert (excl == sd["departs_unpooled"].astype(bool)).all(), "interval and flag disagree"
    sd.to_csv(PATH, index=False)
    rj = sd.set_index("UF").loc["RJ"]
    print(f"t crit (df={DF}) = {tcrit:.4f}")
    print(f"RJ contrast {rj.contrast:.4f} (95% CI {rj.contrast_lo:.4f}, {rj.contrast_hi:.4f})")
    print(f"states whose interval excludes zero: {int(excl.sum())}")


if __name__ == "__main__":
    main()
