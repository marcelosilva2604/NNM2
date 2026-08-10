"""Compute the two quantities the reframe added to the manuscript.

Both were derived ad hoc while rewriting and typed straight into the text, which is the
same failure 17_orphan_numbers.py exists to correct. They are computed here instead, and
asserted in the notebooks.

    threshold_rule       power against a change the size of the national drift, expressed
                         against median avoidable deaths per unit-year rather than against
                         Brazilian model parameters, so a custodian elsewhere can locate
                         their own units on it without refitting anything
    completeness_bound   how fast death capture would have to improve to account for the
                         observed national decline

The quintiles are the same ones the design analysis used: five equal-count bins of total
avoidable deaths per region over the decade, built by pd.qcut on the same pooled counts in
the same region order. Reproducing that construction here rather than importing it keeps
this script readable, but it means the two must not drift apart; the notebook checks the
power column against the committed design table to catch that.

Run:
    .venv/bin/python 3-results/18_transferability.py
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

N_YEARS = 11  # 2014-2024 inclusive


def threshold_rule():
    """Power against exposure, in events rather than in Brazilian parameters.

    Power is read from the committed arm A table at a true slope of one times the national
    drift; only the exposure axis is added here. Exposure is reported as avoidable deaths
    per unit-year, the number a registry custodian can count for their own units without
    fitting a model.
    """
    panel = pd.read_csv(PROC / "panel_region_year.csv")
    regions = np.sort(panel.rgi_id.unique())
    pooled = panel.groupby("rgi_id").avoidable.sum().loc[regions].values
    quint = pd.qcut(pooled, 5, labels=False) + 1

    per_year = pd.DataFrame({"quintile": quint, "deaths_per_year": pooled / N_YEARS})
    exposure = per_year.groupby("quintile").deaths_per_year.median().round(1)

    design = pd.read_csv(OUT / "tables" / "design_two_arm.csv")
    arm_a = design[design.arm.str.startswith("A") & (design.true_slope_x_national == 1.0)]
    arm_a = arm_a.set_index("exposure_quintile").sort_index()

    table = pd.DataFrame(
        {
            "exposure_quintile": exposure.index,
            "median_avoidable_deaths_per_unit_year": exposure.values,
            "power_vs_national_drift": arm_a.power.values,
            "type_S": arm_a.type_S.values,
            "type_M": arm_a.type_M.values,
        }
    )
    table.to_csv(OUT / "tables" / "threshold_rule.csv", index=False)

    # The relation is steep and we observe only up to the top quintile, so the manuscript
    # states the range and explicitly declines to extrapolate. Recording the endpoint here
    # keeps that refusal checkable rather than rhetorical.
    return {
        "quintiles": table.to_dict(orient="records"),
        "lowest": {"deaths_per_unit_year": float(exposure.iloc[0]),
                   "power": float(arm_a.power.iloc[0])},
        "median": {"deaths_per_unit_year": float(exposure.iloc[2]),
                   "power": float(arm_a.power.iloc[2])},
        "highest": {"deaths_per_unit_year": float(exposure.iloc[4]),
                    "power": float(arm_a.power.iloc[4])},
        "max_observed_exposure": float(exposure.iloc[4]),
    }


def completeness_bound():
    """How fast would death capture have to improve to explain the national decline?

    If a unit ascertains a fraction c_t of its true deaths and c_t grows geometrically at
    rate r per year, then

        log E[observed deaths] = log c_0 + t*log(1 + r) + log E[true deaths]

    so the fitted slope is displaced by exactly +log(1 + r) per year, whatever the true
    slope is. Two consequences matter for the paper and are recorded here.

    First, the displacement is additive in the mean and does not depend on the data, so it
    moves where a slope sits without touching how precisely it is estimated. The
    detectability result is a statement about interval width and therefore survives any
    amount of capture drift; the direction of the national decline does not.

    Second, the rate that would erase the observed drift entirely is r = exp(|mu_b|) - 1,
    not |mu_b| itself. The linear reading understates it slightly and is not used.
    """
    idata = az.from_netcdf(MODEL / "idata_rate.nc")
    mu_b = float(idata.posterior["mu_b"].mean())

    displacement_at_1pct = float(np.log(1.01))
    erasing_rate = float(np.expm1(abs(mu_b)))

    return {
        "national_drift_mu_b": round(mu_b, 5),
        "displacement_if_capture_improves_1pct_per_year": round(displacement_at_1pct, 5),
        "displacement_as_share_of_national_drift": round(
            displacement_at_1pct / abs(mu_b), 3
        ),
        "capture_growth_that_would_erase_the_drift_pct_per_year": round(
            100 * erasing_rate, 2
        ),
        "affects_precision": False,
    }


def main():
    result = {
        "threshold_rule": threshold_rule(),
        "completeness_bound": completeness_bound(),
    }
    (OUT / "TRANSFERABILITY.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
