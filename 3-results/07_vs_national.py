"""Separate lagging behind the country from being unable to tell.

Classifying a unit against zero answers "did anything change here". It does not answer
"did this place improve less than the rest of the country", which is the question a
programme manager actually asks. That requires the contrast between the unit slope and
the national drift, b_i - mu_b, evaluated inside the posterior so that the correlation
between the two is carried through rather than ignored.

Crossing the two comparisons gives every unit one of six states, and in particular
separates a unit that is credibly lagging from a unit about which nothing can be said.

Run:
    .venv/bin/python 3-results/07_vs_national.py
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

LEVELS = {
    "state": ("idata_state.nc", "slopes_state.csv", "UF"),
    "region": ("idata_rate.nc", "slopes_rate.csv", "rgi_id"),
}


def classify(level):
    """Cross the against-zero and against-national comparisons for one level."""
    nc, csv, key = LEVELS[level]
    idata = az.from_netcdf(MODEL / nc)

    b = idata.posterior["b"].stack(sample=("chain", "draw")).values  # (unit, sample)
    mu = idata.posterior["mu_b"].stack(sample=("chain", "draw")).values  # (sample,)

    # The contrast is taken draw by draw, so the posterior correlation between a unit
    # slope and the national drift is respected.
    delta = b - mu[None, :]

    units = pd.read_csv(MODEL / csv)[key].values
    d_lo, d_hi = np.quantile(delta, 0.025, axis=1), np.quantile(delta, 0.975, axis=1)
    b_lo, b_hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)

    changed = (b_lo > 0) | (b_hi < 0)
    # delta > 0 means the unit's slope is above the national one, i.e. falling more
    # slowly (or rising), which is lagging.
    lagging = d_lo > 0
    leading = d_hi < 0

    label = np.where(
        lagging,
        "Lagging the country",
        np.where(
            leading,
            "Ahead of the country",
            np.where(changed, "Changed, in line with the country", "No determination"),
        ),
    )

    out = pd.DataFrame(
        {
            key: units,
            "b_mean": b.mean(axis=1),
            "b_lo95": b_lo,
            "b_hi95": b_hi,
            "changed_vs_zero": changed,
            "delta_mean": delta.mean(axis=1),
            "delta_lo95": d_lo,
            "delta_hi95": d_hi,
            "lagging": lagging,
            "leading": leading,
            "label": label,
        }
    )
    out.to_csv(OUT / "tables" / f"vs_national_{level}.csv", index=False)
    return out, key


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    log = ["Unit slope against the national drift", "=" * 60]

    for level in ("state", "region"):
        out, key = classify(level)
        n = len(out)
        log.append("")
        log.append(f"=== {level.upper()} (n = {n}) ===")
        counts = out.label.value_counts()
        for lab in [
            "Ahead of the country",
            "Changed, in line with the country",
            "Lagging the country",
            "No determination",
        ]:
            c = int(counts.get(lab, 0))
            log.append(f"  {lab:<36} {c:>4}  ({100 * c / n:.1f}%)")

        log.append("")
        log.append(
            f"  credibly different from the national drift in either direction: "
            f"{int((out.lagging | out.leading).sum())} of {n}"
        )

        if level == "state":
            log.append("")
            log.append("  all 27 states, ordered by slope (least to most improvement):")
            show = out.sort_values("b_mean", ascending=False)[
                [key, "b_mean", "b_lo95", "b_hi95", "delta_mean", "delta_lo95",
                 "delta_hi95", "label"]
            ]
            log.append(show.round(4).to_string(index=False))
        else:
            # Among the units that cannot be separated from zero, how many can still be
            # separated from the national drift?
            und = out[~out.changed_vs_zero]
            log.append("")
            log.append(
                f"  of the {len(und)} regions not separable from zero, "
                f"{int((und.lagging | und.leading).sum())} are still separable from the "
                f"national drift"
            )

    text = "\n".join(log)
    (OUT / "07_vs_national.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
