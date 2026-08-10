"""Turn the canonical fit into the precision statements the paper is built on.

Consumes only the InferenceData written by 2-model/02_canonical_fit.py. Nothing is
refitted here. Four blocks:

    1. Classification reported as precision, not as a test: probability of direction,
       expected sign errors, minimum detectable slope in policy units.
    2. ROPE sensitivity curve, so "83% indeterminate" is reported as a function of the
       tolerance rather than as a property of Brazil.
    3. Benjamini-Hochberg column, so the multiplicity question is answered rather than
       left silent.
    4. Joint classification of the two estimands, since rate-resolved and share-resolved
       are not the same set of regions.

Run:
    .venv/bin/python 3-results/03_precision_report.py
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

# The national change over the decade, used to anchor the tolerance grid in something
# interpretable rather than in an arbitrary width on the logit scale.
YEARS_SPAN = 10


def load(kind):
    """Load one model's posterior and its per-region slope table."""
    idata = az.from_netcdf(MODEL / f"idata_{kind}.nc")
    slopes = pd.read_csv(MODEL / f"slopes_{kind}.csv")
    return idata, slopes


def precision_block(idata, slopes, kind):
    """Report what each unit's own data can and cannot establish."""
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    mu_b = float(idata.posterior["mu_b"].mean())

    p_dir = slopes.prob_direction.values
    mde = slopes.mde.values

    # Expected number of wrong-sign calls if every unit's point estimate is read at face
    # value: one minus the probability of direction, summed.
    expected_sign_errors = float((1 - p_dir).sum())

    rows = []
    for thr in (0.90, 0.95, 0.975):
        sel = p_dir >= thr
        rows.append(
            {
                "threshold": thr,
                "regions": int(sel.sum()),
                "share": float(sel.mean()),
                "expected_sign_errors": float((1 - p_dir[sel]).sum()),
            }
        )
    pd_table = pd.DataFrame(rows)

    # Express the detection floor in the same units a reader thinks in.
    if kind == "share":
        # Percentage points of avoidable share per decade, evaluated at the national level.
        p0 = float(stats.logistic.cdf(idata.posterior["mu_a"].mean()))
        unit = "pp of avoidable share per decade"
        to_units = lambda x: 100 * (
            stats.logistic.cdf(stats.logistic.ppf(p0) + x * YEARS_SPAN) - p0
        )
    else:
        unit = "% change in avoidable deaths per birth per decade"
        to_units = lambda x: 100 * (np.exp(x * YEARS_SPAN) - 1)

    lines = [
        f"[{kind}] national drift mu_b            : {mu_b:+.4f} per year"
        f"  ({to_units(abs(mu_b)):.1f} {unit})",
        f"[{kind}] median minimum detectable      : {np.median(mde):.4f} per year"
        f"  ({to_units(np.median(mde)):.1f} {unit})",
        f"[{kind}] detection floor / national     : {np.median(mde) / abs(mu_b):.2f}x",
        f"[{kind}] naive sign error rate          : {expected_sign_errors / len(p_dir):.1%}"
        f"  ({expected_sign_errors:.0f} of {len(p_dir)} regions)",
        "",
        pd_table.to_string(index=False),
    ]
    return "\n".join(lines), pd_table


def rope_curve(idata, kind, mu_b):
    """How many units are drifting, credibly flat, or indeterminate, as a function of tolerance.

    The tolerance grid is anchored on the national decade change so the reader can see
    where the answer flips, instead of being handed a single hand-picked width.
    """
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo = np.quantile(b, 0.025, axis=1)
    hi = np.quantile(b, 0.975, axis=1)

    rows = []
    for mult in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        r = mult * abs(mu_b)
        # Standard three-way ROPE decision. The comparator for "drifting" is the
        # tolerance, not zero: a slope credibly different from zero but entirely inside
        # the tolerance band is practically null, not a detected change. Testing against
        # zero instead double-counts those units and can drive the categories past 100%.
        drifting = ((lo > r) | (hi < -r)).sum()
        flat = ((lo > -r) & (hi < r)).sum()
        conclusive = drifting + flat  # the two rules are mutually exclusive by construction
        rows.append(
            {
                "tolerance_x_national": mult,
                "rope_halfwidth": r,
                "drifting": int(drifting),
                "credibly_flat": int(flat),
                "conclusive": int(conclusive),
                "indeterminate_share": float(1 - conclusive / len(lo)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / f"rope_curve_{kind}.csv", index=False)
    return table


def fdr_column(slopes, kind):
    """Benjamini-Hochberg on the posterior z, as the supplementary multiplicity answer.

    Hierarchical shrinkage already damps multiplicity, which is the defence the paper
    makes in the text; this column exists so the reader can see the stricter answer too.
    """
    z = slopes.b_mean.values / slopes.b_sd.values
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    thresholds = np.arange(1, n + 1) / n * 0.05
    passing = ranked <= thresholds
    k = np.max(np.where(passing)[0]) + 1 if passing.any() else 0

    bh = np.zeros(n, dtype=bool)
    if k:
        bh[order[:k]] = True
    # Written to the results tree, not back into the model artefact it read. A results
    # script that rewrites its own input makes the pipeline unreproducible in place.
    slopes["bh_fdr05"] = bh
    slopes.to_csv(OUT / "tables" / f"bh_{kind}.csv", index=False)

    return (
        f"[{kind}] classified at 95%   : {int(slopes.classified_95.sum())}\n"
        f"[{kind}] surviving BH-FDR 5% : {int(bh.sum())}\n"
        f"[{kind}] Bonferroni 5%       : {int((p <= 0.05 / n).sum())}"
    )


def joint_table(rate_slopes, share_slopes):
    """Cross-classify the two estimands; they do not select the same regions."""
    merged = rate_slopes[["rgi_id", "classified_95", "b_mean"]].merge(
        share_slopes[["rgi_id", "classified_95", "b_mean"]],
        on="rgi_id",
        suffixes=("_rate", "_share"),
    )
    tab = pd.crosstab(merged.classified_95_rate, merged.classified_95_share)
    tab.to_csv(OUT / "tables" / "joint_classification.csv")

    both = int((merged.classified_95_rate & merged.classified_95_share).sum())
    only_rate = int((merged.classified_95_rate & ~merged.classified_95_share).sum())
    only_share = int((~merged.classified_95_rate & merged.classified_95_share).sum())

    return "\n".join(
        [
            "JOINT CLASSIFICATION (rate vs share)",
            f"  resolved by both        : {both}",
            f"  resolved by rate only   : {only_rate}",
            f"  resolved by share only  : {only_share}",
            f"  agreement among resolved: {both / max(both + only_rate + only_share, 1):.1%}",
            "",
            tab.to_string(),
        ]
    )


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    log = ["Paper 2 precision report", "=" * 60]

    tables = {}
    for kind in ("rate", "share"):
        idata, slopes = load(kind)
        mu_b = float(idata.posterior["mu_b"].mean())

        log.append(f"\n{'=' * 60}\n{kind.upper()}\n{'=' * 60}")

        text, _ = precision_block(idata, slopes, kind)
        log.append(text)

        log.append("\nROPE SENSITIVITY (tolerance as a multiple of the national drift)")
        log.append(rope_curve(idata, kind, mu_b).to_string(index=False))

        log.append("\nMULTIPLICITY")
        log.append(fdr_column(slopes, kind))

        tables[kind] = pd.read_csv(MODEL / f"slopes_{kind}.csv")

    log.append(f"\n{'=' * 60}")
    log.append(joint_table(tables["rate"], tables["share"]))

    text = "\n".join(log)
    (OUT / "03_precision_report.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
