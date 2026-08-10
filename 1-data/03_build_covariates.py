"""Region-level covariates for the covariate-assisted models.

Three covariates are assembled, deliberately of two different kinds because the
distinction is the point of the analysis that consumes them.

    idhm            Municipal Human Development Index, 2010. STATIC: one value per
                    region for the whole decade. A static covariate can shift a region's
                    level but carries no information about how that level changed, so it
                    cannot, in principle, sharpen a trend.

    nicu_per_1000   Neonatal intensive care beds per 1,000 live births, by year.
    ubs_per_1000    Primary care units per 1,000 live births, by year.
                    Both TIME-VARYING, from the establishment registry (CNES).

Municipal values are aggregated to immediate regions weighted by live births, so a
region's covariate reflects the conditions the region's babies were actually born into
rather than an unweighted average over municipalities of very different size.

A caution that belongs with the output rather than only in the manuscript: CNES is an
establishment registry. A change in a region's bed count may record a real change in
capacity or a change in registration practice, and this dataset cannot separate the two.

Outputs (data/processed/):
    covariates_region_year.csv
    covariates_report.txt

Run:
    .venv/bin/python 1-data/03_build_covariates.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"


def load():
    panel = pd.read_csv(PROC / "panel_muni_year.csv")
    panel = panel.dropna(subset=["rgi_id"]).copy()
    panel["rgi_id"] = panel.rgi_id.astype("int64")

    idhm = pd.read_csv(PROC / "idhm_muni.csv")[
        ["CODMUNRES", "idhm", "idhm_renda", "idhm_educ"]
    ]
    cnes = pd.read_csv(PROC / "cnes_muni_ano.csv").rename(
        columns={"ANO": "year", "ubs": "ubs", "uti_neo_leitos": "nicu"}
    )
    return panel, idhm, cnes


def build(panel, idhm, cnes):
    """Aggregate municipal covariates to region-years, weighting by live births."""
    # Static index: one birth-weighted value per region, constant across years.
    births_total = panel.groupby("CODMUNRES", as_index=False).births.sum()
    static = (
        idhm.merge(births_total, on="CODMUNRES")
        .merge(panel[["CODMUNRES", "rgi_id"]].drop_duplicates(), on="CODMUNRES")
        .dropna(subset=["idhm"])
    )
    static = static[static.births > 0]
    idhm_region = (
        static.groupby("rgi_id")
        .apply(
            lambda g: pd.Series(
                {
                    "idhm": np.average(g.idhm, weights=g.births),
                    "idhm_renda": np.average(g.idhm_renda, weights=g.births),
                    "idhm_educ": np.average(g.idhm_educ, weights=g.births),
                    "idhm_coverage": g.births.sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Time-varying capacity: summed over the region, expressed per 1,000 live births so
    # it is a rate rather than a count that simply tracks region size.
    cap = panel[["CODMUNRES", "year", "rgi_id", "births"]].merge(
        cnes[["CODMUNRES", "year", "ubs", "nicu"]], on=["CODMUNRES", "year"], how="left"
    )
    cap[["ubs", "nicu"]] = cap[["ubs", "nicu"]].fillna(0)
    region_year = (
        cap.groupby(["rgi_id", "year"], as_index=False)[["births", "ubs", "nicu"]].sum()
    )
    region_year["nicu_per_1000"] = 1000 * region_year.nicu / region_year.births
    region_year["ubs_per_1000"] = 1000 * region_year.ubs / region_year.births

    out = region_year.merge(idhm_region, on="rgi_id", how="left")

    # Standardise so coefficients are per standard deviation and priors are comparable.
    for col in ("idhm", "nicu_per_1000", "ubs_per_1000"):
        out[f"{col}_z"] = (out[col] - out[col].mean()) / out[col].std()
    return out


def report(out, panel, path):
    """Coverage, variation, and how much of each covariate is between versus within region."""
    lines = ["Region-level covariates", "=" * 60, ""]
    lines.append(f"region-years        : {len(out):,}")
    lines.append(f"regions             : {out.rgi_id.nunique():,}")
    lines.append(f"missing idhm        : {int(out.idhm.isna().sum())} region-years")
    lines.append("")

    covered = out.groupby("rgi_id").idhm_coverage.first().sum()
    lines.append(
        f"births under a matched IDHM municipality: {covered:,.0f} of "
        f"{panel.births.sum():,.0f} ({100 * covered / panel.births.sum():.1f}%)"
    )
    lines.append("")

    for col in ("idhm", "nicu_per_1000", "ubs_per_1000"):
        v = out[col].dropna()
        # Variance decomposition: a covariate with no within-region variance cannot
        # inform a trend, whatever its between-region spread.
        grand = v.mean()
        between = out.groupby("rgi_id")[col].transform("mean")
        within_var = float(((out[col] - between) ** 2).mean())
        between_var = float(((between - grand) ** 2).mean())
        share = within_var / (within_var + between_var) if (within_var + between_var) else 0.0
        lines.append(
            f"{col:>14}: mean {v.mean():8.3f}  sd {v.std():7.3f}  "
            f"range {v.min():7.3f} to {v.max():8.3f}  within-region share of variance "
            f"{share:.3f}"
        )

    lines += [
        "",
        "Note: IDHM is fixed at 2010 and has, by construction, zero within-region",
        "variance. It can shift a region's level but cannot carry information about the",
        "direction of change within that region.",
        "",
        "Note: CNES is an establishment registry. A change in a region's counts may",
        "record a change in capacity or a change in registration; this dataset cannot",
        "distinguish them.",
    ]
    text = "\n".join(lines)
    path.write_text(text + "\n")
    return text


def main():
    panel, idhm, cnes = load()
    out = build(panel, idhm, cnes)
    out.to_csv(PROC / "covariates_region_year.csv", index=False)
    print(report(out, panel, PROC / "covariates_report.txt"))


if __name__ == "__main__":
    main()
