"""Build the canonical analysis panels for Paper 2.

Two panels are produced, both COMPLETE (every unit-year present, including unit-years
with zero neonatal deaths). The archived panel dropped zero-death unit-years, which is
informative missingness concentrated in exactly the small units the paper argues about.

Outputs (written to data/processed/):
    panel_region_year.csv       region-year, cause counts + births + state
    panel_muni_year.csv         municipality-year, cause counts + births + state
    panel_build_report.txt      reconciliation of totals against the raw inputs

Run:
    .venv/bin/python 1-data/01_build_panel.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"

CAUSES = ["prenatal", "delivery", "newborn", "malformation", "illdef"]
YEARS = range(2014, 2025)


def load_inputs():
    """Read the cause panel, the birth series and the municipality/region crosswalk."""
    deaths = pd.read_csv(PROC / "muni_year_cause_panel.csv")
    births = pd.read_csv(PROC / "nascidos_muni_ano.csv").rename(
        columns={"ANO": "year", "nascimentos": "births"}
    )
    xwalk = pd.read_csv(REF / "muni_regiao_imediata.csv")[
        ["CODMUNRES", "rgi_id", "rgi_nome", "UF"]
    ]
    return deaths, births, xwalk


def build_complete_muni_panel(deaths, births, xwalk):
    """Expand to every municipality-year with a birth record, filling absent deaths with zero.

    The universe is defined by the birth series, not by the death series: a municipality-year
    with births and no deaths is a real observation of zero, not a missing value.
    """
    births = births[births.year.isin(YEARS)].copy()

    universe = (
        births[["CODMUNRES", "year"]]
        .drop_duplicates()
        .merge(births, on=["CODMUNRES", "year"], how="left")
    )

    panel = universe.merge(
        deaths.drop(columns=["births"], errors="ignore"),
        on=["CODMUNRES", "year"],
        how="left",
    )
    count_cols = CAUSES + ["deaths_total"]
    panel[count_cols] = panel[count_cols].fillna(0).astype("int64")

    panel = panel.merge(xwalk, on="CODMUNRES", how="left")

    # Deaths carrying a municipality code absent from the crosswalk cannot be placed in a
    # region; they are reported in the build report rather than silently dropped.
    panel["avoidable"] = panel["newborn"] + panel["delivery"]
    panel["four_group"] = panel[
        ["prenatal", "delivery", "newborn", "malformation"]
    ].sum(axis=1)
    return panel


def aggregate_to_region(muni_panel):
    """Collapse the municipality panel to IBGE immediate regions."""
    keep = muni_panel.dropna(subset=["rgi_id"]).copy()
    keep["rgi_id"] = keep["rgi_id"].astype("int64")

    agg = (
        keep.groupby(["rgi_id", "rgi_nome", "UF", "year"], as_index=False)[
            CAUSES + ["deaths_total", "births", "avoidable", "four_group"]
        ]
        .sum()
        .sort_values(["rgi_id", "year"])
        .reset_index(drop=True)
    )
    return agg


def write_report(deaths, muni_panel, region_panel, path):
    """Reconcile the built panels against the inputs so totals are auditable."""
    unmatched = muni_panel[muni_panel.rgi_id.isna()]
    lines = [
        "Paper 2 panel build report",
        "=" * 60,
        "",
        f"Years                         : {min(YEARS)}-{max(YEARS)}",
        "",
        "MUNICIPALITY PANEL",
        f"  rows (complete)             : {len(muni_panel):,}",
        f"  municipalities              : {muni_panel.CODMUNRES.nunique():,}",
        f"  unit-years with zero deaths : {(muni_panel.deaths_total == 0).sum():,}"
        f" ({(muni_panel.deaths_total == 0).mean():.1%})",
        f"  deaths                      : {muni_panel.deaths_total.sum():,}",
        f"  births                      : {muni_panel.births.sum():,}",
        "",
        "  reference (archived panel, zero-death unit-years dropped)",
        f"    rows                      : {len(deaths):,}",
        f"    deaths                    : {deaths.deaths_total.sum():,}",
        f"    deaths not carried over   : {deaths.deaths_total.sum() - muni_panel.deaths_total.sum():,}",
        "",
        "REGION PANEL",
        f"  rows                        : {len(region_panel):,}",
        f"  immediate regions           : {region_panel.rgi_id.nunique():,}",
        f"  deaths                      : {region_panel.deaths_total.sum():,}",
        f"  births                      : {region_panel.births.sum():,}",
        "",
        "CROSSWALK LOSSES",
        f"  municipality-years unmatched: {len(unmatched):,}",
        f"  deaths in unmatched rows    : {unmatched.deaths_total.sum():,}",
        f"  municipalities unmatched    : {unmatched.CODMUNRES.nunique():,}",
        "",
        "EXPOSURE TREND (the shrinking-denominator result)",
    ]
    by_year = region_panel.groupby("year").agg(
        births=("births", "sum"), deaths=("deaths_total", "sum")
    )
    by_year["nmr"] = 1000 * by_year.deaths / by_year.births
    for year, row in by_year.iterrows():
        lines.append(
            f"  {year}  births {row.births:>10,.0f}   deaths {row.deaths:>7,.0f}"
            f"   NMR {row.nmr:5.2f}"
        )
    first, last = by_year.index.min(), by_year.index.max()
    lines += [
        "",
        f"  births {first}->{last}: "
        f"{by_year.births[last] / by_year.births[first] - 1:+.1%}",
        f"  NMR    {first}->{last}: "
        f"{by_year.nmr[last] / by_year.nmr[first] - 1:+.1%}",
        "",
        "RELIABILITY TIERS (pooled deaths per unit, whole decade)",
    ]
    for label, panel, key in [
        ("municipalities", muni_panel, "CODMUNRES"),
        ("immediate regions", region_panel, "rgi_id"),
    ]:
        pooled = panel.groupby(key).deaths_total.sum()
        lines.append(
            f"  {label:<20} total {len(pooled):>5,}"
            f" | >=50 deaths {(pooled >= 50).sum():>5,}"
            f" | >=400 {(pooled >= 400).sum():>5,}"
            f" | >=800 {(pooled >= 800).sum():>5,}"
        )
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    deaths, births, xwalk = load_inputs()
    muni_panel = build_complete_muni_panel(deaths, births, xwalk)
    region_panel = aggregate_to_region(muni_panel)

    muni_panel.to_csv(PROC / "panel_muni_year.csv", index=False)
    region_panel.to_csv(PROC / "panel_region_year.csv", index=False)

    report = write_report(
        deaths, muni_panel, region_panel, PROC / "panel_build_report.txt"
    )
    print(report)


if __name__ == "__main__":
    main()
