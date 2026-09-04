"""
Step 0. ICD-10 underlying cause -> action group, and the municipality-year cause panel.

This is the step the README used to call "not part of this repository": the rule that
assigns every neonatal death's underlying cause (SIM field CAUSABAS) to one of five
groups, and the aggregation of those deaths to municipality-year counts. It is published
so that the mapping can be inspected and the cause panel rebuilt from the DATASUS
download.

Groups (the Brazilian list of avoidable causes of death by SUS interventions, under-5
chapter; the reference document is data/ref/lista_causas_evitaveis_DATASUS.pdf, not
versioned because it is a PDF freely available from DATASUS):

  prenatal      reducible by adequate care during pregnancy
                P00-P02, P04, P05, P07, P08, P55, A50
  delivery      reducible by adequate care during labour and delivery
                P03, P10-P15, P20, P21
  newborn       reducible by adequate care of the newborn
                every other P code except P96 (P06, P09, P22-P54 excluding P29-P54 that
                fall here too, P56-P95, P97-P99)
  malformation  congenital malformations, Q00-Q99; not reducible by SUS interventions
  illdef        ill-defined or not attributable: P96, R codes, external causes (V, W, X, Y)
                and every other chapter; also codes that are missing or shorter than 3
                characters

The paper's primary outcome, "avoidable", is delivery + newborn. "four_group" is
prenatal + delivery + newborn + malformation (ill-defined excluded), used for the
avoidable-share outcome in the supplement.

Residence codes. SIM records whose municipality-of-residence code ends in "0000" are the
state-level "municipality unknown" codes and cannot be placed in a municipality or an
immediate region. They are dropped here, before the panel exists. In 2014-2024 that is
196 of 260,219 records (0.075%), in 18 codes and 71 code-years.

Outputs (all in data/ref unless noted):
  icd10_action_group_rules.csv           the rule table, one row per ICD-10 range
  icd10_codes_observed_2014_2024.csv     every CAUSABAS value seen in the 2014-2024 SIM
                                         extract, its group, and its national death count
  data/processed/muni_year_cause_panel.csv   rebuilt only with --write; by default the
                                         rebuilt panel is compared with the existing file
                                         and the script asserts they are identical.

Usage:
  python3 1-data/00_cause_grouping.py            # rules + observed codes + reconciliation
  python3 1-data/00_cause_grouping.py --write    # also (re)write the cause panel
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "coorte_neonatal_2014_2024.csv"
REF = ROOT / "data" / "ref"
PROC = ROOT / "data" / "processed"
GROUPS = ["prenatal", "delivery", "newborn", "malformation", "illdef"]
YEARS = range(2014, 2025)


def cause_group(code: str | float) -> str:
    """Assign one ICD-10 underlying-cause code to an action group."""
    if pd.isna(code) or len(str(code)) < 3:
        return "illdef"
    code = str(code).upper()
    if code[:3] == "A50":  # congenital syphilis
        return "prenatal"
    letter = code[0]
    try:
        n = int(code[1:3])
    except ValueError:
        return "illdef"
    if letter == "Q":
        return "malformation"
    if letter == "P":
        if n in (0, 1, 2, 4, 5, 7, 8, 55):
            return "prenatal"
        if n == 3 or 10 <= n <= 15 or n in (20, 21):
            return "delivery"
        if n == 96:
            return "illdef"
        return "newborn"
    return "illdef"


RULES = [
    # (ICD-10 range, group, description)
    ("A50", "prenatal", "Congenital syphilis"),
    ("P00-P02", "prenatal", "Fetus and newborn affected by maternal conditions, maternal complications of pregnancy, complications of placenta, cord and membranes"),
    ("P04", "prenatal", "Fetus and newborn affected by noxious influences transmitted via placenta or breast milk"),
    ("P05", "prenatal", "Slow fetal growth and fetal malnutrition"),
    ("P07", "prenatal", "Disorders related to short gestation and low birthweight"),
    ("P08", "prenatal", "Disorders related to long gestation and high birthweight"),
    ("P55", "prenatal", "Haemolytic disease of fetus and newborn"),
    ("P03", "delivery", "Fetus and newborn affected by other complications of labour and delivery"),
    ("P10-P15", "delivery", "Birth trauma"),
    ("P20-P21", "delivery", "Intrauterine hypoxia and birth asphyxia"),
    ("P06, P09", "newborn", "Other P0x codes not listed above"),
    ("P22-P28", "newborn", "Respiratory disorders specific to the perinatal period"),
    ("P29", "newborn", "Cardiovascular disorders originating in the perinatal period"),
    ("P35-P39", "newborn", "Infections specific to the perinatal period"),
    ("P50-P54", "newborn", "Haemorrhagic disorders of fetus and newborn"),
    ("P56-P61", "newborn", "Other haemolytic and haematological disorders (P55 excluded)"),
    ("P70-P74", "newborn", "Transitory endocrine and metabolic disorders"),
    ("P75-P78", "newborn", "Digestive system disorders of fetus and newborn"),
    ("P80-P83", "newborn", "Conditions involving the integument and temperature regulation"),
    ("P90-P95", "newborn", "Other disorders originating in the perinatal period (P96 excluded)"),
    ("P97-P99", "newborn", "Other P codes"),
    ("Q00-Q99", "malformation", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("P96", "illdef", "Other conditions originating in the perinatal period, including P96.9 unspecified"),
    ("R00-R99", "illdef", "Symptoms, signs and abnormal findings not elsewhere classified"),
    ("V01-Y98", "illdef", "External causes of morbidity and mortality"),
    ("all other chapters", "illdef", "A (except A50), B, C, D, E, F, G, H, I, J, K, L, M, N"),
    ("missing or < 3 characters", "illdef", "Unusable code"),
]


def write_rules() -> None:
    pd.DataFrame(RULES, columns=["icd10", "group", "description"]).to_csv(
        REF / "icd10_action_group_rules.csv", index=False
    )


def load_sim() -> pd.DataFrame:
    sim = pd.read_csv(
        RAW, sep=";", dtype=str, usecols=["CODMUNRES", "CAUSABAS", "DTOBITO"], low_memory=False
    )
    sim["CODMUNRES"] = sim["CODMUNRES"].str.strip().str[:6]
    sim["year"] = sim["DTOBITO"].str[-4:].astype(int)
    n_raw = len(sim)
    unknown = sim["CODMUNRES"].str.endswith("0000")
    print(f"SIM records 2014-2024           : {n_raw:,}")
    print(
        f"residence unknown (…0000) dropped: {int(unknown.sum()):,} records, "
        f"{sim.loc[unknown, 'CODMUNRES'].nunique()} codes, "
        f"{sim.loc[unknown].groupby(['CODMUNRES', 'year']).ngroups} code-years"
    )
    sim = sim.loc[~unknown].copy()
    sim["group"] = sim["CAUSABAS"].map(cause_group)
    print(f"deaths carried to the panel      : {len(sim):,}")
    return sim


def write_observed(sim: pd.DataFrame) -> None:
    obs = (
        sim.groupby(["CAUSABAS", "group"], as_index=False)
        .size()
        .rename(columns={"size": "deaths_2014_2024"})
        .sort_values(["group", "CAUSABAS"])
    )
    obs.to_csv(REF / "icd10_codes_observed_2014_2024.csv", index=False)
    print(f"distinct CAUSABAS values         : {len(obs):,}")
    dist = sim["group"].value_counts().reindex(GROUPS)
    for g, n in dist.items():
        print(f"  {g:<13} {n:>8,}  {n / len(sim) * 100:5.1f}%")
    print(f"  avoidable (delivery+newborn)  {int(dist['delivery'] + dist['newborn']):,}")


def build_panel(sim: pd.DataFrame) -> pd.DataFrame:
    panel = (
        sim.pivot_table(
            index=["CODMUNRES", "year"], columns="group", values="CAUSABAS",
            aggfunc="size", fill_value=0,
        )
        .reindex(columns=GROUPS, fill_value=0)
        .reset_index()
    )
    panel.columns.name = None
    panel["deaths_total"] = panel[GROUPS].sum(axis=1)
    return panel


def reconcile(panel: pd.DataFrame) -> None:
    existing_path = PROC / "muni_year_cause_panel.csv"
    if not existing_path.exists():
        print("no existing cause panel to compare with")
        return
    old = pd.read_csv(existing_path, dtype={"CODMUNRES": str})
    old["CODMUNRES"] = old["CODMUNRES"].str[:6]
    cols = ["CODMUNRES", "year"] + GROUPS + ["deaths_total"]
    a = old[cols].sort_values(["CODMUNRES", "year"]).reset_index(drop=True)
    b = panel[cols].sort_values(["CODMUNRES", "year"]).reset_index(drop=True)
    b["year"] = b["year"].astype(a["year"].dtype)
    for c in GROUPS + ["deaths_total"]:
        b[c] = b[c].astype(a[c].dtype)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print(
        f"rebuilt cause panel identical to data/processed/muni_year_cause_panel.csv "
        f"({len(b):,} municipality-years, {int(b.deaths_total.sum()):,} deaths)"
    )


def main() -> None:
    write_rules()
    print(f"wrote {REF / 'icd10_action_group_rules.csv'}")
    if not RAW.exists():
        print(f"{RAW} not present; rules written, observed codes and panel skipped")
        return
    sim = load_sim()
    write_observed(sim)
    panel = build_panel(sim)
    reconcile(panel)
    if "--write" in sys.argv:
        old = pd.read_csv(PROC / "muni_year_cause_panel.csv", dtype={"CODMUNRES": str})
        births = old[["CODMUNRES", "year", "births"]] if "births" in old else None
        out = panel if births is None else panel.merge(births, on=["CODMUNRES", "year"], how="left")
        out.to_csv(PROC / "muni_year_cause_panel.csv", index=False)
        print("wrote data/processed/muni_year_cause_panel.csv")


if __name__ == "__main__":
    main()
