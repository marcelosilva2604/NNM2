# ICD-10 underlying cause to action group

This is the mapping behind every cause count in the paper. It is applied by
`1-data/00_cause_grouping.py`, which also rebuilds the municipality-year cause panel from
the SIM download and asserts that the rebuilt panel is identical to the one the analysis
used. Two machine-readable files sit next to this note:

- `icd10_action_group_rules.csv`: the rule table, one row per ICD-10 range.
- `icd10_codes_observed_2014_2024.csv`: every value of the SIM field CAUSABAS observed among
  the 260,023 analysed neonatal deaths, its group, and its national death count for the
  decade. National counts by four-character code are aggregate and non-identifying.

## Source

The grouping follows the Brazilian list of avoidable causes of death by interventions of
the Unified Health System (SUS), chapter for children under five years, as published by
the Ministry of Health through DATASUS (reference document:
`lista_causas_evitaveis_DATASUS.pdf`, obtained from DATASUS; not versioned here because it
is a PDF freely available from the source). The list defines avoidability by the stage of
care at which the death is reducible. Congenital malformations are listed as not
reducible by SUS interventions; ill-defined causes and causes outside the perinatal and
malformation chapters form a residual.

## Rules

| Group | ICD-10 | Meaning |
|---|---|---|
| prenatal | A50; P00-P02; P04; P05; P07; P08; P55 | reducible by adequate care during pregnancy |
| delivery | P03; P10-P15; P20; P21 | reducible by adequate care during labour and delivery |
| newborn | every other P code except P96 | reducible by adequate care of the newborn |
| malformation | Q00-Q99 | congenital malformations, not reducible by SUS interventions |
| illdef | P96; R00-R99; V01-Y98; all other chapters; missing or unusable code | ill-defined or not attributable |

Precedence: A50 first; then chapter Q; then chapter P by the two-digit block; everything
else is ill-defined. Codes are matched on their first three characters, so a fourth
character never changes the group.

Paper outcomes built from the groups:

- **avoidable** (primary outcome): delivery + newborn.
- **four_group** (denominator of the avoidable-share outcome in the supplement): prenatal
  + delivery + newborn + malformation. Ill-defined deaths are excluded from the share.

## Reconciliation, 2014-2024

| Quantity | Count |
|---|---|
| SIM neonatal death records in the extract | 260,219 |
| dropped: residence code ending in 0000 (municipality unknown), 18 codes, 71 code-years | 196 |
| deaths in the cause panel | 260,023 |
| prenatal | 73,598 |
| delivery | 19,971 |
| newborn | 97,607 |
| malformation | 54,620 |
| illdef | 14,227 |
| avoidable (delivery + newborn) | 117,578 |

Running `python3 1-data/00_cause_grouping.py` with the SIM extract in `data/raw`
reproduces this table and asserts that the rebuilt panel equals
`data/processed/muni_year_cause_panel.csv` row by row.

## Limits

The list assigns avoidability by cause, not by review of individual cases. Deaths coded to
prematurity (P07) sit in the prenatal group even when the neonatal care received would
have mattered, and deaths coded to ill-defined perinatal conditions (P96) leave the
action groups altogether. The paper reports the ill-defined share by year and its
correlation with the estimated slopes precisely because this residual could move. The
mapping itself was not validated against medical records.
