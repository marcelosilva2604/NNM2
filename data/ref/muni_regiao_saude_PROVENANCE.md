# Municipality to health region crosswalk

`muni_regiao_saude.csv`, 5,572 rows, one per Brazilian municipality.

| Column | Meaning |
|---|---|
| `cod6` | six-digit IBGE municipality code, matching `CODMUNRES[:6]` in the panels |
| `regsaud_id` | five-digit health region code (first two digits are the state) |
| `regsaud_nome` | health region name as published |

## Source

DATASUS territorial base, **December 2024 vintage**, downloaded from
`ftp://ftp.datasus.gov.br/territorio/tabelas/2024/` (file `12-base territorial_dez24.zip`).

Two tables from that archive were used:

- `rl_municip_regsaud.csv`, the municipality-to-health-region relation
- `tb_regsaud.csv`, health region names, filtered to `CO_STATUS = ATIVO`

Health regions (*regiões de saúde*) are defined by each state's Comissão Intergestores
Regional under Decree 7,508/2011. They are a health-planning geography, not a statistical
one, which is exactly why this file exists: the paper's ladder otherwise uses only IBGE's
immediate regions, an economic geography.

## What the file resolves to

- **433 health regions** contain at least one municipality. `tb_regsaud` lists 460 as
  active, but 27 of those have no municipality assigned in this vintage: 20 are Ceará's
  older numbered regions, superseded when the state repactuated into 5, and 7 are the
  Distrito Federal's internal health regions, which cannot be separated because DF is a
  single municipality of residence.
- **23 municipality codes in the panel do not map.** These are the same 23 codes that fail
  the immediate-region crosswalk: placeholder codes ending in `0000` for unknown
  municipality of residence. They carry 980 births and no deaths, so they cannot affect
  any numerator.

## The caveat that matters

Health regions were repactuated during the study decade in several states, most visibly
Ceará (22 regions to 5) and Espírito Santo. Applying the December 2024 partition
retrospectively to 2014-2024 therefore assigns some municipalities to a region that did not
exist in that form for the whole period.

This is the same treatment the IBGE immediate regions receive, since those follow the 2017
division, and it is the standard choice when the estimand is a within-unit trend: the unit
must be held fixed for a trend within it to mean anything. But it is an assumption, not a
fact about the data, and the supplement states it.

The two states where it bites hardest are also the two where health regions are largest, so
those units gain exposure rather than losing it, which works against the paper's conclusion
rather than for it.
