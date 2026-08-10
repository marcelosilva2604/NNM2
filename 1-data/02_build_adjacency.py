"""Build the immediate-region adjacency graph the spatial models need.

Municipal polygons are dissolved into IBGE immediate regions with the same crosswalk the
panel uses, then queen contiguity gives the neighbour list. Islands (regions with no
land neighbour) are connected to their nearest region by centroid distance, because an
ICAR prior on a disconnected graph is improper on each isolated component.

Outputs (data/processed/):
    adjacency_pairs.csv     one row per undirected neighbour pair (i < j), 0-indexed
    region_centroids.csv    rgi_id, longitude, latitude, state
    adjacency_report.txt    connectivity summary

Run:
    .venv/bin/python 1-data/02_build_adjacency.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"


def load_regions():
    """Dissolve municipal polygons into immediate regions, in the panel's region order."""
    munis = gpd.read_file(REF / "br_municipios.geojson")

    # The geometry file keys municipalities by the 7-digit IBGE code; the crosswalk
    # carries both that and the 6-digit code the health data uses.
    code_col = next(
        c
        for c in munis.columns
        if munis[c].astype(str).str.fullmatch(r"\d{7}").fillna(False).mean() > 0.9
    )
    munis["cod7"] = munis[code_col].astype("int64")

    xwalk = pd.read_csv(REF / "muni_regiao_imediata.csv")[["cod7", "rgi_id", "UF"]]
    merged = munis.merge(xwalk, on="cod7", how="inner")

    # The published mesh carries self-intersections and unassignable holes that make the
    # union fail outright; repairing first is required, not cosmetic.
    invalid = int((~merged.geometry.is_valid).sum())
    if invalid:
        merged["geometry"] = merged.geometry.make_valid()
        print(f"repaired {invalid} invalid municipal geometries")

    regions = merged.dissolve(by="rgi_id", aggfunc={"UF": "first"}).reset_index()

    # Match the ordering used by the canonical fit so indices line up with the posterior.
    panel_regions = np.sort(pd.read_csv(PROC / "panel_region_year.csv").rgi_id.unique())
    regions = (
        regions.set_index("rgi_id").reindex(panel_regions).reset_index()
    )
    return regions, panel_regions


def build_graph(regions):
    """Queen contiguity, then attach any isolated region to its nearest neighbour."""
    w = Queen.from_dataframe(regions, use_index=False)
    neighbours = {i: set(v) for i, v in w.neighbors.items()}

    proj = regions.to_crs(5880)  # Brazil Polyconic, metric, for honest distances
    centroids = proj.geometry.centroid
    xy = np.column_stack([centroids.x.values, centroids.y.values])

    islands = [i for i, v in neighbours.items() if not v]

    # An ICAR prior is improper on every disconnected component, so the graph is welded
    # into one piece: each component is joined to the rest at its closest centroid pair.
    # Queen contiguity alone leaves five components here (offshore regions, and coastlines
    # whose published polygons do not quite touch).
    joins = []
    while True:
        labels = _component_labels(n_regions=len(xy), neighbours=neighbours)
        if labels.max() == 0:
            break
        target = labels == 0
        other = ~target
        best = None
        for i in np.where(other)[0]:
            d = np.hypot(xy[target, 0] - xy[i, 0], xy[target, 1] - xy[i, 1])
            k = int(np.argmin(d))
            j = int(np.where(target)[0][k])
            if best is None or d[k] < best[0]:
                best = (float(d[k]), i, j)
        _, i, j = best
        neighbours[i].add(j)
        neighbours[j].add(i)
        joins.append((i, j, best[0] / 1000.0))

    pairs = sorted({(min(i, j), max(i, j)) for i, v in neighbours.items() for j in v})
    return pairs, neighbours, islands, joins


def _component_labels(n_regions, neighbours):
    """Label connected components by breadth-first search."""
    labels = np.full(n_regions, -1)
    current = 0
    for start in range(n_regions):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current
        while stack:
            node = stack.pop()
            for nb in neighbours[node]:
                if labels[nb] == -1:
                    labels[nb] = current
                    stack.append(nb)
        current += 1
    return labels


def connected_components(n, pairs):
    """Count components so a disconnected graph is caught before it reaches a model."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n)})


def main():
    regions, panel_regions = load_regions()
    pairs, neighbours, islands, joins = build_graph(regions)

    n = len(regions)
    degrees = np.array([len(neighbours[i]) for i in range(n)])
    n_comp = connected_components(n, pairs)

    pd.DataFrame(pairs, columns=["i", "j"]).to_csv(
        PROC / "adjacency_pairs.csv", index=False
    )

    # Centroids are computed on the projected geometry and only then converted back, so
    # they are not distorted by taking a mean of longitudes and latitudes.
    wgs = regions.to_crs(5880).geometry.centroid.to_crs(4326)
    pd.DataFrame(
        {
            "rgi_id": regions.rgi_id.values,
            "lon": wgs.x.values,
            "lat": wgs.y.values,
            "UF": regions.UF.values,
        }
    ).to_csv(PROC / "region_centroids.csv", index=False)

    report = "\n".join(
        [
            "Immediate-region adjacency",
            "=" * 50,
            f"regions              : {n}",
            f"matches panel order  : {bool((regions.rgi_id.values == panel_regions).all())}",
            f"undirected pairs     : {len(pairs):,}",
            f"mean degree          : {degrees.mean():.2f}",
            f"min / max degree     : {degrees.min()} / {degrees.max()}",
            f"islands (no queen nb): {len(islands)}",
            f"components welded    : {len(joins)}",
            *[
                f"    joined {regions.rgi_id.iloc[i]} <-> {regions.rgi_id.iloc[j]}"
                f"  ({km:.0f} km apart)"
                for i, j, km in joins
            ],
            f"connected components : {n_comp}"
            f"{'  (OK for ICAR)' if n_comp == 1 else '  (WARNING: ICAR improper)'}",
        ]
    )
    (PROC / "adjacency_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
