"""
Plot a mapped substation network on an interactive OSM map for visual
sense-checking.

Reads directly from graph.sqlite using the schema defined by
create_graph_db() and NetworkData.to_sql():

    incidence_list  (edge_fid, node_from, node_to, parent_substation, ...)
    node_list       (fid, Geometry [WKB], Asset_Type, Voltage, ...)
    edge_list       (fid, Geometry [WKB], Asset_Type, Voltage, ...)

The substation node itself is not stored in node_list — its geometry is
fetched from lv_assets.sqlite.

Usage:
    uv run python validation/plot_network.py 11375001
    uv run python validation/plot_network.py 11375001 --buildings
    uv run python validation/plot_network.py 11375001 --buildings --matches
    uv run python validation/plot_network.py 11375001 --db results/test/graph.sqlite
"""

import argparse
import sqlite3
import sys

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.wkb import loads


# ── Schema-based queries ──────────────────────────────────────────

def load_network_from_db(fid, db_path):
    """
    Load node and edge geometries for a single substation directly from
    graph.sqlite, using the schema from create_graph_db / to_sql.

    Returns (nodes_gdf, edges_gdf) in EPSG:27700.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()

    # -- 1. Get all node FIDs from the incidence list --
    cur.execute(
        "SELECT DISTINCT node_from FROM incidence_list "
        "WHERE parent_substation = ?", (fid,)
    )
    nodes_from = {r[0] for r in cur.fetchall()}

    cur.execute(
        "SELECT DISTINCT node_to FROM incidence_list "
        "WHERE parent_substation = ?", (fid,)
    )
    nodes_to = {r[0] for r in cur.fetchall()}
    node_fids = nodes_from | nodes_to

    if not node_fids:
        conn.close()
        sys.exit(f"No incidence rows found for FID {fid} in {db_path}")

    # -- 2. Load node geometries and asset types --
    node_records = []
    for nfid in node_fids:
        cur.execute(
            "SELECT Geometry, Asset_Type FROM node_list WHERE fid = ?",
            (nfid,)
        )
        row = cur.fetchone()
        if row is None:
            # The substation itself isn't in node_list; skip for now
            continue
        geom = loads(row[0])
        if not isinstance(geom, Point):
            geom = geom.centroid
        node_records.append({
            "fid": nfid,
            "asset_type": row[1],
            "geometry": geom,
        })

    # -- 3. Load substation geometry from lv_assets --
    lv_conn = sqlite3.connect("data/lv_assets.sqlite", timeout=30)
    lv_cur = lv_conn.cursor()
    lv_cur.execute("SELECT Geometry FROM lv_assets WHERE fid = ?", (fid,))
    sub_row = lv_cur.fetchone()
    lv_cur.close()
    lv_conn.close()

    if sub_row is not None:
        sub_geom = loads(sub_row[0])
        node_records.append({
            "fid": fid,
            "asset_type": "Substation",
            "geometry": sub_geom.centroid,
        })

    # -- 4. Load edge geometries --
    cur.execute(
        "SELECT edge_fid, node_from, node_to FROM incidence_list "
        "WHERE parent_substation = ?", (fid,)
    )
    incidence_rows = cur.fetchall()

    edge_records = []
    for edge_fid, node_from, node_to in incidence_rows:
        cur.execute(
            "SELECT Geometry, Asset_Type FROM edge_list WHERE fid = ?",
            (edge_fid,)
        )
        erow = cur.fetchone()
        if erow is None:
            continue
        geom = loads(erow[0])
        edge_records.append({
            "edge_fid": edge_fid,
            "node_from": node_from,
            "node_to": node_to,
            "asset_type": erow[1],
            "geometry": geom,
        })

    cur.close()
    conn.close()

    nodes_gdf = gpd.GeoDataFrame(node_records, crs="EPSG:27700")
    edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:27700")
    return nodes_gdf, edges_gdf


# ── Building loading ──────────────────────────────────────────────

def load_buildings_near_network(
    nodes_gdf,
    edges_gdf,
    buildings_path="data/all_buildings_in_enwl_27700.parquet",
    buffer_scale=1.7,
):
    """
    Load building footprints from parquet that fall within a buffered
    convex hull of the network geometries.

    Parameters
    ----------
    nodes_gdf, edges_gdf : GeoDataFrame
        Network geometries in EPSG:27700.
    buildings_path : str
        Path to the buildings GeoParquet file.
    buffer_scale : float
        Buffer factor applied to the convex hull of the network outline.

    Returns
    -------
    GeoDataFrame in EPSG:4326 with columns: toid, activity_type, geometry.
    """
    all_geoms = pd.concat(
        [nodes_gdf.geometry, edges_gdf.geometry], ignore_index=True
    )
    combined = unary_union(all_geoms)
    outline = combined.convex_hull
    buffered = outline.buffer(
        outline.length * (buffer_scale - 1) / (2 * 3.14159)
    )

    minx, miny, maxx, maxy = buffered.bounds
    buildings = gpd.read_parquet(
        buildings_path,
        columns=["uprn", "toid", "activity_type", "geometry"],
    )
    buildings = buildings.cx[minx:maxx, miny:maxy]
    buildings = buildings[buildings.within(buffered)].copy()

    return buildings.to_crs(epsg=4326)


ACTIVITY_CODE_MAP = {
    "DO": "domestic",
}
DEFAULT_ACTIVITY = "no address"

ACTIVITY_COLORS = {
    "domestic": "rgba(0,128,0,0.45)",
    "non-domestic": "rgba(255,165,0,0.45)",
    "no address": "rgba(128,0,128,0.45)",
    "mixed": "rgba(0,255,255,0.45)",
}
DEFAULT_BUILDING_COLOR = "rgba(128,128,128,0.45)"

MATCH_COLORS = {
    "overlap": "rgba(0,128,0,0.5)",
    "buffer_1m": "rgba(0,100,255,0.5)",
    "unmatched": "rgba(200,200,200,0.3)",
}


def _classify_activity(code):
    """Map 2-letter activity_type codes to display categories."""
    if pd.isna(code) or code == "":
        return DEFAULT_ACTIVITY
    return ACTIVITY_CODE_MAP.get(code, "non-domestic")


def build_building_traces(buildings_gdf, match_df=None):
    """
    Create Plotly polygon traces for building footprints.

    If match_df is provided, buildings are colored by match method;
    otherwise they are colored by activity_type category.
    """
    traces = []

    if match_df is not None and not match_df.empty:
        bldg = buildings_gdf.merge(
            match_df[["toid", "match_method"]].drop_duplicates("toid"),
            on="toid",
            how="left",
        )
        bldg["match_method"] = bldg["match_method"].fillna("unmatched")
        group_col, color_map = "match_method", MATCH_COLORS
    else:
        bldg = buildings_gdf.copy()
        bldg["activity_group"] = bldg["activity_type"].apply(_classify_activity)
        group_col, color_map = "activity_group", ACTIVITY_COLORS

    for group in bldg[group_col].unique():
        subset = bldg[bldg[group_col] == group]
        color = color_map.get(group, DEFAULT_BUILDING_COLOR)

        lons, lats, hovers = [], [], []
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                xs, ys = poly.exterior.coords.xy
                lons.extend(list(xs) + [None])
                lats.extend(list(ys) + [None])
                toid = row.get("toid", "")
                activity = row.get("activity_type", "")
                hovers.extend(
                    [f"TOID: {toid}<br>Activity: {activity}"]
                    + [None] * len(xs)
                )

        if lons:
            traces.append(go.Scattermap(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=1, color="black"),
                fill="toself",
                fillcolor=color,
                opacity=0.9,
                hovertext=hovers,
                hoverinfo="text",
                name=f"Building: {group}",
            ))

    return traces


# ── Plotting ──────────────────────────────────────────────────────

NODE_COLORS = {
    "Substation":    "black",
    "Service Point": "red",
    "Switch":        "orange",
}
DEFAULT_NODE_COLOR = "gray"  # lamp posts / other

EDGE_COLORS = {
    "LV Conductor": "blue",
    "LV Service":   "green",
}
DEFAULT_EDGE_COLOR = "steelblue"


def build_edge_traces(edges_gdf):
    """One Scattermap trace per edge asset type (for a clean legend)."""
    traces = []
    for asset_type in edges_gdf["asset_type"].unique():
        subset = edges_gdf[edges_gdf["asset_type"] == asset_type]
        color = EDGE_COLORS.get(asset_type, DEFAULT_EDGE_COLOR)

        lons, lats = [], []
        for geom in subset.geometry:
            if geom is None:
                continue
            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                xs, ys = line.xy
                lons.extend(list(xs) + [None])
                lats.extend(list(ys) + [None])

        traces.append(go.Scattermap(
            lon=lons, lat=lats,
            mode="lines",
            line=dict(width=2, color=color),
            name=asset_type or "Edge",
            legendgroup=f"edge_{asset_type}",
            hoverinfo="name",
        ))
    return traces


def build_node_traces(nodes_gdf):
    """One Scattermap trace per node asset type."""
    traces = []
    for asset_type in nodes_gdf["asset_type"].unique():
        subset = nodes_gdf[nodes_gdf["asset_type"] == asset_type]
        color = NODE_COLORS.get(asset_type, DEFAULT_NODE_COLOR)
        size = 12 if asset_type == "Substation" else 7

        traces.append(go.Scattermap(
            lon=subset.geometry.x,
            lat=subset.geometry.y,
            mode="markers",
            marker=dict(size=size, color=color),
            text=subset.apply(
                lambda r: f"FID: {r['fid']}<br>Type: {r['asset_type']}", axis=1
            ),
            hoverinfo="text",
            name=asset_type or "Node",
        ))
    return traces


def plot_network(fid, db_path, buildings=False, matches_path=None,
                 buildings_path="data/all_buildings_in_enwl_27700.parquet"):
    nodes_gdf, edges_gdf = load_network_from_db(fid, db_path)

    # Load buildings before CRS conversion (parquet is EPSG:27700)
    building_traces = []
    n_buildings = 0
    if buildings:
        buildings_gdf = load_buildings_near_network(
            nodes_gdf, edges_gdf, buildings_path=buildings_path
        )
        n_buildings = len(buildings_gdf)

        match_df = None
        if matches_path is not None:
            try:
                all_matches = pd.read_csv(matches_path)
                match_df = all_matches[
                    all_matches["substation_fid"] == fid
                ]
            except (FileNotFoundError, KeyError):
                match_df = None

        building_traces = build_building_traces(buildings_gdf, match_df)

    # Convert to WGS84 for Plotly
    nodes_gdf = nodes_gdf.to_crs(epsg=4326)
    edges_gdf = edges_gdf.to_crs(epsg=4326)

    edge_traces = build_edge_traces(edges_gdf)
    node_traces = build_node_traces(nodes_gdf)

    fig = go.Figure(building_traces + edge_traces + node_traces)
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=nodes_gdf.geometry.y.mean(),
                lon=nodes_gdf.geometry.x.mean(),
            ),
            zoom=15,
        ),
        title=f"Mapped network — substation {fid}",
        legend=dict(title="Asset type", bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.show()

    n_nodes = len(nodes_gdf)
    n_edges = len(edges_gdf)
    n_sp = (nodes_gdf["asset_type"] == "Service Point").sum()
    msg = f"Plotted {n_nodes} nodes ({n_sp} service points), {n_edges} edges"
    if buildings:
        msg += f", {n_buildings} buildings"
    print(msg)


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot a mapped substation network on an interactive map."
    )
    parser.add_argument("fid", type=int, help="Substation FID to plot")
    parser.add_argument(
        "--db", default="results/graph.sqlite",
        help="Path to graph.sqlite (default: results/graph.sqlite)",
    )
    parser.add_argument(
        "--buildings", action="store_true",
        help="Overlay nearby building footprints from the parquet file",
    )
    parser.add_argument(
        "--buildings-path", default="data/all_buildings_in_enwl_27700.parquet",
        help="Path to the buildings GeoParquet file",
    )
    parser.add_argument(
        "--matches", default=None, nargs="?",
        const="results/building_matches.csv",
        help="Color buildings by match method from building_matches.csv "
             "(implies --buildings)",
    )
    args = parser.parse_args()

    show_buildings = args.buildings or args.matches is not None
    plot_network(
        args.fid, args.db,
        buildings=show_buildings,
        matches_path=args.matches,
        buildings_path=args.buildings_path,
    )


if __name__ == "__main__":
    main()
