import os
import sqlite3
import pandas as pd
import geopandas as gpd
from shapely import from_wkb
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import contextily as cx
from typing import Union
from shapely.geometry import LineString, Point
from gridstock.network_loader import MappedNetwork
import networkx as nx
import sys
import time
from shapely.ops import unary_union
from shapely.wkb import loads
import plotly.graph_objects as go
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


# ── Per-worker building parquet cache ─────────────────────────────
# When running under mp.Pool with _init_worker as the initializer,
# each worker reads the 234 MB parquet once and reuses it for every
# substation assigned to that process.

_BUILDINGS_CACHE = None

_BUILDINGS_COLUMNS = ["uprn", "toid", "activity_type", "activity_type_2", "geometry"]


def _init_worker(buildings_path):
    """Pool initializer: load buildings parquet once per worker process."""
    global _BUILDINGS_CACHE
    _BUILDINGS_CACHE = gpd.read_parquet(
        buildings_path, columns=_BUILDINGS_COLUMNS,
    )
    _BUILDINGS_CACHE.sindex  # pre-build spatial index


def get_building_geom(buffered, buildings_path="data/all_buildings_in_enwl_27700_bbox.parquet"):
    """
    Return building footprints that fall within a buffered network area.

    Uses the per-worker cache if available (parallel mode), otherwise
    reads from parquet with bbox pushdown (single-FID / serial mode).

    Parameters
    ----------
    buffered : shapely.geometry
        Buffered convex hull of the network (EPSG:27700).
    buildings_path : str
        Path to the GeoParquet file containing building geometries.

    Returns
    -------
    GeoDataFrame
        Subset of buildings within the buffered network area.
    """
    minx, miny, maxx, maxy = buffered.bounds

    if _BUILDINGS_CACHE is not None:
        # Parallel mode — fast spatial filter against in-memory cache
        buildings = _BUILDINGS_CACHE.cx[minx:maxx, miny:maxy]
    else:
        # Serial / single-FID mode — bbox pushdown (requires GeoParquet
        # with bbox covering column; see data/all_buildings_*_bbox.parquet)
        buildings = gpd.read_parquet(
            buildings_path, columns=_BUILDINGS_COLUMNS,
            bbox=(minx, miny, maxx, maxy),
        )

    return buildings[buildings.within(buffered)]



def get_buildings_near_fid(
    network: MappedNetwork,
    fid: str,
    buffer_scale: float = 1.1,
    ) -> gpd.GeoDataFrame:
    """
    Get buildings within `buffer_m` of the building with TOID `fid`.

    Parameters
    ----------
    fid : str
        The TOID of the building to buffer around.
    buffer_m : float
        Buffer distance in metres.

    Returns
    -------
    GeoDataFrame in EPSG:27700 with columns: toid, activity, geometry
    """
    net = network.net

    if hasattr(net, "geometry"):
        geoms = net.geometry
    else:
        # assume a NetworkX graph with geometry attributes
        
        geoms = []
        for _, data in net.nodes(data=True):
            if "geometry" in data:
                geoms.append(loads(data["geometry"]))
        for _, _, data in net.edges(data=True):
            if "geometry" in data:
                geoms.append(loads(data["geometry"]))
        geoms = gpd.GeoSeries(geoms, crs="EPSG:27700")

    # Find the maximum and minimum x and y coordinates in the geometries
    combined = unary_union(geoms)
    outline = combined.convex_hull
    outline_buffered = outline.buffer(outline.length * (buffer_scale - 1) / (2 * 3.14159))

    # # 3️⃣ Quick plot to check
    # fig, ax = plt.subplots(figsize=(8, 8))
    # gpd.GeoSeries([outline_buffered]).plot(ax=ax, facecolor="none", edgecolor="red")
    # gpd.GeoSeries(geoms).plot(ax=ax, color="blue", markersize=5)
    # plt.show()
    

    # # Select buildings within the buffer
    building_geom = get_building_geom(outline_buffered)

    sel_3857 = building_geom.to_crs(3857)
    buffer_3857 = gpd.GeoSeries([outline_buffered], crs=27700).to_crs(3857)

    # fig, ax = plt.subplots(figsize=(8, 10))
    # if not sel_3857.empty:
    #     sel_3857.plot(ax=ax, linewidth=0.6, alpha=0.8, column="activity", legend=True)
    # buffer_3857.boundary.plot(ax=ax, linewidth=2, linestyle="--")

    # minx, miny, maxx, maxy = buffer_3857.total_bounds
    # pad = (maxx - minx) * 0.15
    # ax.set_xlim(minx - pad, maxx + pad)
    # ax.set_ylim(miny - pad, maxy + pad)

    # cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)

    # ax.set_title(f"SCUs near substation (OSM basemap)")
    # ax.axis("off")
    # plt.show()

    return sel_3857, buffer_3857



def match_buildings_to_network(network, building_geoms, buffer_dist=1.0, nearest_dist=10.0):
    """
    Match building geometries to service point nodes in the network.

    Parameters
    ----------
    network : MappedNetwork
        Network object with node geometries and asset_type attributes.
    building_geoms : GeoDataFrame
        Building geometries with valid CRS and (optionally) attributes like 'activity'.
    buffer_dist : float, optional
        Distance (m) to buffer nodes for the second matching pass.
    nearest_dist : float, optional
        Max distance (m) to consider for final nearest-neighbor matching.

    Returns
    -------
    final_nodes : DataFrame
        DataFrame with node_id, toid, match_method, dist_to_building, activity, premises_rows.
        This includes both matched and unmatched nodes, with unmatched nodes marked accordingly (to be designated as lamp posts later on)
    """
    # --- Extract service point nodes ---
    G = network.net
    node_records = []
    for n, data in G.nodes(data=True):
        geom = from_wkb(data.get("geometry"))
        if isinstance(geom, Point) and data.get("asset_type") == "Service Point":
            node_records.append({"node_id": n, "geometry": geom})

    gdf_nodes = gpd.GeoDataFrame(node_records, crs="EPSG:27700")
    gdf_nodes = gdf_nodes.to_crs(building_geoms.crs)

    # Ensure building geometries are polygons
    buildings = building_geoms.copy()
    buildings = buildings[buildings.geometry.notnull()].reset_index(drop=True)
    buildings["building_id"] = buildings.index

    # --- Test 1: Nodes overlapping buildings ---
    overlap_join = gpd.sjoin(buildings, gdf_nodes, predicate="intersects", how="left")
    mapped_overlap = overlap_join[~overlap_join["building_id"].isna()].copy()
    mapped_overlap["match_method"] = "overlap"

    mapped_nodes = mapped_overlap.drop_duplicates("node_id").copy()
    unmapped_nodes = gdf_nodes.loc[~gdf_nodes["node_id"].isin(mapped_nodes["node_id"])].copy()

    # filter out buildings already matched
    buildings = buildings[~buildings["building_id"].isin(mapped_nodes["building_id"])].copy()

    # --- Test 2: Buffered overlap (variable buffer distance) ---
    if not unmapped_nodes.empty:
        buffered = unmapped_nodes.copy()
        buffered["geometry"] = buffered.buffer(buffer_dist)

        buffer_join = gpd.sjoin(buffered, buildings, predicate="intersects", how="left")
        buffer_matches = buffer_join[~buffer_join["building_id"].isna()].copy()

        # Handle multiple matches — mark them for review
        single_match = buffer_matches.groupby("node_id").filter(lambda g: len(g) == 1).copy()
        single_match["match_method"] = "buffer_1m"

        mapped_nodes = pd.concat([mapped_nodes, single_match], ignore_index=True)
        unmapped_nodes = gdf_nodes.loc[~gdf_nodes["node_id"].isin(mapped_nodes["node_id"])].copy()

    ## Clean up mapped_nodes GeoDataFrame for saving/returning
    mapped_nodes_saving = mapped_nodes.copy()

    # Add unmapped node details 
    unmapped_nodes["toid"] = None
    unmapped_nodes["match_method"] = "unmatched"
    unmapped_nodes["dist_to_building"] = None
    unmapped_nodes["activity"] = None
    unmapped_nodes["premises_rows"] = None

    unmapped_nodes = unmapped_nodes[["node_id","toid", "match_method","dist_to_building", "activity","premises_rows"]].copy()

    # Join mapped and unmapped_nodes for final output
    final_nodes = pd.concat([mapped_nodes_saving, unmapped_nodes], ignore_index=True)

    # rename columns
    final_nodes = final_nodes.rename(columns={"node_id": "service_point_fid"})

    return final_nodes




def plot_network_on_map(network, substation_fid=None, building_geoms=None, crs="EPSG:4326"):
    """
    Interactive Plotly map of the LV network and related assets (uses go.Scattermap).
    """
    G = network.net
    fid = substation_fid if substation_fid is not None else "Network"

    # --- Extract geometries from network ---
    node_records, edge_records = [], []
    for n, data in G.nodes(data=True):
        geom = from_wkb(data.get("geometry"))
        if isinstance(geom, Point):
            node_records.append({"node": n,"node_type": data.get("asset_type") , "geometry": geom})
    for u, v, data in G.edges(data=True):
        geom = from_wkb(data.get("geometry"))
        if isinstance(geom, LineString):
            edge_records.append({"u": u, "v": v, "geometry": geom})

    gdf_nodes = gpd.GeoDataFrame(node_records, crs=crs)
    gdf_edges = gpd.GeoDataFrame(edge_records, crs=crs)

    # --- Ensure all geometries are lat/lon ---
    gdf_nodes = gdf_nodes.to_crs(epsg=4326)
    gdf_edges = gdf_edges.to_crs(epsg=4326)
    if building_geoms is not None and not building_geoms.empty:
        building_geoms = building_geoms.to_crs(epsg=4326)

    # --- Build edge traces ---
    edge_traces = []
    first_edge = True  # Flag to show legend only once

    for _, row in gdf_edges.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            lines = [geom]

        for line in lines:
            lons, lats = line.xy
            edge_traces.append(
                go.Scattermap(
                    lon=list(lons),
                    lat=list(lats),
                    mode="lines",
                    line=dict(width=2, color="blue"),
                    hoverinfo="none",
                    name="Lines",
                    showlegend=first_edge,  # Only show legend for first trace
                    legendgroup="edges"  # Group all edges together
                )
            )
            first_edge = False  # Set to False after first trace

    # --- Build node trace ---
    node_trace = go.Scattermap(
        lon=gdf_nodes.geometry.x,
        lat=gdf_nodes.geometry.y,
        mode="markers",
        marker=dict(size=8, color= "red"),
        text=gdf_nodes.apply(lambda row: f"Node: {row['node']}<br>Type: {row['node_type']}", axis=1),
        hoverinfo="text",
        name="Nodes"
    )

    # --- Add buildings (if provided) ---
    building_traces = []
    if building_geoms is not None and not building_geoms.empty:
        activity_colors = {
            "domestic": "green",
            "non-domestic": "orange",
            "no address": "purple",
            "mixed": "cyan",
            "unknown": "gray"
        }

        building_geoms["color"] = building_geoms["activity"].map(activity_colors).fillna("gray")

        for activity, color in activity_colors.items():
            activity_buildings = building_geoms[building_geoms["activity"] == activity]
            
            if activity_buildings.empty:
                continue
            
            lons_all = []
            lats_all = []
            hovertexts = []
            
            for idx, row in activity_buildings.iterrows():
                geom = row.geometry
                
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    lons, lats = zip(*coords)
                    lons_all.extend(list(lons) + [None])  # None creates a break
                    lats_all.extend(list(lats) + [None])
                    hovertexts.extend([row["premises_rows"]] + [None])
            
            if lons_all:
                building_traces.append(
                    go.Scattermap(
                        lon=lons_all,
                        lat=lats_all,
                        mode="lines",
                        line=dict(width=1, color="black"),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.8,
                        hovertext=hovertexts,
                        name=activity,
                        showlegend=True
                    )
                )

    # --- Combine all traces ---
    fig = go.Figure( building_traces + edge_traces + [node_trace])

    # --- Layout ---
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=gdf_nodes.geometry.y.mean(), lon=gdf_nodes.geometry.x.mean()),
            zoom=15
        ),
        title=f"Network Map: {fid}",
        legend=dict(title="Legend", bgcolor="rgba(255,255,255,0.8)")
    )

    fig.show()




# Example usage
# 11374925 - 503 service points with loads of no address
# 11375001 - 162 with good mix of domestic/non-domestic and (seemingly) correct activity
# 11375012 - 136 mostly domestic but with example of school building with no address
# 11375080 - large-ish network with loads of obviously domestic buildings as no address




def process_fid(fid, sql_fname="results/graph.sqlite"):
    """
    Process one substation FID: load network, match buildings, persist results
    to the building_matches table in graph.sqlite.

    Returns a dict with status and per-substation match summary stats.
    """
    try:
        time_start = time.time()

        network = MappedNetwork()
        network.load_from_sqlite(fid, sql_fname)

        if not network.service_points:
            return {
                "fid": fid, "status": "skipped",
                "reason": "no service points",
            }

        building_geoms, buffer_outline = get_buildings_near_fid(network, fid, buffer_scale=1.7)

        match_df = match_buildings_to_network(network, building_geoms)

        # --- Persist per-service-point matches to graph.sqlite ---
        save_cols = ["service_point_fid", "toid", "uprn",
                     "match_method", "dist_to_building", "activity_type"]
        save_cols = [c for c in save_cols if c in match_df.columns]
        out_df = match_df[save_cols].copy()
        out_df.insert(0, "substation_fid", fid)

        conn = sqlite3.connect(sql_fname, timeout=30)
        cur = conn.cursor()
        cols = ["service_point_fid", "substation_fid", "toid", "uprn",
                "match_method", "dist_to_building", "activity_type"]
        cols = [c for c in cols if c in out_df.columns]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        cur.executemany(
            f"INSERT OR REPLACE INTO building_matches ({col_names}) "
            f"VALUES ({placeholders})",
            out_df[cols].values.tolist(),
        )
        conn.commit()
        cur.close()
        conn.close()

        # --- Compute per-substation summary ---
        n_matched = int(match_df["toid"].notna().sum()) if "toid" in match_df.columns else 0
        n_total = len(match_df)
        n_overlap = int((match_df["match_method"] == "overlap").sum())
        match_rate = round(n_matched / n_total, 4) if n_total > 0 else 0.0
        pct_overlap = round(n_overlap / n_matched, 4) if n_matched > 0 else 0.0

        elapsed = time.time() - time_start
        return {
            "fid": fid,
            "status": "success",
            "time": elapsed,
            "buildings_matched": n_matched,
            "buildings_unmatched": n_total - n_matched,
            "match_rate": match_rate,
            "pct_overlap": pct_overlap,
        }

    except Exception as e:
        return {"fid": fid, "status": "failed", "error": str(e)}


def run_parallel(fids, sql_fname="results/graph.sqlite", max_workers=None,
                 buildings_path="data/all_buildings_in_enwl_27700_bbox.parquet"):
    """
    Run process_fid() across a list of FIDs in parallel with tqdm progress.

    Each worker loads the buildings parquet once at startup via _init_worker,
    then reuses the cached DataFrame for all substations it processes.
    """
    if max_workers is None:
        # Each worker loads ~3.5 GB (buildings parquet + spatial index).
        # Cap at 4 to stay within ~14 GB on a 32 GB machine.
        max_workers = min(4, max(1, os.cpu_count() - 5))

    results = []
    with mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(buildings_path,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(
                process_fid_wrapper, [(fid, sql_fname) for fid in fids]
            ),
            total=len(fids),
            desc=f"Building matching ({max_workers} workers)",
        ):
            results.append(result)
    return results


def process_fid_wrapper(args):
    """
    Wrapper to unpack args for multiprocessing (avoids lambda pickling issues).
    """
    fid, sql_fname = args
    return process_fid(fid, sql_fname=sql_fname)


def main():
    sql_fname = "results/graph.sqlite"

    # --- Load all mapped substation FIDs from graph.sqlite ---
    conn = sqlite3.connect(sql_fname, timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT substation_fid FROM mapped_substations")
    all_fids = [str(r[0]) for r in cur.fetchall()]

    # --- Resume: skip substations already in building_matches ---
    cur.execute("SELECT DISTINCT substation_fid FROM building_matches")
    done_fids = {str(r[0]) for r in cur.fetchall()}
    cur.close()
    conn.close()

    fids = [fid for fid in all_fids if fid not in done_fids]

    tqdm.write(f"Mapped substations: {len(all_fids)}  |  "
               f"Already matched: {len(done_fids)}  |  "
               f"Remaining: {len(fids)}")

    if not fids:
        tqdm.write("No substations to process!")
        return

    # --- Run parallel matching ---
    try:
        results = run_parallel(fids, sql_fname=sql_fname)
    except KeyboardInterrupt:
        tqdm.write("\nBuilding matching interrupted by user!")
        results = []

    # --- Save per-substation building match summary ---
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("results/building_match_summary.csv", index=False)
        tqdm.write(f"Building matching complete. {len(results)} processed. "
                   f"Summary saved to results/building_match_summary.csv")
    else:
        tqdm.write("No results to save.")


if __name__ == "__main__":
    # Windows needs this guard for multiprocessing to work properly
    # mp.freeze_support()
    main()

