import os
import pandas as pd
import geopandas as gpd
from shapely import from_wkb
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import contextily as cx
from typing import Union
from shapely.geometry import LineString, Point
from gridstock.network_parsing_EEA import DistributionNetwork
import networkx as nx
import sys
import time
from shapely.ops import unary_union
from shapely.wkb import loads
import plotly.graph_objects as go
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

def get_building_geom(buffered, buildings_path="data/enwl_buildings_activity.parquet", buffer_scale=1.1):
    """
    Given a network GeoDataFrame or NetworkX graph with geometry attributes,
    finds the outer boundary of the network, buffers it out by a percentage,
    and returns the subset of building geometries from a Parquet file that fall within it.
    
    Parameters
    ----------
    network : GeoDataFrame or networkx.Graph
        The mapped network. Must have a 'geometry' column or geometry attributes.
    buildings_path : str
        Path to the GeoParquet file containing building geometries.
    buffer_scale : float
        Scale factor to buffer the network area (1.1 = +10% area).
    
    Returns
    -------
    GeoDataFrame
        Subset of buildings within the buffered network area.
    """
    # --- 1️⃣ Get all geometries for this network ---
    
    
    # simpler (less precise but more intuitive):
    # buffered = outline.buffer(0.001)  # degrees or meters depending on CRS

    # --- 3️⃣ Read buildings only within bounding box of buffered area ---
    minx, miny, maxx, maxy = buffered.bounds
    buildings = gpd.read_parquet(buildings_path, columns=["toid", "activity", "premises_rows", "geometry"])
    buildings = buildings.cx[minx:maxx, miny:maxy]  # bbox crop

    # --- 4️⃣ Filter precisely to those within the buffer ---
    buildings_within = buildings[buildings.within(buffered)]

    # print(f"Loaded {len(buildings_within)} buildings within network area.")


    return buildings_within



def get_buildings_near_fid(
    network: DistributionNetwork,
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
    building_geom = get_building_geom(outline_buffered, buffer_scale=buffer_scale)

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



def match_buildings_to_network(network, building_geoms, buffer_dist=5.0, nearest_dist=10.0):
    """
    Match building geometries to service point nodes in the network.

    Parameters
    ----------
    network : DistributionNetwork
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
    overlap_join = gpd.sjoin(gdf_nodes, buildings, predicate="intersects", how="left")
    mapped_overlap = overlap_join[~overlap_join["building_id"].isna()].copy()
    mapped_overlap["match_method"] = "overlap"

    mapped_nodes = mapped_overlap.drop_duplicates("node_id").copy()
    unmapped_nodes = gdf_nodes.loc[~gdf_nodes["node_id"].isin(mapped_nodes["node_id"])].copy()

    # filter out buildings already matched
    buildings = buildings[~buildings["building_id"].isin(mapped_nodes["building_id"])].copy()

    # --- Test 2: Buffered overlap (1m buffer) ---
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

    # --- Test 3: Nearest building edge within threshold ---
    if not unmapped_nodes.empty:
        # Ensure both are in same CRS
        unmapped_nodes = unmapped_nodes.to_crs(buildings.crs)

        # Compute distance from each service point to the nearest building *edge*
        # shapely's .distance() already measures to polygon boundaries
        distances = []
        nearest_building_ids = []

        for _, node_row in unmapped_nodes.iterrows():
            node_point = node_row.geometry
            # Compute distances to all building edges
            building_distances = buildings.geometry.distance(node_point)
            min_idx = building_distances.idxmin()
            min_dist = building_distances[min_idx]

            distances.append(min_dist)
            nearest_building_ids.append(buildings.loc[min_idx, "building_id"])

        # Add distance results
        unmapped_nodes["nearest_building_id"] = nearest_building_ids
        unmapped_nodes["dist_to_building"] = distances

        # Filter matches within threshold (e.g. 10 m)
        nearest_match = unmapped_nodes[unmapped_nodes["dist_to_building"] <= nearest_dist].copy()
        nearest_match["match_method"] = "nearest_edge_within_10m"

        # Merge building attributes if desired
        nearest_match = nearest_match.merge(
            buildings[["building_id", "geometry"]],
            left_on="nearest_building_id",
            right_on="building_id",
            how="left",
            suffixes=("_node", "_building")
        )

        # Update mapped/unmapped
        mapped_nodes = pd.concat([mapped_nodes, nearest_match], ignore_index=True)
        unmapped_nodes = gdf_nodes.loc[~gdf_nodes["node_id"].isin(mapped_nodes["node_id"])].copy()

    # print(f"Matched {len(mapped_nodes)} / {len(gdf_nodes)} nodes.")
    # print(f"Remaining unmatched: {len(unmapped_nodes)}")

    # print("Match methods distribution:")
    # print(mapped_nodes["match_method"].value_counts())

    ## Clean up mapped_nodes GeoDataFrame for saving/returning
    mapped_nodes_saving = mapped_nodes[["node_id","toid", "match_method","dist_to_building", "activity","premises_rows"]].copy()

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

fid = 11378262
 
sql_fname = "data/graph.sqlite" 

network = DistributionNetwork() 
network.get_substation_networkx(fid, sql_fname) 
print("Substation networkx graph obtained.") 

building_geoms, buffer_outline = get_buildings_near_fid(network, fid, buffer_scale=1.7) 

# sp_to_building_match = match_buildings_to_network(network, building_geoms) 

# sp_to_building_match["substation_fid"] = fid sp_to_building_match.to_csv(f"data/service_point_to_building/sp_to_building_match_{fid}.csv", index=False) 


plot_network_on_map(network, fid,building_geoms, crs="EPSG:27700")