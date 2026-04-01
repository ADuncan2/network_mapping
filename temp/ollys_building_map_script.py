import os
import pandas as pd
import geopandas as gpd
from shapely import from_wkb
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import contextily as cx
from typing import Union

def select_scu_within(
    geom: Union[object, "shapely.geometry.base.BaseGeometry"],
    geom_crs: Union[int, str] = 27700,
    csv_gz_path: str = os.path.join("data", "enwl_buildings_activity.csv.gz"),
    chunksize: int = 150_000,
    ) -> gpd.GeoDataFrame:
    """
    Stream-select SCUs whose polygons are fully WITHIN `geom`.

    Expects CSV columns:
      ['scu_id','storeys_above','n_uprns','sum_commercial_count','sum_dom_count',
       'sum_other_count','usage_type','geom_wkb_hex']
    Returns a GeoDataFrame in EPSG:27700 with the same columns + geometry.
    """
    # Ensure the query geometry is in EPSG:27700
    q = gpd.GeoSeries([geom], crs=geom_crs).to_crs(27700).iloc[0]
    qminx, qminy, qmaxx, qmaxy = q.bounds

    hits = []
    for df in pd.read_csv(csv_gz_path, chunksize=chunksize):
        if "geom_wkb_hex" not in df.columns:
            raise RuntimeError("CSV is missing 'geom_wkb_hex' column.")
        geoms = [(from_wkb(bytes.fromhex(h)) if isinstance(h, str) and h else None) for h in df["geom_wkb_hex"]]
        gdf = gpd.GeoDataFrame(df.drop(columns=["geom_wkb_hex"]), geometry=geoms, crs=27700).dropna(subset=["geometry"])
        if gdf.empty:
            continue

        # bbox prefilter
        b = gdf.bounds
        mask = (b["minx"] <= qmaxx) & (b["maxx"] >= qminx) & (b["miny"] <= qmaxy) & (b["maxy"] >= qminy)
        cand = gdf.loc[mask]
        if cand.empty:
            continue

        # Exact within
        sel = cand[cand.within(q)]
        if not sel.empty:
            hits.append(sel)

    if hits:
        return gpd.GeoDataFrame(pd.concat(hits, ignore_index=True), crs=27700)
    else:
        cols = ["scu_id","storeys_above","n_uprns","sum_commercial_count","sum_dom_count","sum_other_count","usage_type","geometry"]
        return gpd.GeoDataFrame(columns=cols, crs=27700)



# Ambleside centre (approx, WGS84 lon/lat)
AMBLESIDE_LON, AMBLESIDE_LAT = -2.884531, 54.062086
RADIUS_M = 500

centre_bng = gpd.GeoSeries([Point(AMBLESIDE_LON, AMBLESIDE_LAT)], crs=4326).to_crs(27700).iloc[0]
query_buffer_bng = centre_bng.buffer(RADIUS_M)

sel = select_scu_within(query_buffer_bng, chunksize=150_000)

sel_3857 = sel.to_crs(3857)
buffer_3857 = gpd.GeoSeries([query_buffer_bng], crs=27700).to_crs(3857)

fig, ax = plt.subplots(figsize=(8, 10))
if not sel_3857.empty:
    sel_3857.plot(ax=ax, linewidth=0.6, alpha=0.8, column="activity", legend=True)
buffer_3857.boundary.plot(ax=ax, linewidth=2, linestyle="--")

minx, miny, maxx, maxy = buffer_3857.total_bounds
pad = (maxx - minx) * 0.15
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)

cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)

ax.set_title(f"SCUs within {RADIUS_M} m of Ambleside centre (OSM basemap)")
ax.axis("off")
plt.show()
