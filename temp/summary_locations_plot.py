import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import geopandas as gpd
from shapely.geometry import Point, box, shape
import fiona
import pyarrow







# --- File paths (adjust as needed) ---
boundary_fp = "data/enwl_control_boundary.geojson"
pop_density_fp = "data/pop_density_20211_2022.csv"  # columns: lsoa_id, population_density
lsoa_geojson_fp = "data/lsoa_shapes.geojson" 
summary_fp = "data/summary.csv"




## TO BE RUN ONCE TO CREATE PARQUET OF LSOA SHAPES IN TERRITORY
# # territory is a shapely Polygon defining your area of interest
# minx, miny, maxx, maxy = territory.total_bounds
# bbox = box(minx, miny, maxx, maxy)
# # Step 2: use Fiona to filter while reading
# # (this only parses features within the bounding box)

# with fiona.open(lsoa_geojson_fp, "r") as src:
#     filtered_features = [
#         feat for feat in src.filter(bbox=(minx, miny, maxx, maxy))
#     ]

# # Step 3: convert only those filtered features into a GeoDataFrame
# lsoa_subset = gpd.GeoDataFrame.from_features(filtered_features, crs=src.crs)

# print(f"Loaded {len(lsoa_subset)} features within bounding box")

# lsoa_subset = lsoa_subset[lsoa_subset.intersects(territory.unary_union)]

# print(f"LSOAs intersecting ENWL territory: {len(lsoa_subset)}")

# lsoa_subset.to_parquet("data/lsoa_subset.parquet")


# # --- Load substation geometry from ENWL original data ---
# def extract_all_substations(gpkg_file='data/assets.gpkg', layer_name='General Boundary'):
#     """
#     Extracts FIDs and centroid coordinates from the specified Geopackage layer,
#     reprojected to EPSG:4326 for lat/lon compatibility.
    
#     Returns:
#         pd.DataFrame: DataFrame with columns ['FID', 'latitude', 'longitude']
#     """
#     # Read the layer as a GeoDataFrame
#     gdf = gpd.read_file(gpkg_file, layer=layer_name)
    
#     # Set CRS explicitly if missing, then reproject
#     if gdf.crs is None:
#         gdf.set_crs(epsg=27700, inplace=True)
#     gdf = gdf.to_crs(epsg=4326)

#     # Calculate centroids in EPSG:4326
#     gdf['centroid'] = gdf.geometry.centroid

#     # Extract coordinates
#     gdf['latitude'] = gdf.centroid.x
#     gdf['longitude'] = gdf.centroid.y

#     gdf["geometry"] = gpd.points_from_xy( gdf["latitude"], gdf["longitude"])

#     # Keep only relevant columns
#     df = gdf[['FID', 'longitude','latitude', 'geometry']].copy()

#     return df

# df = extract_all_substations()
# # remove duplicates fids if any
# df = df.drop_duplicates(subset="FID")

# print(f"Extracted {len(df)} substations from Geopackage")

# df.to_parquet('data/substation_centroids.parquet', index=False)


## MAIN ANALYSIS BELOW

# --- Load ENWL boundary ---
territory = gpd.read_file(boundary_fp)
territory = territory.to_crs(epsg=4326)  # ensure WGS84


# --- Load all substation centroids ---
all_sub_centroids = gpd.read_parquet("data/substation_centroids.parquet")
all_sub_centroids = all_sub_centroids.set_crs(epsg=4326)
print(f"Total substations loaded: {len(all_sub_centroids)}")

# --- Load summary data ---
mapped_subs = pd.read_csv(summary_fp)
# Ensure unique FIDs
mapped_subs = mapped_subs.drop_duplicates(subset="substation_fid")

# Extract long/lat from substation_geom
# Format looks like "(long, lat)"
def parse_geom(geom_str):
    match = re.match(r"\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?", str(geom_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None
mapped_subs[["longitude", "latitude"]] = mapped_subs["substation_geom"].apply(lambda g: pd.Series(parse_geom(g)))
# Drop rows where parsing failed
mapped_subs = mapped_subs.dropna(subset=["longitude", "latitude"])
# convert to geodataframe
mapped_subs = gpd.GeoDataFrame(
    mapped_subs,
    geometry= gpd.points_from_xy(mapped_subs["latitude"], mapped_subs["longitude"]),
    crs="EPSG:4326"
)
print(f"Number of substations after deduplication and cleaning: {len(mapped_subs)}")

# Load lsoa subset from parquet
lsoa_subset = gpd.read_parquet("data/lsoa_subset.parquet")
lsoa_subset = lsoa_subset.set_crs(epsg=4326)  # set to BNG
lsoa_subset = lsoa_subset.rename(columns={"LSOA21CD":"geo_code"})
print(f"LSOAs in ENWL territory: {len(lsoa_subset)}")
# print(lsoa_subset.head())



# --- Load population density ---
pop_density = pd.read_csv(pop_density_fp,skiprows=3)
pop_density["geo_code"] = pop_density["LSOA 2021 Code"]
pop_density["people_per_km_22"] = pop_density["Mid-2022: People per Sq Km"]
pop_density = pop_density.loc[:,["geo_code","people_per_km_22"]]



### Replace the geometry of mapped_subs with geometry from all_sub_centroids based on substation_fid and FID
mapped_subs = mapped_subs.merge(
    all_sub_centroids[["FID", "geometry"]],
    left_on="substation_fid",
    right_on="FID",
    how="left",
    suffixes=("", "_from_all")
)
# Replace geometry if available
mapped_subs["geometry"] = mapped_subs["geometry_from_all"].combine_first(mapped_subs["geometry"])
mapped_subs = mapped_subs.drop(columns=["FID", "geometry_from_all"])




## FIND SUBSTATIONS PER LSOA (points in polygon)
# Perform the spatial join
gdf_joined = gpd.sjoin(
    mapped_subs,
    lsoa_subset[["geo_code", "RUC21NM", "Urban_rura", "geometry"]],
    how="left",
    predicate="within"  # ensures point-in-polygon matching
)

print(f"Number of mapped_subs after spatial join: {len(gdf_joined)}")
# --- Population density points prep ---
# First make sure the population density column is numeric
pop_density["people_per_km_22"] = (
    pop_density["people_per_km_22"]
    .astype(str).str.replace(",", "").astype(float)
)

# Join population density data
gdf_joined = gdf_joined.merge(
    pop_density,
    on="geo_code",
    how="left"
)

# Save results to CSV
gdf_joined.to_csv("data/substations_with_lsoa.csv", index=False)


# --- Aggregate substation counts by LSOA ---
df_lsoa_summary = (
    gdf_joined
    .groupby(["geo_code", "RUC21NM", "Urban_rura"], as_index=False)
    .agg(substations_per_lsoa=("substation_fid", "count"),
        total_service_points=("service_points", "sum"),
         people_per_km_22=("people_per_km_22", "first"))  # assuming constant per LSOA
)

print(f"Unique LSOAs with mapped_subs in them: {len(df_lsoa_summary)}")

# Find all substations in each LSOA using all_substations_centroids

# FOR CHECKING THE MAPPING PROCESS Filter all_subs_by_lsoa to only those in df_lsoa_summary
# all_sub_centroids = all_sub_centroids[all_sub_centroids["FID"].isin(mapped_subs["substation_fid"])]

# Perform the spatial join
all_subs_joined = gpd.sjoin(
    all_sub_centroids,
    lsoa_subset[["geo_code", "RUC21NM", "Urban_rura", "geometry"]],
    how="left",
    predicate="within"  # ensures point-in-polygon matching
)

print(f"Number of all_subs after spatial join: {len(all_subs_joined)}")

# --- Aggregate substation counts by LSOA ---
all_subs_by_lsoa = (
    all_subs_joined
    .groupby(["geo_code", "RUC21NM", "Urban_rura"], as_index=False)
    .agg(substations_per_lsoa=("FID", "count"))
)

all_subs_by_lsoa = all_subs_by_lsoa.rename(columns={"substations_per_lsoa":"all_substations_per_lsoa"})
all_subs_by_lsoa = all_subs_by_lsoa.loc[:,["geo_code","all_substations_per_lsoa"]]

lsoa_with_all_subs = df_lsoa_summary.merge(
    all_subs_by_lsoa,
    on="geo_code",
    how="left"
).fillna({"substations_per_lsoa":0})

# Calculate percentage mapped
lsoa_with_all_subs["percent_mapped"] = round((
    lsoa_with_all_subs["substations_per_lsoa"] / lsoa_with_all_subs["all_substations_per_lsoa"]
).fillna(0) * 100, 1)

print("LSOAs with all substations counts:")
print(lsoa_with_all_subs.head())

# print(f"Total of all_subs: {lsoa_with_all_subs['all_substations_per_lsoa'].sum()}")
# print(f"Total of mapped_subs: {lsoa_with_all_subs['substations_per_lsoa'].sum()}")


# ### Examining one geo_code in detail E01004770 ###
# example_lsoa = ["E01004770", "E01004773", "E01004810", "E01004767", "E01004806"]
# all_subs_example = all_subs_joined[all_subs_joined["geo_code"].isin(example_lsoa)]
# mapped_subs_example = gdf_joined[gdf_joined["geo_code"].isin(example_lsoa)]
# print(all_subs_example)
# print("Mapped Substations in this LSOA:")
# print(mapped_subs_example)



# # ### PLOTS ###
# # Distribution of percentage mapped
# plt.figure(figsize=(8,6))
# sns.histplot(lsoa_with_all_subs["percent_mapped"], bins=50, kde=False)
# plt.xlabel("Percentage of Substations Mapped (%)")
# plt.ylabel("Number of LSOAs")
# plt.title("Distribution of Substations Mapped per LSOA")
# plt.tight_layout()
# plt.show()

## Plotting maps of substations

# # --- Compute summary statistics ---
# def get_summary(gdf, label):
#     counts = gdf["Urban_rura"].value_counts(dropna=False)
#     perc = gdf["Urban_rura"].value_counts(normalize=True, dropna=False) * 100
#     summary = "\n".join([f"{k}: {counts[k]} ({perc[k]:.1f}%)" for k in counts.index])
#     return f"{label}\n{summary}"

# summary_all = get_summary(all_subs_joined, "All Substations")
# summary_mapped = get_summary(gdf_joined, "Mapped Substations")

# # --- Define consistent colours for Rural/Urban ---
# colors = {"Urban": "royalblue", "Rural": "forestgreen"}

# # --- Create figure and subplots ---
# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# # --- Plot all substations ---
# territory.boundary.plot(ax=axes[0], color="black", linewidth=1)
# all_subs_joined.plot(ax=axes[0], column="Urban_rura", categorical=True,
#                      legend=True, markersize=10, alpha=0.7, c=all_subs_joined["Urban_rura"].map(colors))
# axes[0].set_title("All Substations in Dataset")

# # --- Plot mapped substations ---
# territory.boundary.plot(ax=axes[1], color="black", linewidth=1)
# gdf_joined.plot(ax=axes[1], column="Urban_rura", categorical=True,
#                 legend=True, markersize=10, alpha=0.7, c=gdf_joined["Urban_rura"].map(colors))
# axes[1].set_title("Mapped Substations")

# # --- Add summary text ---
# axes[0].text(0.02, 0.02, summary_all, transform=axes[0].transAxes,
#              fontsize=9, verticalalignment="bottom", bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
# axes[1].text(0.02, 0.02, summary_mapped, transform=axes[1].transAxes,
#              fontsize=9, verticalalignment="bottom", bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# # --- Clean up axes ---
# for ax in axes:
#     ax.set_axis_off()
#     ax.set_aspect('equal')

# plt.tight_layout()
# plt.show()


# # --- 2. Plot 1: substations per LSOA ---
# plt.figure(figsize=(8,6))
# plt.scatter(
#     df_lsoa_summary["people_per_km_22"],
#     df_lsoa_summary["substations_per_lsoa"],
#     c=pd.Categorical(df_lsoa_summary["Urban_rura"]).codes,
#     cmap="tab10", alpha=0.7, edgecolor="k", linewidth=0.3
# )
# plt.xlabel("People per km² (LSOA)")
# plt.ylabel("Substations per LSOA")
# plt.title("Substations vs Population Density by LSOA")
# plt.legend(handles=[
#     plt.Line2D([0], [0], marker='o', color='w', label='Urban', markerfacecolor='C0', markersize=10, markeredgecolor='k'),
#     plt.Line2D([0], [0], marker='o', color='w', label='Rural', markerfacecolor='C1', markersize=10, markeredgecolor='k')
# ], title="Urban/Rural", loc='upper left')
# plt.tight_layout()
# plt.show()

# --- 3. Plot 2: total service points ---
plt.figure(figsize=(8,6))
plt.scatter(
    df_lsoa_summary["people_per_km_22"],
    df_lsoa_summary["total_service_points"],
    c=pd.Categorical(df_lsoa_summary["Urban_rura"]).codes,
    cmap="tab10", alpha=0.7, edgecolor="k", linewidth=0.3
)
plt.xlabel("People per km² (LSOA)")
plt.ylabel("Total Service Points per LSOA")
plt.title("Service Points vs Population Density by LSOA")
plt.tight_layout()
plt.show()

# ## Comparing Urban/Rural proportions
# # --- 1. Count proportions in both datasets ---
# total_counts = all_subs_joined["Urban_rura"].value_counts(normalize=True) * 100
# mapped_counts = gdf_joined["Urban_rura"].value_counts(normalize=True) * 100

# # --- 2. Combine into a single DataFrame for plotting ---
# compare_df = pd.DataFrame({
#     "All substations": total_counts,
#     "Mapped substations": mapped_counts
# }).fillna(0)

# print(compare_df)

# # --- 3. Plot side-by-side bars ---
# compare_df.plot(kind="bar", figsize=(8,6))
# plt.ylabel("Percentage of substations (%)")
# plt.title("Urban vs Rural: All vs mapped Substations")
# plt.xticks(rotation=0)
# plt.legend(title="Category")
# plt.tight_layout()
# plt.show()