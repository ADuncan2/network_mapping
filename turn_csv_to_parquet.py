
import pandas as pd
import geopandas as gpd
from shapely import wkb
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
csv_path = Path("data/uprn_all_buildings_in_enwl.csv")
out_parquet = Path("data/all_buildings_in_enwl_27700.parquet")


# 1. Read entire CSV
print("Reading CSV...")
df = pd.read_csv(csv_path)

# 2. Convert WKB hex to shapely geometry
print("Converting geometry...")
df["geometry"] = df["polygon"].apply(
    lambda x: wkb.loads(bytes.fromhex(x)) if pd.notna(x) else None
)

# 3. Drop raw polygon column
df = df.drop(columns=["polygon"])

# 4. Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs="EPSG:27700"  # change if needed
)

# 5. Write single GeoParquet file
print("Writing parquet...")
gdf.to_parquet(
    out_parquet,
    engine="pyarrow",
    compression="zstd"
)

print("✅ Done")

