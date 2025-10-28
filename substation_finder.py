from shapely.geometry import Point, LineString, shape, Polygon
import sqlite3
from shapely.wkb import loads, dumps
from pyproj import Transformer

def get_substation_details(fid):
    # Get contextual data for the substation
    with sqlite3.connect(r"data\lv_assets.sqlite") as lv_con:
        lv_curs = lv_con.cursor()
        lv_curs.execute(
            "SELECT Geometry, Asset_Type, Voltage, Installation_Date FROM lv_assets WHERE fid = ?",
            (fid,)
        )
        sub_data = lv_curs.fetchone()

    if not sub_data:
        raise ValueError(f"No substation data found for fid {fid}")

    # Convert geometry polygon to centroid point
    sub_point = loads(sub_data[0]).centroid
    sub_point_wkb = dumps(sub_point)
    sub_data = (sub_point_wkb,) + sub_data[1:]

    # Transform to lat/long
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4277", always_xy=True)
    longitude, latitude = transformer.transform(sub_point.x, sub_point.y)

    # Find connected edges in separate DB
    with sqlite3.connect("data/network_data.sqlite") as connection_net:
        cursor_net = connection_net.cursor()
        cursor_net.execute(
            """
            SELECT fid_to, fid_from 
            FROM conn_comp 
            WHERE fid_from = ? OR fid_to = ?
            """,
            (fid, fid)
        )
        rows = cursor_net.fetchall()

    # Return results as a dict for easy access
    return {
        "latitude": latitude,
        "longitude": longitude,
        "asset_type": sub_data[1],
        "voltage": sub_data[2],
        "installation_date": sub_data[3],
        "connected_objects": rows
    }

# Example usage
fid = 11390319
substation_info = get_substation_details(fid)
print(f"Coordinates: {substation_info['latitude']}, {substation_info['longitude']}")
print(f"Connected objects: {substation_info['connected_objects']}")
print(f"Install date: {substation_info['installation_date']}")