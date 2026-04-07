"""
Database preprocessing functions for the Gridstock network mapping pipeline.

Preprocessing pipeline order:
1. create_lv_db()              — Import 15 layers from assets.gpkg
2. create_conn_comp()          — Fresh copy of conn with indexes
3. merge_service_points()      — Collapse collocated service points/joints
4. simplify_substations()      — Collapse internal substation nodes to boundary FID
5. simplify_pmt_substations()  — Create 5m buffer circles for PMTs, find flux lines
6. enrich_substations()        — Spatial join: transformer capacity → General Boundary rows
7. copy_conn_to_lv()           — Copy conn_comp into lv_assets.sqlite
8. create_graph_db()           — Create empty output database
"""

import sqlite3
import fiona
from shapely.geometry import (
    shape,
    Polygon,
    Point
)
from shapely.affinity import translate, rotate
from shapely import wkb, STRtree
import os
import re
from datetime import datetime
import time
import pandas as pd

LINE_CLEAR = '\x1b[2K'


# ──────────────────────────────────────────────
# Preprocessing report
# ──────────────────────────────────────────────

class PreprocessingReport:
    """Collects metrics from each preprocessing step for a final report."""

    def __init__(self):
        self.start_time = time.time()
        self.steps = {}
        self._current_step = None

    def start_step(self, name):
        self.steps[name] = {"start": time.time(), "metrics": {}}
        self._current_step = name

    def add_metric(self, step, key, value):
        if step not in self.steps:
            self.steps[step] = {"start": time.time(), "metrics": {}}
        self.steps[step]["metrics"][key] = value

    def end_step(self, name):
        if name in self.steps:
            self.steps[name]["duration"] = time.time() - self.steps[name]["start"]

    def write(self, path="data/cleaning_report.txt"):
        total_duration = time.time() - self.start_time
        lines = [
            "=== Preprocessing Report ===",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {total_duration / 60:.1f} minutes",
            "",
        ]
        for step_name, step_data in self.steps.items():
            duration = step_data.get("duration", 0)
            lines.append(f"--- {step_name} ({duration:.1f}s) ---")
            for key, value in step_data.get("metrics", {}).items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        report_text = "\n".join(lines)
        with open(path, "w") as f:
            f.write(report_text)
        print(report_text)
        return report_text


# ──────────────────────────────────────────────
# Step 1: create_lv_db
# ──────────────────────────────────────────────

def create_lv_db(report=None) -> None:
    """
    Create lv_assets.sqlite from 15 layers of assets.gpkg.

    Layers: 12 original + Transformer, PMT Transformer, Substation.
    Fixes the Way Normal Status gap (tries 'Normal Status' after
    'Open or Closed' KeyError).  Adds Rating_Normal, Variable_Rating,
    Substation_Type, Infeed_Voltage columns.
    """
    step = "create_lv_db"
    if report:
        report.start_step(step)

    layer_names = [
        "LV Board",
        "LV Conductor",
        "LV Joint",
        "LV Service",
        "Way",
        "Leader Line",
        "Service Point",
        "Wall Termination",
        "Ground Line",
        "Ground Point",
        "Switch",
        "General Boundary",
        "Transformer",
        "PMT Transformer",
        "Substation",
    ]

    # Create the new database
    file_path = "data/lv_assets.sqlite"
    if os.path.exists(file_path):
        os.remove(file_path)
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lv_assets (
            fid INTEGER PRIMARY KEY,
            Geometry BLOB NULL,
            Asset_Type TEXT NULL,
            Voltage INTEGER NULL,
            Material TEXT NULL,
            Conductors_Per_Phase INTEGER NULL,
            Cable_Size TEXT NULL,
            Insulation TEXT NULL,
            Installation_Date DATETIME NULL,
            Phases_Connected TEXT NULL,
            Sleeve_Type TEXT NULL,
            Associated_Cable TEXT NULL,
            Switch_Status TEXT NULL,
            Is_Substation INTEGER,
            Is_Switch INTEGER,
            Rating_Normal TEXT NULL,
            Variable_Rating TEXT NULL,
            Substation_Type TEXT NULL,
            Infeed_Voltage TEXT NULL
        )
    """)

    # Error tracking
    layer_counts = {}
    fid_collisions = 0
    voltage_parse_failures = 0
    date_parse_failures = 0
    way_status_ambiguous = 0
    switch_status_null = 0
    transformer_rating_undefined = 0

    for i, layer in enumerate(layer_names):
        count = 0
        print(f"Processing layer {i + 1}/{len(layer_names)}: {layer}...")
        with fiona.open('data/assets.gpkg', layer=layer) as src:
            for feature in src:
                fid = feature['properties']["FID"]

                # Geometry → WKB
                geometry = feature['geometry']
                wkb_geom = wkb.dumps(shape(geometry))

                # Asset type — fall back to layer name if missing
                try:
                    asset_type = str(feature["properties"]["asset_type"])
                    if not asset_type or asset_type == "None":
                        asset_type = layer
                except KeyError:
                    asset_type = layer

                # Material (note trailing space in column name)
                try:
                    material = str(feature["properties"]["Material "])
                except KeyError:
                    material = ""

                # Conductors per phase
                try:
                    conductors_per_phase = int(feature["properties"]["Conductors_Per_Phase"])
                except (TypeError, KeyError):
                    conductors_per_phase = ""

                # Cable size
                try:
                    cable_size = str(feature["properties"]["Cable Size"])
                except KeyError:
                    cable_size = ""

                # Insulation
                try:
                    insulation = str(feature["properties"]["Insulation"])
                except KeyError:
                    insulation = ""

                # Phases connected
                try:
                    phases_connected = str(feature["properties"]["Phases Connected"])
                except KeyError:
                    phases_connected = ""

                # Sleeve type
                try:
                    sleeve_type = str(feature["properties"]["Sleeve Type"])
                except KeyError:
                    sleeve_type = ""

                # Associated cable
                try:
                    associated_cable = str(feature["properties"]["Associated Cable"])
                except KeyError:
                    associated_cable = ""

                # Switch status — Way Normal Status fix
                try:
                    switch_status = str(feature["properties"]["Open or Closed"])
                except KeyError:
                    try:
                        switch_status = str(feature["properties"]["Normal Status"])
                    except KeyError:
                        switch_status = ""
                # Normalise "None" string to empty
                if switch_status == "None":
                    switch_status = ""

                # Track ambiguous statuses
                if layer == "Way" and switch_status in ("", "Not Defined"):
                    way_status_ambiguous += 1
                if layer == "Switch" and switch_status == "":
                    switch_status_null += 1

                # Flags
                is_switch = 1 if layer == "Switch" else 0
                is_substation = 1 if layer == "General Boundary" else 0

                # Installation date
                try:
                    install_date = str(feature["properties"]["Installation Date"])
                except KeyError:
                    install_date = ""
                if install_date and install_date != "None":
                    try:
                        parsed = datetime.strptime(install_date, "%d/%m/%Y")
                        install_date = parsed.strftime("%Y-%m-%d")
                        year = int(install_date[:4])
                        if year < 1900 or year > 2026:
                            install_date = ""
                            date_parse_failures += 1
                    except ValueError:
                        install_date = ""
                        date_parse_failures += 1
                else:
                    install_date = ""

                # Voltage — numeric conversion
                try:
                    voltage = str(feature["properties"]["Nominal Voltage"])
                except KeyError:
                    try:
                        voltage = str(feature["properties"]["Voltage"])
                    except KeyError:
                        voltage = ""
                if voltage and voltage != "None":
                    match = re.match(r"([\d.]+)(\w+)", voltage)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)
                        if unit == 'kV':
                            value *= 1000
                        elif unit == 'MV':
                            value *= 1000000
                        voltage = int(value)
                    else:
                        voltage = ""
                        voltage_parse_failures += 1
                else:
                    voltage = ""

                # --- New columns for transformers / substations ---

                # Rating Normal
                try:
                    rating_normal = str(feature["properties"]["Rating Normal"])
                except KeyError:
                    rating_normal = ""
                if rating_normal == "None":
                    rating_normal = ""
                if layer in ("Transformer", "PMT Transformer") and rating_normal in ("", "Not Defined"):
                    transformer_rating_undefined += 1

                # Variable Rating
                try:
                    variable_rating = str(feature["properties"]["Variable Rating"])
                except KeyError:
                    variable_rating = ""
                if variable_rating == "None":
                    variable_rating = ""

                # Substation Type
                try:
                    substation_type = str(feature["properties"]["Substation Type"])
                except KeyError:
                    substation_type = ""
                if substation_type == "None":
                    substation_type = ""

                # Infeed Voltage (stored as text, not converted)
                try:
                    infeed_voltage = str(feature["properties"]["Infeed Voltage"])
                except KeyError:
                    infeed_voltage = ""
                if infeed_voltage == "None":
                    infeed_voltage = ""

                # Insert row
                try:
                    cursor.execute(
                        'INSERT INTO lv_assets VALUES '
                        '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (fid, wkb_geom, asset_type, voltage, material,
                         conductors_per_phase, cable_size, insulation,
                         install_date, phases_connected, sleeve_type,
                         associated_cable, switch_status,
                         is_substation, is_switch,
                         rating_normal, variable_rating,
                         substation_type, infeed_voltage)
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    fid_collisions += 1

        layer_counts[layer] = count
        connection.commit()
        print(f"  -> {count} rows")

    # Classify assets as node or edge
    cursor.execute("ALTER TABLE lv_assets ADD COLUMN cat TEXT")
    cursor.execute("""
        UPDATE lv_assets SET cat = CASE
            WHEN Asset_Type IN ('LV Conductor', 'LV Service', 'Leader Line', 'Ground Line')
                THEN 'edge'
            WHEN Asset_Type IN ('LV Board', 'LV Joint', 'Way', 'Service Point',
                                'Wall Termination', 'Ground Point', 'Switch',
                                'General Boundary', 'Transformer', 'PMT Transformer',
                                'Substation')
                THEN 'node'
            ELSE NULL
        END
    """)

    connection.commit()
    total = sum(layer_counts.values())
    print(f"Total rows: {total}")

    if report:
        report.add_metric(step, "Layers imported", len(layer_names))
        report.add_metric(step, "Total rows", total)
        for lyr, cnt in layer_counts.items():
            report.add_metric(step, f"  {lyr}", cnt)
        report.add_metric(step, "FID collisions", fid_collisions)
        report.add_metric(step, "Voltage parse failures", voltage_parse_failures)
        report.add_metric(step, "Date parse failures", date_parse_failures)
        report.add_metric(
            step,
            f"Way status ambiguous (of {layer_counts.get('Way', 0)})",
            way_status_ambiguous,
        )
        report.add_metric(
            step,
            f"Switch status null (of {layer_counts.get('Switch', 0)})",
            switch_status_null,
        )
        report.add_metric(
            step,
            f"Transformer rating undefined (of {layer_counts.get('Transformer', 0) + layer_counts.get('PMT Transformer', 0)})",
            transformer_rating_undefined,
        )
        report.end_step(step)

    cursor.close()
    connection.close()


# ──────────────────────────────────────────────
# Step 2: create_conn_comp
# ──────────────────────────────────────────────

def create_conn_comp(report=None) -> None:
    """
    Create conn_comp as a fresh indexed copy of conn in network_data.sqlite.
    Replaces both the lost original creation and creating_conn_new().
    """
    step = "create_conn_comp"
    if report:
        report.start_step(step)

    db_path = "data/network_data.sqlite"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS conn_comp")
    cursor.execute("CREATE TABLE conn_comp AS SELECT * FROM conn")

    cursor.execute("SELECT COUNT(*) FROM conn")
    conn_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM conn_comp")
    comp_count = cursor.fetchone()[0]

    print(f"conn_comp created: {comp_count} rows (conn has {conn_count})")

    print("Creating indexes...")
    cursor.execute("CREATE INDEX idx_cc_fid_from ON conn_comp(fid_from)")
    cursor.execute("CREATE INDEX idx_cc_fid_to ON conn_comp(fid_to)")

    connection.commit()

    if report:
        report.add_metric(step, "Rows", comp_count)
        report.add_metric(
            step, "Matches conn",
            "YES" if comp_count == conn_count else f"NO (conn={conn_count})",
        )
        report.end_step(step)

    cursor.close()
    connection.close()


# ──────────────────────────────────────────────
# Flux lines table helpers
# ──────────────────────────────────────────────

def create_station_flux_lines_table(flux_db_path) -> None:
    """Create an empty flux_lines table at the given path."""
    connection = sqlite3.connect(flux_db_path)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flux_lines (
            fid INTEGER PRIMARY KEY,
            Visited INTEGER
        )
    """)
    cursor.close()
    connection.commit()
    connection.close()


def reset_station_flux_lines_table(flux_db_path) -> None:
    """Reset all Visited flags to 0 in the flux_lines table."""
    connection = sqlite3.connect(flux_db_path)
    cursor = connection.cursor()
    cursor.execute("UPDATE flux_lines SET Visited = 0")
    cursor.close()
    connection.commit()
    connection.close()


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────

def square_from_point(
        centroid: Point,
        theta: float,
        side_length: float = 1.5
    ) -> Polygon:

    half_side_length = side_length / 2
    square = Polygon([
        (0, 0),
        (side_length, 0),
        (side_length, side_length),
        (0, side_length)
    ])
    square = translate(square, -half_side_length, -half_side_length)
    rotated_square = rotate(square, theta, origin=(0, 0))
    return translate(rotated_square, centroid.x, centroid.y)


def get_things_in_vicinity(
        centroid: Point,
        buffer_radius: int | float = 20
        ) -> list:
    """
    Get all low voltage assets within buffer_radius of the centroid,
    querying assets.gpkg (which has spatial indexing).
    """
    layer_names = [
        "LV Board",
        "LV Conductor",
        "LV Joint",
        "LV Service",
        "Way",
        "Leader Line",
        "Service Point",
        "Switch"
    ]

    substation_centroid_point = Point(centroid)
    buffered_area = substation_centroid_point.buffer(buffer_radius)

    features_in_vicinity = []
    for layer in layer_names:
        with fiona.open("data/assets.gpkg", "r", layer=layer) as src:
            for feature in src.filter(bbox=buffered_area.bounds):
                features_in_vicinity.append(feature)

    return features_in_vicinity


def _get_nearby_features_from_sources(centroid_coords, layer_sources, buffer_radius=10):
    """
    Find features near centroid_coords using pre-opened fiona layer sources.
    Faster than get_things_in_vicinity when called many times (avoids
    repeated fiona.open overhead).
    """
    point = Point(centroid_coords)
    buffered = point.buffer(buffer_radius)
    bbox = buffered.bounds

    features = []
    for src in layer_sources.values():
        for feat in src.filter(bbox=bbox):
            features.append(feat)
    return features


# ──────────────────────────────────────────────
# Step 3: merge_service_points (unchanged)
# ──────────────────────────────────────────────

def merge_service_points(
        fname: str,
        report=None
    ):
    """
    Simplify collocated service points and joints into single nodes
    by replacing joint FIDs with the nearby service point FID in conn_comp.
    """
    step = "merge_service_points"
    if report:
        report.start_step(step)

    # Get the service point geometries
    services = []
    with fiona.open(fname, layer='Service Point') as src:
        for i, feature in enumerate(src):
            print(100 * i / len(src), "%", end='\r')
            print(end=LINE_CLEAR)

            centroid = shape(feature['geometry'])
            box = square_from_point(centroid, 0.0, side_length=0.3)
            results = [
                str(feature['properties']['FID']),
                box
                ]
            services.append(results)

    # Open the SQLite database connection
    database_path = 'data/network_data.sqlite'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    rows_modified = 0

    # Find collocated LV Joints and replace with service point FID
    with fiona.open(fname, layer='LV Joint') as src2:

        ll = len(services)
        for i, service in enumerate(services):

            print(f"{i} of service {ll}...", end='\r')
            print(end=LINE_CLEAR)

            service_fid, service_geom = service

            filtered_features = src2.filter(bbox=service_geom.bounds)

            for joint in filtered_features:
                joint_fid = str(joint['properties']['FID'])

                cursor.execute(
                    f"UPDATE conn_comp SET fid_to = {service_fid} WHERE fid_to = {joint_fid}"
                )
                rows_modified += cursor.rowcount
                cursor.execute(
                    f"UPDATE conn_comp SET fid_from = {service_fid} WHERE fid_from = {joint_fid}"
                )
                rows_modified += cursor.rowcount
                connection.commit()

    # Delete self-referencing rows
    cursor.execute("DELETE FROM conn_comp WHERE fid_to = fid_from")
    self_refs = cursor.rowcount

    connection.commit()
    connection.close()

    print(f"Service points merged. Rows modified: {rows_modified}, self-refs removed: {self_refs}")

    if report:
        report.add_metric(step, "Service points processed", len(services))
        report.add_metric(step, "conn_comp rows modified", rows_modified)
        report.add_metric(step, "Self-references removed", self_refs)
        report.end_step(step)


# ──────────────────────────────────────────────
# Step 4: simplify_substations (minor fix only)
# ──────────────────────────────────────────────

def simplify_substations(report=None) -> None:
    """
    Collapse internal substation nodes to the General Boundary FID.
    For each substation curtilage, finds edges crossing the boundary
    (flux lines), identifies the internal node, and replaces that
    node's FID with the substation boundary FID in conn_comp.
    """
    step = "simplify_substations"
    if report:
        report.start_step(step)

    # Connect to the connectivity database
    connection = sqlite3.connect('data/network_data.sqlite')
    cursor = connection.cursor()

    # Create flux_lines table
    flux_db_path = 'data/flux_lines.sqlite'
    create_station_flux_lines_table(flux_db_path)

    # Connect to flux table
    connection_flux = sqlite3.connect(flux_db_path)
    cursor_flux = connection_flux.cursor()

    substations_processed = 0
    total_flux = 0
    zero_flux = []
    no_internal = 0

    # Get substation geometries
    with fiona.open('data/assets.gpkg', layer='General Boundary', crs="osgb36") as src:
        total_subs = len(src)
        for i, feature in enumerate(src):
            print(f"{i} of {total_subs}")
            if feature['properties']['Boundary Type'] == 'Substation Curtilage':

                substation_geom = shape(feature['geometry'])
                substation_fid = feature['properties']['FID']

                substation_centroid = list(
                    shape(substation_geom).centroid.coords
                    )[0]

                nearby_stuff = get_things_in_vicinity(substation_centroid)

                flux_count = 0
                for thing in nearby_stuff:
                    feat_geom = shape(thing["geometry"])
                    edge_fid = thing["properties"]["FID"]

                    # Check for lines that cross the substation boundary
                    if substation_geom.intersects(feat_geom) and not feat_geom.within(substation_geom):
                        try:
                            flux_count += 1

                            # This is a flux edge, store it
                            cursor_flux.execute(
                                'INSERT INTO flux_lines (fid, Visited) VALUES (?, ?)',
                                (edge_fid, 0)
                            )

                            # Find what the edge is connected to
                            cursor.execute(
                                f"""
                                SELECT fid_to, fid_from FROM conn_comp
                                WHERE fid_from = {edge_fid} OR fid_to = {edge_fid}
                                """
                                )

                            rows = cursor.fetchall()
                            end_nodes = [
                                (v[0] if v[1] == edge_fid else v[1]) for v in rows
                                ]

                            # Find which node is inside the substation
                            matching_features = [
                                feat for feat in nearby_stuff
                                if feat["properties"]["FID"] in end_nodes
                                ]

                            internal_node_fid = None
                            for feat in matching_features:
                                if shape(feat["geometry"]).within(substation_geom):
                                    internal_node_fid = feat["properties"]["FID"]

                            if internal_node_fid is not None:
                                # Replace internal node with substation FID
                                cursor.execute(
                                    f"UPDATE conn_comp SET fid_to = {substation_fid} "
                                    f"WHERE fid_to = {internal_node_fid} AND fid_from = {edge_fid}"
                                )
                                cursor.execute(
                                    f"UPDATE conn_comp SET fid_from = {substation_fid} "
                                    f"WHERE fid_from = {internal_node_fid} AND fid_to = {edge_fid}"
                                )
                            else:
                                no_internal += 1

                        except sqlite3.IntegrityError:
                            pass

                total_flux += flux_count
                if flux_count == 0:
                    zero_flux.append(substation_fid)
                substations_processed += 1

                connection.commit()
                connection_flux.commit()

    print(f"Substations processed: {substations_processed}")
    print(f"Total flux lines: {total_flux}")
    print(f"Zero-flux substations: {len(zero_flux)}")

    if report:
        avg_flux = total_flux / max(substations_processed, 1)
        report.add_metric(step, "Substations processed", substations_processed)
        report.add_metric(step, "Flux lines found", f"{total_flux} total (avg {avg_flux:.1f} per substation)")
        report.add_metric(step, "Zero-flux substations", f"{len(zero_flux)} {zero_flux[:10]}")
        report.add_metric(step, "No-internal-node cases", no_internal)
        report.end_step(step)

    cursor_flux.close()
    connection_flux.close()
    cursor.close()
    connection.close()


# ──────────────────────────────────────────────
# Step 5: simplify_pmt_substations (NEW)
# ──────────────────────────────────────────────

def simplify_pmt_substations(report=None) -> None:
    """
    Create synthetic 5m buffer boundaries for PMT transformers.

    For each PMT:
    1. Buffer the transformer point by 5m → synthetic curtilage circle
    2. Find LV edges crossing the circle (flux lines) via spatial query
    3. Collapse internal nodes (Ways, joints inside the circle) to PMT FID
    4. Record flux lines in the flux_lines table
    5. UPDATE lv_assets: Geometry → buffer circle, Is_Substation → 1

    Uses connectivity validation to check agreement with spatial results.
    """
    step = "simplify_pmt_substations"
    if report:
        report.start_step(step)

    # Connect to databases
    conn_lv = sqlite3.connect("data/lv_assets.sqlite")
    cur_lv = conn_lv.cursor()

    conn_net = sqlite3.connect("data/network_data.sqlite")
    cur_net = conn_net.cursor()

    flux_db_path = "data/flux_lines.sqlite"
    if not os.path.exists(flux_db_path):
        create_station_flux_lines_table(flux_db_path)
    conn_flux = sqlite3.connect(flux_db_path)
    cur_flux = conn_flux.cursor()

    # Get all PMT Transformer rows
    cur_lv.execute("SELECT fid, Geometry FROM lv_assets WHERE Asset_Type = 'PMT Transformer'")
    pmts = cur_lv.fetchall()

    total_pmts = len(pmts)
    total_flux = 0
    zero_flux_pmts = []
    spatial_conn_disagreements = []
    no_internal = 0

    print(f"Processing {total_pmts} PMT Transformers...")

    # Pre-open assets.gpkg layers for faster repeated spatial queries
    lv_layer_names = [
        "LV Board", "LV Conductor", "LV Joint", "LV Service",
        "Way", "Leader Line", "Service Point", "Switch"
    ]
    layer_sources = {}
    for lyr in lv_layer_names:
        layer_sources[lyr] = fiona.open("data/assets.gpkg", "r", layer=lyr)

    for i, (pmt_fid, geom_wkb) in enumerate(pmts):
        if i % 500 == 0:
            print(f"  PMT {i}/{total_pmts}...")

        # Deserialize point geometry and create 5m buffer
        pmt_point = wkb.loads(geom_wkb)
        buffer_circle = pmt_point.buffer(5)
        buffer_wkb = wkb.dumps(buffer_circle)

        # Find nearby LV assets (10m search radius to catch edges extending beyond 5m buffer)
        pmt_centroid = list(pmt_point.coords)[0]
        nearby_stuff = _get_nearby_features_from_sources(
            pmt_centroid, layer_sources, buffer_radius=10
        )

        # Find flux edges: intersect the buffer but are not fully within it
        flux_count = 0
        for thing in nearby_stuff:
            feat_geom = shape(thing["geometry"])
            edge_fid = thing["properties"]["FID"]

            if buffer_circle.intersects(feat_geom) and not feat_geom.within(buffer_circle):
                flux_count += 1

                try:
                    cur_flux.execute(
                        'INSERT INTO flux_lines (fid, Visited) VALUES (?, ?)',
                        (edge_fid, 0)
                    )
                except sqlite3.IntegrityError:
                    pass  # Already recorded for another substation

                # Find what this flux edge connects to in conn_comp
                cur_net.execute(
                    "SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = ? OR fid_to = ?",
                    (edge_fid, edge_fid)
                )
                rows = cur_net.fetchall()
                end_nodes = [(r[0] if r[1] == edge_fid else r[1]) for r in rows]

                # Find which end node is inside the buffer (internal node)
                matching_features = [
                    feat for feat in nearby_stuff
                    if feat["properties"]["FID"] in end_nodes
                ]

                internal_node_fid = None
                for feat in matching_features:
                    if shape(feat["geometry"]).within(buffer_circle):
                        if feat["properties"]["FID"] != pmt_fid:
                            internal_node_fid = feat["properties"]["FID"]

                if internal_node_fid is not None:
                    # Replace internal node with PMT FID in conn_comp
                    cur_net.execute(
                        "UPDATE conn_comp SET fid_to = ? WHERE fid_to = ? AND fid_from = ?",
                        (pmt_fid, internal_node_fid, edge_fid)
                    )
                    cur_net.execute(
                        "UPDATE conn_comp SET fid_from = ? WHERE fid_from = ? AND fid_to = ?",
                        (pmt_fid, internal_node_fid, edge_fid)
                    )
                else:
                    # Check if one of the end nodes is already the PMT FID
                    if pmt_fid not in end_nodes:
                        no_internal += 1

        total_flux += flux_count
        if flux_count == 0:
            zero_flux_pmts.append(pmt_fid)

        # Connectivity validation
        cur_net.execute(
            "SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = ? OR fid_to = ?",
            (pmt_fid, pmt_fid)
        )
        conn_neighbors = set()
        for r in cur_net.fetchall():
            conn_neighbors.add(r[0] if r[1] == pmt_fid else r[1])

        spatial_flux_fids = set()
        for thing in nearby_stuff:
            feat_geom = shape(thing["geometry"])
            if buffer_circle.intersects(feat_geom) and not feat_geom.within(buffer_circle):
                spatial_flux_fids.add(thing["properties"]["FID"])

        if spatial_flux_fids and not spatial_flux_fids.issubset(conn_neighbors):
            spatial_conn_disagreements.append(pmt_fid)

        # UPDATE lv_assets: geometry → buffer, Is_Substation → 1
        cur_lv.execute(
            "UPDATE lv_assets SET Geometry = ?, Is_Substation = 1 WHERE fid = ?",
            (buffer_wkb, pmt_fid)
        )

        # Commit periodically
        if i % 100 == 0:
            conn_net.commit()
            conn_flux.commit()
            conn_lv.commit()

    # Close pre-opened layer sources
    for src in layer_sources.values():
        src.close()

    # Delete self-referencing rows created by simplification
    cur_net.execute("DELETE FROM conn_comp WHERE fid_to = fid_from")

    conn_net.commit()
    conn_flux.commit()
    conn_lv.commit()

    print(f"  PMTs processed: {total_pmts}")
    print(f"  Total flux lines: {total_flux} (avg {total_flux / max(total_pmts, 1):.1f} per PMT)")
    print(f"  Zero-flux PMTs: {len(zero_flux_pmts)}")
    print(f"  Spatial/connectivity disagreements: {len(spatial_conn_disagreements)}")
    print(f"  No-internal-node cases: {no_internal}")

    if report:
        avg = total_flux / max(total_pmts, 1)
        report.add_metric(step, "PMTs processed", total_pmts)
        report.add_metric(step, "Flux lines found", f"{total_flux} total (avg {avg:.1f} per PMT)")
        report.add_metric(step, "Zero-flux PMTs", f"{len(zero_flux_pmts)} {zero_flux_pmts[:10]}")
        report.add_metric(
            step, "Spatial/connectivity disagreements",
            f"{len(spatial_conn_disagreements)} {spatial_conn_disagreements[:10]}",
        )
        report.add_metric(step, "No-internal-node cases", no_internal)
        report.end_step(step)

    cur_lv.close()
    conn_lv.close()
    cur_net.close()
    conn_net.close()
    cur_flux.close()
    conn_flux.close()


# ──────────────────────────────────────────────
# Step 6: enrich_substations (NEW)
# ──────────────────────────────────────────────

def enrich_substations(report=None) -> None:
    """
    Attach transformer capacity and substation type to General Boundary
    rows via spatial join (transformer point within curtilage polygon).

    PMTs already have their own Rating_Normal directly — this step
    only enriches ground-mounted substations.
    """
    step = "enrich_substations"
    if report:
        report.start_step(step)

    conn_lv = sqlite3.connect("data/lv_assets.sqlite")
    cur_lv = conn_lv.cursor()

    # Load General Boundary rows (substations)
    cur_lv.execute("""
        SELECT fid, Geometry FROM lv_assets
        WHERE Asset_Type = 'General Boundary' AND Is_Substation = 1
    """)
    substations = cur_lv.fetchall()

    # Load Transformer + PMT Transformer rows (for capacity)
    cur_lv.execute("""
        SELECT fid, Geometry, Rating_Normal, Variable_Rating, Infeed_Voltage
        FROM lv_assets WHERE Asset_Type IN ('Transformer', 'PMT Transformer')
    """)
    transformers = cur_lv.fetchall()

    # Load Substation layer rows (for Substation_Type)
    cur_lv.execute("""
        SELECT fid, Geometry, Substation_Type, Infeed_Voltage
        FROM lv_assets WHERE Asset_Type = 'Substation'
    """)
    substation_metadata = cur_lv.fetchall()

    # Build spatial index for transformers
    tx_points = []
    tx_data = []
    for t_fid, t_geom, t_rating, t_var_rating, t_infeed in transformers:
        if t_geom:
            point = wkb.loads(t_geom)
            tx_points.append(point)
            tx_data.append({
                "fid": t_fid,
                "rating": t_rating or "",
                "var_rating": t_var_rating or "",
                "infeed": t_infeed or "",
            })

    # Build spatial index for substation metadata
    sm_points = []
    sm_data = []
    for s_fid, s_geom, s_type, s_infeed in substation_metadata:
        if s_geom:
            point = wkb.loads(s_geom)
            sm_points.append(point)
            sm_data.append({
                "fid": s_fid,
                "type": s_type or "",
                "infeed": s_infeed or "",
            })

    tx_tree = STRtree(tx_points) if tx_points else None
    sm_tree = STRtree(sm_points) if sm_points else None

    matched = 0
    unmatched = 0
    multi_tx = 0
    unmatched_fids = []

    print(f"Enriching {len(substations)} substations with transformer capacity...")

    for sub_fid, sub_geom_wkb in substations:
        if not sub_geom_wkb:
            continue

        sub_polygon = wkb.loads(sub_geom_wkb)

        # Find transformers within this substation polygon
        if tx_tree is not None:
            candidate_idxs = tx_tree.query(sub_polygon)
            tx_within = []
            for idx in candidate_idxs:
                if sub_polygon.contains(tx_points[idx]):
                    tx_within.append(tx_data[idx])

            if tx_within:
                matched += 1
                if len(tx_within) > 1:
                    multi_tx += 1

                ratings = [t["rating"] for t in tx_within if t["rating"]]
                combined_rating = "; ".join(ratings) if ratings else ""
                var_ratings = [
                    t["var_rating"] for t in tx_within
                    if t["var_rating"] and t["var_rating"] != "Not Applicable"
                ]
                combined_var = "; ".join(var_ratings) if var_ratings else ""
                infeed = tx_within[0]["infeed"]

                cur_lv.execute("""
                    UPDATE lv_assets
                    SET Rating_Normal = ?, Variable_Rating = ?, Infeed_Voltage = ?
                    WHERE fid = ?
                """, (combined_rating, combined_var, infeed, sub_fid))
            else:
                unmatched += 1
                if len(unmatched_fids) < 20:
                    unmatched_fids.append(sub_fid)

        # Find substation metadata within this polygon
        if sm_tree is not None:
            candidate_idxs = sm_tree.query(sub_polygon)
            for idx in candidate_idxs:
                if sub_polygon.contains(sm_points[idx]):
                    if sm_data[idx]["type"]:
                        cur_lv.execute(
                            "UPDATE lv_assets SET Substation_Type = ? WHERE fid = ?",
                            (sm_data[idx]["type"], sub_fid)
                        )
                    break  # Take first match

    conn_lv.commit()

    print(f"  Matched: {matched}/{len(substations)}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Multi-transformer: {multi_tx}")

    if report:
        report.add_metric(step, "Substations with transformer match", f"{matched} / {len(substations)}")
        report.add_metric(step, "Unmatched substations", f"{unmatched} {unmatched_fids[:10]}")
        report.add_metric(step, "Multi-transformer substations", multi_tx)
        report.end_step(step)

    cur_lv.close()
    conn_lv.close()


# ──────────────────────────────────────────────
# Step 7: copy_conn_to_lv (NEW)
# ──────────────────────────────────────────────

def copy_conn_to_lv(report=None) -> None:
    """
    Copy conn_comp (fid_from, fid_to, flow) from network_data.sqlite
    into lv_assets.sqlite.  Strips 7 columns including GEOMETRY.
    """
    step = "copy_conn_to_lv"
    if report:
        report.start_step(step)

    conn_lv = sqlite3.connect("data/lv_assets.sqlite")
    cur_lv = conn_lv.cursor()

    cur_lv.execute("DROP TABLE IF EXISTS conn_comp")

    print("Attaching network_data.sqlite and copying conn_comp...")
    cur_lv.execute("ATTACH 'data/network_data.sqlite' AS netdata")
    cur_lv.execute(
        "CREATE TABLE conn_comp AS SELECT fid_from, fid_to, flow FROM netdata.conn_comp"
    )

    cur_lv.execute("SELECT COUNT(*) FROM conn_comp")
    row_count = cur_lv.fetchone()[0]

    print(f"Creating indexes on {row_count} rows...")
    cur_lv.execute("CREATE INDEX idx_cc_from ON conn_comp(fid_from)")
    cur_lv.execute("CREATE INDEX idx_cc_to ON conn_comp(fid_to)")

    cur_lv.execute("DETACH netdata")
    conn_lv.commit()

    print(f"  Rows copied: {row_count}")

    if report:
        report.add_metric(step, "Rows copied", row_count)
        report.add_metric(
            step, "Columns",
            "fid_from, fid_to, flow (stripped 7 of 10, GEOMETRY excluded)",
        )
        report.end_step(step)

    cur_lv.close()
    conn_lv.close()


# ──────────────────────────────────────────────
# Step 8: create_graph_db (unchanged)
# ──────────────────────────────────────────────

def create_graph_db(fname: str = "results/graph.sqlite") -> None:
    """Create empty graph.sqlite output database with WAL mode."""

    file_path = fname
    if os.path.exists(file_path):
        os.remove(file_path)

    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidence_list (
            edge_fid INTEGER PRIMARY KEY,
            node_from INTEGER,
            node_to INTEGER,
            parent_substation INTEGER NULL,
            parent_switch INTEGER NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mapped_substations (
            substation_fid INTEGER PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mapped_switches (
            switch_fid INTEGER PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_list (
            fid INTEGER PRIMARY KEY,
            Geometry BLOB NULL,
            Asset_Type TEXT NULL,
            Voltage INTEGER NULL,
            Material TEXT NULL,
            Conductors_Per_Phase INTEGER NULL,
            Cable_Size TEXT NULL,
            Insulation TEXT NULL,
            Installation_Date DATETIME NULL,
            Phases_Connected TEXT NULL,
            Sleeve_Type TEXT NULL,
            Associated_Cable TEXT NULL,
            Switch_Status TEXT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edge_list (
            fid INTEGER PRIMARY KEY,
            Geometry BLOB NULL,
            Asset_Type TEXT NULL,
            Voltage INTEGER NULL,
            Material TEXT NULL,
            Conductors_Per_Phase INTEGER NULL,
            Cable_Size TEXT NULL,
            Insulation TEXT NULL,
            Installation_Date DATETIME NULL,
            Phases_Connected TEXT NULL,
            Sleeve_Type TEXT NULL,
            Associated_Cable TEXT NULL,
            Switch_Status TEXT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS building_matches (
            service_point_fid INTEGER,
            substation_fid INTEGER,
            toid TEXT NULL,
            uprn TEXT NULL,
            match_method TEXT NULL,
            dist_to_building REAL NULL,
            activity_type TEXT NULL,
            PRIMARY KEY (service_point_fid, substation_fid)
        )
    """)

    connection.commit()
    cursor.close()
    connection.close()


# ──────────────────────────────────────────────
# Utility functions (unchanged)
# ──────────────────────────────────────────────

def clear_graph_db() -> None:
    file_path = "results/graph.sqlite"
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    try:
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DELETE FROM {table_name}")
            print(f"Table '{table_name}' cleared.")
        connection.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        connection.close()


def get_mapped_substations_ids_from_summary() -> set:
    file_path = "results/summary.csv"
    df = pd.read_csv(file_path)
    fids = pd.to_numeric(df["substation_fid"], errors="coerce")
    valid_fids = fids.dropna().astype(int).tolist()
    return valid_fids


def get_mapped_substations_from_logs(log_dir="logs") -> set:
    """
    Scans all log files in the given directory, extracts FIDs,
    and returns a set of unique FIDs as integers.
    """
    print("Scanning log files for mapped substations...")
    fid_pattern = re.compile(r"FID:(\d+)")
    fids = set()

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            with open(os.path.join(log_dir, filename), "r", encoding="utf-8") as f:
                for line in f:
                    match = fid_pattern.search(line)
                    if match:
                        fids.add(int(match.group(1)))
    print(f"Found {len(fids)} unique FIDs in log files.")
    return fids


def get_mapped_substations_ids_from_database() -> set:
    file_path = "results/graph.sqlite"
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM mapped_substations")
        rows = cursor.fetchall()

        result_set = list()
        for row in rows:
            result_set.append(row[0])

        return result_set
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return set()
    finally:
        connection.close()
