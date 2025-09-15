"""
File containing functions for simplifying databases
and creating new databases
"""

import sqlite3
import fiona
from shapely.geometry import (
    shape,
    Polygon,
    Point
)
from shapely.affinity import translate, rotate
import os
import re 
from datetime import datetime
from shapely import wkb
import pandas as pd

LINE_CLEAR = '\x1b[2K'


def create_lv_db() -> None:
    """
    Function that creates a new database called 
    lv_assets.sqlite that contains only low voltage
    relevant things from assets.gpkg.
    """

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
        "General Boundary"
        ]
    
    # Create the new database
    file_path = "data/lv_assets.sqlite"
    if os.path.exists(file_path):
        os.remove(file_path)
    connection = sqlite3.connect(file_path)

    # Set up its structure
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lv_assets (fid INTEGER PRIMARY KEY, Geometry BLOB NULL, Asset_Type TEXT NULL, Voltage INTEGER NULL, Material TEXT NULL, Conductors_Per_Phase INTEGER NULL, Cable_Size TEXT NULL, Insulation TEXT NULL, Installation_Date DATETIME NULL, Phases_Connected TEXT NULL, Sleeve_Type TEXT NULL, Associated_Cable TEXT NULL, Switch_Status TEXT NULL,
        Is_Substation INTEGER, Is_Switch INTEGER)
        """
        )
    
    # Get all rows of interest from the geopackage, put them into a 
    # useful format and insert them into the new database
    for i, layer in enumerate(layer_names):
        with fiona.open('data/assets.gpkg', layer=layer) as src:
            print(f"Processing layer {i+1} of {len(layer_names)}...")
            for feature in src:

                # Get the FID
                fid = feature['properties']["FID"]

                # Get the geometry and put it into wkb format
                geometry = feature['geometry']
                wkb_geom = wkb.dumps(shape(geometry))

                # Get the other properties. Some might not exist for the 
                # particular layer. So try and if not, fill with null
                try:
                    asset_type = str(feature["properties"]["asset_type"])
                except KeyError:
                    asset_type = ""
                try:
                    material = str(feature["properties"]["Material "])
                except KeyError:
                    material = ""
                try:
                    conductors_per_phase = int(feature["properties"]["Conductors_Per_Phase"])
                except TypeError:
                    conductors_per_phase = ""
                except KeyError:
                    conductors_per_phase = ""
                try:
                    cable_size = str(feature["properties"]["Cable Size"])
                except KeyError:
                    cable_size = ""
                try:
                    insulation = str(str(feature["properties"]["Insulation"]))
                except KeyError:
                    insulation = ""
                try:
                    phases_connected = str(str(feature["properties"]["Phases Connected"]))
                except KeyError:
                    phases_connected = ""
                try:
                    sleeve_type = str(str(feature["properties"]["Sleeve Type"]))
                except KeyError:
                    sleeve_type = ""
                try:
                    associated_cable = str(str(feature["properties"]["Associated Cable"]))
                except KeyError:
                    associated_cable = ""
                try:
                    switch_status = str(str(feature["properties"]["Open or Closed"]))
                except KeyError:
                    switch_status = ""

                if layer == "Switch":
                    is_switch = 1
                else:
                    is_switch = 0
                if layer == "General Boundary":
                    is_substation = 1
                else:
                    is_substation = 0
                
                # Installation date needs converting into SQL datetime type
                try:
                    install_date = str(str(feature["properties"]["Installation Date"]))
                except KeyError:
                    install_date = ""
                if install_date != "":
                    try:
                        install_date = datetime.strptime(str(install_date), "%d/%m/%Y")
                        install_date = install_date.strftime("%Y-%m-%d")
                        year = int(install_date[:4])
                        if year < 1900 or year > 2023:
                            install_date = ""
                    except ValueError:
                        install_date = ""
                    

                # Voltage needs converting to int
                # Using regular expression for this
                try:
                    voltage = str(feature["properties"]["Nominal Voltage"])
                except KeyError:
                    try:
                        voltage = str(feature["properties"]["Voltage"])
                    except KeyError:
                        voltage = ""
                if voltage != "":
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

                # Insert the data into the SQLite database
                cursor.execute('INSERT INTO lv_assets (fid, Geometry, Asset_Type, Voltage, Material, Conductors_Per_Phase, Cable_Size, Insulation, Installation_Date, Phases_Connected, Sleeve_Type, Associated_Cable, Switch_Status, Is_Substation, Is_Switch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (fid, wkb_geom, asset_type, voltage, material, conductors_per_phase, cable_size, insulation, install_date, phases_connected, sleeve_type, associated_cable, switch_status, is_substation, is_switch))

    # Add a new column called "cat" to the table
    cursor.execute("ALTER TABLE lv_assets ADD COLUMN cat TEXT")

    # Update the "cat" column based on the values in the "asset_type" column
    cursor.execute("UPDATE lv_assets SET cat = CASE \
                    WHEN asset_type IN ('LV Conductor', 'LV Service', 'Leader Line', 'Ground Line') THEN 'edge' \
                    WHEN asset_type IN ('LV Board', 'LV Joint', 'Way', 'Service Point', 'Wall Termination', 'Ground Point', 'Switch') THEN 'node' \
                    ELSE NULL END")

    # Close the database connection
    cursor.close()
    connection.commit()
    connection.close()

    
def create_station_flux_lines_table(flux_db_path) -> None:
    """
    Function that creates an empty database 
    that will be used to contain the
    ids of lines that are directly incident on 
    substations.
    """

    # Create the new database
    file_path = flux_db_path

    # Set up its structure
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS flux_lines (fid INTEGER PRIMARY KEY, 
        Visited INTEGER)
        """
    )
    cursor.close()
    connection.commit()
    connection.close() 


def reset_station_flux_lines_table(flux_db_path) -> None:
    """
    Function to reset the visited status of lines
    incident on substations.
    """

    connection = sqlite3.connect(flux_db_path)
    cursor = connection.cursor()
    cursor.execute("UPDATE flux_lines SET Visited = 0")
    cursor.close()
    connection.commit()
    connection.close() 


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


def collapse_switch_boxes(
        fname: str
    ):
    """
    Function to simplify switch boxes into 
    single nodes.
    """

    # Get the switch, way and conn data
    try:
        # Open a connection to the SQLite database
        connection = sqlite3.connect(fname)

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # Define the SQL query to select the 'FID' column from the 'Switch' table
        sql_query = "SELECT FID FROM Switch"

        # Execute the SQL query
        cursor.execute(sql_query)

        # Fetch all the rows (FID values) from the query result
        switches = cursor.fetchall()

        # Close the cursor and the database connection
        cursor.close()
        connection.close()

    except sqlite3.Error as e:
        print("SQLite error:", e)


    # Open the SQLite database connection
    database_path = 'data/network_data.sqlite'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # # Get the contained features from the "Way" table
    with fiona.open(fname, layer='Way') as src2:

        ll = len(switches)
        for i, switch in enumerate(switches):

            print(f"{i} of switch {ll}...")

            switch_fid, switch_geom = switch

            buffered_geometry = switch_geom.buffer(5)
            filtered_features = src2.filter(bbox=buffered_geometry.bounds)

            for f in filtered_features:
                f_fid = str(f['properties']['FID'])

                # Update rows where fid_to is equal to f_fid
                cursor.execute(f"UPDATE conn_comp SET fid_to = {switch_fid} WHERE fid_to = {f_fid}")

                # Update rows where fid_from is equal to f_fid
                cursor.execute(f"UPDATE conn_comp SET fid_from = {switch_fid} WHERE fid_from = {f_fid}")
                connection.commit()

    # Delete rows where both fid_to and fid_from are equal to fid_switch
    cursor.execute("DELETE FROM conn_comp WHERE fid_to = fid_from")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()


def merge_service_points(
        fname: str
    ):
    """
    Function that simplifies the case where a service
    point and a joint are collocated into a single node.
    """

    # Get the switch geometries
    services = []
    with fiona.open(fname, layer='Service Point') as src:
        for i, feature in enumerate(src):
            print(100*i/len(src),"%", end='\r')
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
    
    # Get the contained features from the "Way" table
    with fiona.open(fname, layer='LV Joint') as src2:

        ll = len(services)
        for i, service in enumerate(services):

            print(f"{i} of service {ll}...", end='\r')
            print(end=LINE_CLEAR)

            service_fid, service_geom = service

            # buffered_geometry = switch_geom.buffer(5)
            filtered_features = src2.filter(bbox=service_geom.bounds)

            for switch in filtered_features:
                switch_fid = str(switch['properties']['FID'])


                cursor.execute(f"UPDATE conn_comp SET fid_to = {service_fid} WHERE fid_to = {switch_fid}")

                # Update rows where fid_from is equal to f_fid
                cursor.execute(f"UPDATE conn_comp SET fid_from = {service_fid} WHERE fid_from = {switch_fid}")
                connection.commit()

    # Delete rows where both fid_to and fid_from are equal to fid_switch
    cursor.execute("DELETE FROM conn_comp WHERE fid_to = fid_from")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()


def get_things_in_vicinity(
        centroid: Point,
        buffer_radius: int | float = 20
        ) -> list:
    """
    Function to get all low voltage assets that 
    are within buffer_radius of the centroid.
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

    # Create a Shapely Point object from centroid coordinates
    substation_centroid_point = Point(centroid)

    # Create a buffered area around the centroid (20m radius)
    buffered_area = substation_centroid_point.buffer(buffer_radius)

    # Open the Geopackage file, one layer at a time
    features_in_vicinity = []
    for layer in layer_names:
        with fiona.open("data/assets.gpkg", "r", layer=layer) as src:
            
            # Filter features within the buffered area
            for feature in src.filter(bbox=buffered_area.bounds):
                features_in_vicinity.append(feature)

    return features_in_vicinity

#function to check if a flux line is connected to an open or closed way
def flux_way_check(fid, cursor):
    #find connecting parts to flux edge
    cursor.execute(
        f"""
        SELECT fid_to, fid_from FROM conn_new WHERE fid_from = {fid} OR fid_to = {fid}
        """
    )
    rows = cursor.fetchall()
    # Using a set to store unique FIDs
    unique_fids = set()

    # Iterate through the pairs and add each FID to the set
    for pair in rows:
        unique_fids.update(pair)

    # Convert the set to a list to get individual, unique FIDs
    individual_fids = list(unique_fids)
    # Construct the SQL query using the IN clause
    placeholders = ', '.join('?' for _ in individual_fids)
    query = f"SELECT fid,`Normal Status` FROM Way WHERE fid IN ({placeholders})"
    conn_assets_ways = sqlite3.connect('data/assets.gpkg')
    cursor_assets_ways = conn_assets_ways.cursor()
    cursor_assets_ways.execute(query, individual_fids)

    # Fetch all rows
    ways_details = cursor_assets_ways.fetchall()
    closed_way_fids = [fid for fid, status in ways_details if status == 'Closed']

    # close connection to ways
    conn_assets_ways.close()

    return False if not closed_way_fids else True

def simplify_substations() -> None:
    """
    Function to simplify substations into a single node.
    """

    # Connect to the SQLite database
    connection = sqlite3.connect('data/network_data.sqlite')
    cursor = connection.cursor()

    # Empty data base to record lines coming 
    # out of substations
    create_station_flux_lines_table()

    # Connect to flux table
    connection_flux = sqlite3.connect('data/flux_lines.sqlite')
    cursor_flux = connection_flux.cursor()

    # Get substation geometries
    with fiona.open('data/assets.gpkg', layer='General Boundary', crs="osgb36") as src:
        for i, feature in enumerate(src):
            print(f"{i} of {len(src)}")
            if feature['properties']['Boundary Type'] == 'Substation Curtilage':

                # Get the geometry and ID of the substation
                substation_geom = shape(feature['geometry'])
                substation_fid = feature['properties']['FID']

                # Get the central point of this substation 
                substation_centroid = list(
                    shape(substation_geom).centroid.coords
                    )[0]

                nearby_stuff = get_things_in_vicinity(substation_centroid)

                for thing in nearby_stuff:
                    feat_geom = shape(thing["geometry"])
                    edge_fid = thing["properties"]["FID"]

                    #Check if edge is connected to a closed way
                    #way_status = flux_way_check(edge_fid,cursor)
                    #print(f"edge {edge_fid} has way status: {way_status}")
                    way_status = True

                    # Check foe lines that cross the substation boundary.
                    if substation_geom.intersects(feat_geom) and not feat_geom.within(substation_geom) and way_status:
                        try:


                            # This is a flux edge, store it
                            cursor_flux.execute('INSERT INTO flux_lines (fid, Visited) VALUES (?, ?)', (edge_fid, 0))

                            # Find what the edge is connected to
                            cursor.execute(
                                f"""
                                SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = {edge_fid} OR fid_to = {edge_fid}
                                """
                                )

                            rows = cursor.fetchall()
                            end_nodes = [
                                (v[0] if v[1] == edge_fid else v[1]) for v in rows
                                ]

                            # Find more data about these nodes.
                            matching_features = [
                                feat for feat in nearby_stuff if feat["properties"]["FID"] in end_nodes
                                ]

                            # Find which node is inside the substation.
                            internal_node_fid = None
                            for feat in matching_features:
                                if shape(feat["geometry"]).within(substation_geom):
                                    internal_node_fid = feat["properties"]["FID"]

                            if internal_node_fid != None:
                                print("Something found")

                                # Now find all refs of internal node in the conn_comp
                                # table. Replace these refs with the fid of the 
                                # substation curtilage
                                cursor.execute(f"UPDATE conn_comp SET fid_to = {substation_fid} WHERE fid_to = {internal_node_fid} AND fid_from = {edge_fid}")
                                cursor.execute(f"UPDATE conn_comp SET fid_from = {substation_fid} WHERE fid_from = {internal_node_fid} AND fid_to = {edge_fid}")
                        
                        except sqlite3.IntegrityError:
                            pass

                connection.commit()
                connection_flux.commit()

    # Close the database connection
    cursor_flux.close()
    connection_flux.close()
    cursor.close()
    connection.close()


def create_graph_db() -> None:
    """
    Function to set up main sqlite file
    it should contain incidence list
    list of mapped substations
    node list and edge list
    """

    # Create the new database
    file_path = "data/graph.sqlite"
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # connect to it
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    # Enable WAL mode
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidence_list (edge_fid INTEGER PRIMARY KEY, node_from INTEGER, node_to INTEGER, parent_substation INTEGER NULL, parent_switch INTEGER NULL)
        """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS mapped_substations (substation_fid INTEGER PRIMARY KEY)
        """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS mapped_switches (switch_fid INTEGER PRIMARY KEY)
        """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS node_list (fid INTEGER PRIMARY KEY, Geometry BLOB NULL, Asset_Type TEXT NULL, Voltage INTEGER NULL, Material TEXT NULL, Conductors_Per_Phase INTEGER NULL, Cable_Size TEXT NULL, Insulation TEXT NULL, Installation_Date DATETIME NULL, Phases_Connected TEXT NULL, Sleeve_Type TEXT NULL, Associated_Cable TEXT NULL, Switch_Status TEXT NULL)
        """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS edge_list (fid INTEGER PRIMARY KEY, Geometry BLOB NULL, Asset_Type TEXT NULL, Voltage INTEGER NULL, Material TEXT NULL, Conductors_Per_Phase INTEGER NULL, Cable_Size TEXT NULL, Insulation TEXT NULL, Installation_Date DATETIME NULL, Phases_Connected TEXT NULL, Sleeve_Type TEXT NULL, Associated_Cable TEXT NULL, Switch_Status TEXT NULL)
        """
        )
    
    connection.commit()
    cursor.close()
    connection.close()


def clear_graph_db() -> None:

    file_path = "data/graph.sqlite"
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    # Get a list of all table names in the database
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
    file_path = "data/summary.csv"
    
    df = pd.read_csv(file_path)
    # Convert to numeric, forcing errors to NaN
    fids = pd.to_numeric(df["substation_fid"], errors="coerce")
    # Drop NaN and convert to int
    valid_fids = fids.dropna().astype(int).tolist()
    return valid_fids



def get_mapped_substations_ids_from_database() -> set:
    file_path = "data/graph.sqlite"
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM mapped_substations")
        rows = cursor.fetchall()

        # Create a set to store the rows
        result_set = list()
        for row in rows:
            result_set.append(row[0])

        return result_set
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return set()
    finally:
        connection.close()



