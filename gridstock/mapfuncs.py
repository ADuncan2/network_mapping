"""
File containing functions to search and
map networks.
"""

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import sqlite3
from gridstock.recorder import NetworkData
from shapely.geometry import Point, LineString, shape
from shapely import wkb
from pyproj import Transformer
import osmnx as ox
import fiona
import sys
sys.setrecursionlimit(5000)
import matplotlib.pyplot as plt
from shapely.wkt import loads
from matplotlib.colors import ListedColormap
import numpy as np
from itertools import combinations
from gridstock.dbcleaning import (
    reset_station_flux_lines_table,
    create_station_flux_lines_table)

def interpolate_coordinates(coordinates, num_new_points):
    new_coordinates = []
    for i in range(len(coordinates) - 1):
        start_point = coordinates[i]
        end_point = coordinates[i + 1]

        direction_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])

        for step in range(0, num_new_points):
            interpolation_factor = step / (num_new_points-1)
            new_x = start_point[0] + (interpolation_factor * direction_vector[0])
            new_y = start_point[1] + (interpolation_factor * direction_vector[1])
            if step == num_new_points:
                new_x = end_point[0]
                new_y = end_point[1]
            new_coordinates.append((new_x, new_y))

    return new_coordinates

def lines_segmentation(
        linestring: LineString,
        points: dict
        ) -> tuple[list, list]:
    sub_lines = []
    bookends = []
    start_point = linestring.coords[0]
    start_fid = points.keys[0] #problem here - points not in spatial order

    for fid, geom in points.items():
        if geom.geom_type == "Polygon":
            end_point = shape(geom).centroid
            end_fid = fid
            segment = LineString([start_point, end_point])
            sub_lines.append(segment)
            bookends.append([start_fid,end_fid])
            start_point = end_point
            start_fid = end_fid
        else:
            end_point= geom.coords[0]
            end_fid = fid
            segment = LineString([start_point, end_point])
            sub_lines.append(segment)
            bookends.append([start_fid,end_fid])
            start_point = end_point
            start_fid = end_fid
    
    return sub_lines, bookends

##checking for overlapping points in hyper-edges
#if overlapping points present one of the points is deleted. 
#This could be changed in future to check which points are nodes and delete the other, for now it just deletes at random AS A TEST
def check_point_distances(points, threshold):
    #print("checking points")
    points_list = list(points.values())
    #print(points_list)
    overlap = []
    for i in range(len(points_list)):
        for j in range(i + 1, len(points_list)):
            distance = points_list[i].distance(points_list[j])
            #print(distance)
            if distance < threshold:
                overlap.append(points_list[i])
                overlap.append(points_list[j])
            else:
                pass
    #print(f"overlap:{overlap}")
    if len(overlap)>0:
        matching_keys = [key for key, value in points.items() if value in overlap]
        # Create a new dictionary and remove the first matching key-value pair
        new_points = points.copy()
        key_to_remove = matching_keys[0]
        print(f"points found to overlap, removed node:{key_to_remove}")
        del new_points[key_to_remove]
        return new_points
    else:
        return points
            

def finding_insert_indices(coordinates, points):
    insert_indices = dict()
    count = 0
    for fid, geom in points.items():
        if geom.geom_type == "Polygon":
            #print("ITS A POLYGON!! FUCK!!!")
            central_point = shape(geom).centroid
            #print(f"central_point:{central_point}, central coords: {central_point.coords[0]}")
            coordinates[0] = central_point.coords[0]
            
            insert_indices[0] = fid
        else:
            min_distance = float('inf')
            closest_index = None
            for i, coord in enumerate(coordinates):
                distance = Point(coord).distance(geom)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            #print(closest_index)
            coordinates[closest_index] = geom.coords[0]
            insert_indices[closest_index] = fid
            count = count + 1
    #print(f"insert_ind:{len(insert_indices)}, points:{len(points)}")
    #print(coordinates)
    if len(insert_indices)<len(points):
        coordinates_new = interpolate_coordinates(coordinates,5)
        #print(points)
        #print("interpolated!")
        return(finding_insert_indices(coordinates_new, points))

    else:
        return(insert_indices,coordinates)

def add_points_and_divide_linestring(
        linestring: LineString,
        points: dict
        ) -> tuple[list, list]:
    """
    Function to take an edge that has multiple
    junctions on it, given in the points dict, 
    and subdivide it into smaller regular edges.
    """
    coordinates = list(linestring.coords)
    #insert_indices = dict()
    bookends = []

    #CHECKING SECTION
    points = check_point_distances(points, 0.04)
    
    #interpolate additional coordinates if ratio of coordinates to points is <10
    # if len(coordinates) <= len(points)*50:
    #     try:
    #         coordinates = interpolate_coordinates(coordinates,50)
    #     except:
    #         print("Interpolation error")
    #     #print("Interpolation to expand number of coordinates")


    insert_indices, coordinates = finding_insert_indices(coordinates,points)



    # count = 0
    # for fid, geom in points.items():
    #     if geom.geom_type == "Polygon":
    #         coordinates[0] = shape(geom).centroid
    #         insert_indices[0] = fid
            
    #     else:
    #         min_distance = float('inf')
    #         closest_index = None
    #         for i, coord in enumerate(coordinates):
    #             distance = Point(coord).distance(geom)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_index = i
    #         #print(closest_index)
    #         coordinates[closest_index] = geom.coords[0]
    #         insert_indices[closest_index] = fid
    #         count = count + 1
    
    # Divide the LineString into smaller LineStrings
    #print(f"count: {count}")
    linestrings = []
    ordered_points = []
    start_index = 0
    
    if len(insert_indices)!=len(points):
        print("Insert_ind problem!")

    ##this section makes sure that the index used to add the first node to the line is found, without assuming it will be 0
    #this is because large amounts of interpolation may mean the start node will happen at the 1st or 2nd coordinate in the line
    filtered_keys = [key for key in range(4) if key in insert_indices]
    # Sort the filtered keys
    sorted_keys = sorted(filtered_keys)

    #needs to catch case where end node isn't recongnised so there is no point at the end of the line.
    smallest_value = sorted_keys[0] if sorted_keys else None

    if smallest_value is not None:
        ordered_points.append(insert_indices[smallest_value])

    #this should run through the points on the line, starting from one end (which end?) and then adds each substring in order
    #it should output a list of all the points apart from the end one at index i=0, does it?
    if len(points) != len(insert_indices):
        print(f"something missing! points: {len(points)}, insert_ind: {len(insert_indices)}")
    
    for i in range(len(coordinates)-1):
        if i > 0 and i in insert_indices.keys():
            linestrings.append(LineString(coordinates[start_index:i+1]))
            #print(f"i={i}")
            #print(f"insert_indices:{insert_indices[i]}")
            ordered_points.append(insert_indices[i])
            start_index = i
            #print("point added!")
            #print(insert_indices[i])
            #print(f"i:{i}")
    
    linestrings.append(LineString(coordinates[start_index:]))
    
    #print(ordered_points)
    
    outstanding = [z for z in list(points.keys()) if z not in ordered_points]
    if len(outstanding) >= 1:
        ordered_points.append(outstanding[0])
    
    if len(points)!=len(ordered_points):
        print("ordered_points problem!")

    bookends = []
    for i in range(len(ordered_points) - 1):
        bookends.append((ordered_points[i], ordered_points[i+1]))
    #print("subdivided linestrings:")
    #print(linestrings)
    #print("bookends of substring:")
    #print(bookends)
    return linestrings, bookends


def DFS(
        recorder: NetworkData, 
        current_asset: str,
        cursor_conn: sqlite3.Cursor,
        cursor_lv_assets: sqlite3.Cursor,
        cursor_flux: sqlite3.Cursor,
        cursor_graph: sqlite3.Cursor,
        connection_flux: sqlite3.Connection,
        recursion_depth: int,
        max_recursion_depth: int = 5,
        hyper_edges: list = None,
    ) -> None:
    """
    Recursive function to implement a custom
    depth first search to discover all assets 
    downstream of a secondary substation.
    """
    if hyper_edges is None:
        hyper_edges = []
    
    #print(f"current asset: {current_asset}")
    recursion_depth += 1
    if max_recursion_depth:
        if recursion_depth > max_recursion_depth:
            print("max recursion depth reached!")
            return

    # Query the conn table for all connections to this asset
    cursor_conn.execute(
            f"""
            SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = {current_asset} 
            OR fid_to = {current_asset}
            """
        )
    
    rows_conn = cursor_conn.fetchall()

    if rows_conn is None:
        print(f"Nothing found in conn_comp for fid {current_asset}")
        return
    if len(rows_conn) == 0:
        print(f"Nothing found in conn_comp for fid {current_asset}")
        return

    # Query the lv_assets table to find everything out about this asset
    cursor_lv_assets.execute(
        f"""
        SELECT * FROM lv_assets WHERE fid = {current_asset} 
        """
        )
    row_lv = cursor_lv_assets.fetchone()
    
    if row_lv is None:
        print(f"No properties found in LV table for fid {current_asset}")
        return
    if len(row_lv) == 0:
        print(f"No properties found in LV table for fid {current_asset}")
        return
    asset_type = row_lv[-1]

    # All things joining this node
    joining_things = [
        (x[0] if x[1] == current_asset else x[1]) for x in rows_conn
    ]
    
    if asset_type == "node":

        # Eliminate thing that we've already visited
        # or just come from
        new_joining_things = [
            x for x in joining_things if
            x not in recorder.visited_edges and x not in recorder.visited_nodes
        ]

        # Dump this info into the network recorder
        recorder.node_list.append([val for val in row_lv])
        recorder.visited_nodes.append(int(current_asset))

        # Is node a switch or substation?
        is_switch = row_lv[-2]
        is_substation = row_lv[-3]
        if is_substation:
            print("substation reached!")
            return

        if is_switch:
            print(f"switch reached: {current_asset}")
            #Finding all the ways connected to the switch
            #Looking in conn rather than conn_comp because conn_comp removes ways in cleaning step
            cursor_conn.execute(
                f"""
                        SELECT fid_to, fid_from FROM conn_new WHERE fid_from = {current_asset} 
                        OR fid_to = {current_asset}
                        """
            )

            switch_conn = cursor_conn.fetchall()
            # Extract all unique FIDs from the data
            all_fids = set(fid for fid, _ in switch_conn)

            # Filter out the current_fid
            ways = [fid for fid in all_fids if fid != current_asset]

            # Construct the SQL query using the IN clause
            placeholders = ', '.join('?' for _ in ways)
            query = f"SELECT fid,`Normal Status` FROM Way WHERE fid IN ({placeholders})"
            conn_assets_ways = sqlite3.connect('data/assets.gpkg')
            cursor_assets_ways = conn_assets_ways.cursor()
            cursor_assets_ways.execute(query,ways)


            # Fetch all rows
            ways_details = cursor_assets_ways.fetchall()

            closed_way_fids = [fid for fid, status in ways_details if status == 'Closed']

            #close connection to ways
            conn_assets_ways.close()

            cursor_conn.execute(
                f"""
                SELECT fid_to, fid_from 
                FROM conn_new 
                WHERE fid_from IN ({', '.join(map(str, closed_way_fids))}) 
                OR fid_to IN ({', '.join(map(str, closed_way_fids))})
                """
            )

            switch_closed_edges = cursor_conn.fetchall()


            # Flatten the list of paired fids into a single list of all fids
            all_fids = [fid for pair in switch_closed_edges for fid in pair]

            # Exclude fids in way_fids
            fid_without_ways = [fid for fid in all_fids if fid not in closed_way_fids]

            fid_without_ways_or_boundary = [fid for fid in fid_without_ways if fid not in recorder.visited_nodes]
            # # Get the contained features from the "Way" table

            if any(item in recorder.visited_edges for item in fid_without_ways_or_boundary):
                fid_to_map = [fid for fid in fid_without_ways_or_boundary if fid not in recorder.visited_edges]

                print(f"Reached closed switch and continuing to {fid_to_map}")

                for x in fid_to_map:
                    DFS(
                        recorder,
                        x,
                        cursor_conn,
                        cursor_lv_assets,
                        cursor_flux,
                        cursor_graph,
                        connection_flux,
                        recursion_depth,
                        max_recursion_depth,
                        hyper_edges=hyper_edges
                    )

            else:
                print("Switch was open")

            return
        # Recursively call the function on each unvisited thing
        try:
            if len(new_joining_things) > 0:
                for x in new_joining_things:
                    
                    DFS(
                        recorder,
                        x,
                        cursor_conn,
                        cursor_lv_assets,
                        cursor_flux,
                        cursor_graph,
                        connection_flux,
                        recursion_depth,
                        max_recursion_depth,
                        hyper_edges=hyper_edges
                    )
            else:
                #print(f"Nothing joining {current_asset}")
                #print(f"It had these joining things: {joining_things}")
                #print(f"But {vis} were already visisted.")
                #print("gets here")
                return
        except TypeError as e:
            print(e)
            return

    elif asset_type == "edge":
        # If this is a regular edge
        if len(joining_things) == 2: 

            # Check if this edge, or anything with a 
            # name that is a modified version of it 
            # exists in the graph mapped_edges table
            # Convert the target integer to a string
            target_string = str(current_asset)

            # Construct the SQL query with a parameterized query to prevent SQL injection
            query = "SELECT * FROM edge_list WHERE fid LIKE ?;"

            # Append the '%' wildcard to the target string
            target_string_with_wildcard = target_string + '%'

            # Execute the query with the target string as a parameter
            cursor_graph.execute(query, (target_string_with_wildcard,))
            row_graph = cursor_graph.fetchone()
            if row_graph is not None:
                print("new names of hyperedge already exist")
                return


            # cursor_graph.execute(
            #     f"SELECT * FROM edge_list WHERE fid = {int(current_asset)}"
            # )
            # row_graph = cursor_graph.fetchone()
            # if row_graph != None:
            #     return

            # Check if it a flux edge
            # If it is then return
            visited = 0
            cursor_flux.execute(f"SELECT * FROM flux_lines WHERE fid = {int(current_asset)}")
            row_flux = cursor_flux.fetchone()
            if row_flux is not None:
                visited = row_flux[1]
                if visited == 0:
                    cursor_flux.execute(f"UPDATE flux_lines SET Visited = {1} WHERE fid = {current_asset}")
                    connection_flux.commit()
                elif visited == 1:
                    return
            
            # Store all edge info in edge_list 
            recorder.edge_list.append([val for val in row_lv])

            # Add this edge to visited_edges
            recorder.visited_edges.append(int(current_asset))
            
            recorder.incidence_list.append([
                current_asset,
                joining_things[0],
                joining_things[1]
                ])

            # Look up the terminal node that is not the active asset
            # and recurse on it. Pass this edge as the new active
            # asset.
            x = joining_things[0] if joining_things[1] in recorder.visited_nodes else joining_things[1]
            DFS(
                recorder,
                x,
                cursor_conn,
                cursor_lv_assets,
                cursor_flux,
                cursor_graph,
                connection_flux,
                recursion_depth,
                max_recursion_depth,
                hyper_edges=hyper_edges
            )

        # If it is a hyper edge
        elif len(joining_things) > 2:
            
            # Added to ensure that further steps of DFS stop here if encountered again (ADAM)
            recorder.visited_edges.append(int(current_asset))
            
            target_string = str(current_asset)
            
            # Construct the SQL query with a parameterized query to prevent SQL injection
            query = "SELECT * FROM edge_list WHERE fid LIKE ?;"

            # Append the '%' wildcard to the target string
            target_string_with_wildcard = target_string + '%'

            # Execute the query with the target string as a parameter
            cursor_graph.execute(query, (target_string_with_wildcard,))
            row_graph = cursor_graph.fetchone()
            
            if row_graph is not None:
                print("New names of hyper_edge were present in database")
                return

            # Pass the edge's geometry together with this 
            # list of "terminal points" to the divide 
            # linestring function
            # First look up lv_assets to find out everything about this edge
            cursor_lv_assets.execute(
                f"""
                SELECT * FROM lv_assets WHERE fid = {current_asset} 
                """
                )
            row_lv = cursor_lv_assets.fetchone()

            
            edge_geom = wkb.loads(row_lv[1])

            # Now need to look up geom of each terminal point
            points = dict()
            
            
            for point in joining_things:
                try:
                    cursor_lv_assets.execute(
                        f"""
                        SELECT Geometry FROM lv_assets WHERE fid = {point} 
                        """
                    )
                    points[point] = wkb.loads(cursor_lv_assets.fetchone()[0])
                except:
                    print(f"Point {point} not found")
                    points[point] = None
                    # Look up this points geom? 
                    # use cursor conn
                    # cursor_conn.execute(
                    #     f"""
                    #     SELECT * FROM 
                    #     """
                    # )
            
            #filters out any points that were not found, allowing network to continue mapping for nodes that were found instead of exiting completely
            points = {key: value for key, value in points.items() if value is not None}


            #print(f"current_asset = {current_asset}")
            sub_edge_geoms, bookends = add_points_and_divide_linestring(
                edge_geom, points
                )
            # if int(current_asset) == 9144845:
            #     print("bookends:")
            #     print(bookends)
            #     print(f"points: {len(points)}")
            #     print(f"bookends: {len(bookends)}")
            #     print("Look here ^^^^^^^^^^^^^^^^^^^^^^^")
            
            # Now each of these sub edges needs recording.
            # They should inheret the properties of the 
            # original hyper edge
            if len(sub_edge_geoms) != len(bookends):
                del sub_edge_geoms[0]
                #TODO change this so it checks which line reaches the substation centroid point, as its not gauranteed to always be the first line in the list (i dont think)

                # # Create a colormap for different colors
                # colors = range(len(sub_edge_geoms))
                # cmap = plt.get_cmap('viridis')
                # normalize = plt.Normalize(min(colors), max(colors))
                # colors = [cmap(normalize(color)) for color in colors]
                #
                # # Plot each LineString with a different color
                # fig, ax = plt.subplots()
                # for line, color in zip(sub_edge_geoms, colors):
                #     x, y = line.xy
                #     ax.plot(x, y, color=color)
                #
                # # Show the plot
                # plt.show()

            for i, sub_edge_geom in enumerate(sub_edge_geoms):

                # Give unique FID to new sub edge
                sub_edge_fid = str(current_asset) + f"0{i}"
                

                # Check if this edge, or anything with a 
                # name that is a modified version of it 
                # exists in the graph mapped_edges table
                cursor_graph.execute(
                    f"SELECT * FROM edge_list WHERE fid = {int(sub_edge_fid)}"
                )
                row_graph = cursor_graph.fetchone()
                if row_graph != None:
                    print("sub_edge id was already in edge list!")
                    return

                start_node, end_node = bookends[i]
                    
                recorder.incidence_list.append([
                    sub_edge_fid,
                    start_node,
                    end_node
                    ])
                
                # Store all edge info in edge_list 
                sub_edge_row_lv = list(row_lv)
                sub_edge_row_lv[0] = sub_edge_fid
                sub_edge_row_lv[1] = wkb.dumps(sub_edge_geom)
                recorder.edge_list.append([val for val in sub_edge_row_lv])
                #print(recorder.edge_list)
                for node in bookends[i]:
                    # Now call the function recursively 
                    DFS(
                        recorder,
                        node,
                        cursor_conn,
                        cursor_lv_assets,
                        cursor_flux,
                        cursor_graph,
                        connection_flux,
                        recursion_depth,
                        max_recursion_depth,
                        hyper_edges = hyper_edges
                        )
            return

    return

#function to check if an edge is connected to a closed way
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

def map_substation(
        substation_fid: int,
        flux_db_path: str,
        max_recursion_depth: int = 2000
        ) -> NetworkData:

    connection_net = sqlite3.connect('data/network_data.sqlite')
    cursor_net = connection_net.cursor()

    # Find all edge connected to it
    cursor_net.execute(
            f"""
            SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = {substation_fid} OR fid_to = {substation_fid}
            """
            )
    rows = cursor_net.fetchall()

    incident_edges = [
        (x[0] if x[1] == substation_fid else x[1]) for x in rows
    ]
    
    # Populate the network recorder with this data
    net_data = NetworkData()
    net_data.counter = 0
    net_data.visited_nodes.append(substation_fid)
    

    # Set up database connections
    conn_lv_assets = sqlite3.connect('data/lv_assets.sqlite')
    cursor_lv_assets = conn_lv_assets.cursor()

    # Checks if a database already exists for this path. If it doesn't it creates one, if it does it resets it
    if not os.path.exists(flux_db_path):
        create_station_flux_lines_table(flux_db_path)
    if os.path.exists(flux_db_path):
        reset_station_flux_lines_table(flux_db_path)

    connection_flux = sqlite3.connect(flux_db_path)
    cursor_flux = connection_flux.cursor()
    connection_graph = sqlite3.connect('data/graph.sqlite')
    cursor_graph = connection_graph.cursor()

    # Find the substation's geom
    cursor_lv_assets.execute(
            f"""
            SELECT * FROM lv_assets where fid = {substation_fid}
            """
        )
    row_lv = cursor_lv_assets.fetchone()
    # Tests to ensure row_lv loads correctly
    if row_lv is not None:
        net_data.substation_geom = row_lv[1]
    else:
        print("Substation not found in LV assets")

    # check that all incident edges are wires
    #TODO put this check into the substation cleaning function so it doesn't have to be carried out each time
    incident_edges_filter= []
    for edge in incident_edges:
        # Query the lv_assets table to find everything out about this asset
        cursor_lv_assets.execute(
            f"""
                    SELECT * FROM lv_assets WHERE fid = {edge} 
                    """
        )
        edge_details = cursor_lv_assets.fetchone()
        if edge_details is not None:
            if edge_details[-1] == 'edge':
                incident_edges_filter.append(edge)
            else:
                print(f"{edge} was excluded as it is not an edge")
        else:
            print(f"Couldn't find out if {edge} was an edge from LV assets")

    way_status = []
    for edge in incident_edges_filter:
        way_bool = flux_way_check(edge, cursor_net)
        way_status.append(way_bool)
    print(f"way status: {way_status}")

    print(f"Starting depth first search on substation {substation_fid}")
    for e in incident_edges_filter:
        print(f"Searching branch: {e}")
        DFS(net_data, e, cursor_net, cursor_lv_assets,
            cursor_flux, cursor_graph, connection_flux, 0, 
            max_recursion_depth=max_recursion_depth,hyper_edges = None)
        
    net_data.substation = substation_fid
    print("finishes DFS!")
    #print("incident list:")
    #print(net_data.incidence_list)

    # Close down all database connections
    cursor_lv_assets.close()
    conn_lv_assets.close()
    cursor_flux.close()
    cursor_graph.close()
    connection_flux.close()
    cursor_net.close()
    connection_net.close()

    return net_data




def get_background_map_and_bounds(
        centroid: Point,
        image_fname: str,
        dist: int | float,
        road_color: str = "#dbdbdb",
        bkgd_color: str = "#eef9ee",
        bldg_color: str = "#f5d1ab"
        ) -> tuple[float]:
    
    substation_centroid = list(centroid.coords)[0]

    # Get the image
    # The centroid coordinates are in easting-northing
    # Need a function to convert them to latitude longitude
    transformer = Transformer.from_crs('epsg:27700', 'epsg:4326')

    # This is the centre of the substation in lat-lon
    centroid_lat_lon = transformer.transform(*substation_centroid)

    # Get all roads and buildings around centroid

    print("Getting roads and buildings...")

    gdf = ox.features_from_point(
        centroid_lat_lon, {'building': True}, dist=dist
        )
    

    # Plot roads around centroid
    _, ax = ox.plot_figure_ground(point=centroid_lat_lon, dist=dist,network_type="drive", default_width=2, street_widths=None, save=False,show=False, close=True, edge_color=road_color, bgcolor=bkgd_color)
    
    # Plot the building footprints around centroid
    # _, ax = ox.plot_footprints(
    #     gdf, ax=ax, filepath=image_fname, dpi=180, save=True,
    #     show=False, close=True, color=bldg_color, bgcolor=bkgd_color
    #     )
    print("runs this far")
    # Get bounding box of the data (this will be in lat-lon)
    bounds = gdf.total_bounds
    
    # Transform these bounds back into OS style
    min_lon, min_lat, max_lon, max_lat = bounds
    transformer = Transformer.from_crs('epsg:4326', 'epsg:27700', 
                                    always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    del(gdf)
    return min_x, min_y, max_x, max_y
    

def get_subs_in_region(
        centroid: Point,
        radius: int | float
    ) -> list:

    buffered_area = centroid.buffer(radius)
    nearby_subs = []
    with fiona.open('data/assets.gpkg', layer='General Boundary', crs="osgb36") as src:
        for feature in src.filter(bbox=buffered_area.bounds):
            if feature['properties']['Boundary Type'] == 'Substation Curtilage':

                # Get the geometry and ID of the substation
                substation_geom = shape(feature['geometry'])
                substation_fid = feature['properties']['FID']
                nearby_subs.append([substation_geom, substation_fid])
    return nearby_subs


def get_switches_in_region(
        centroid: Point,
        radius: int | float
    ) -> list:

    buffered_area = centroid.buffer(radius)
    nearby_switches = []
    with fiona.open("data/assets.gpkg", layer='Switch') as src:
        for feature in src.filter(bbox=buffered_area.bounds):
            if feature['properties']['Switch Type'] == 'Link Box':

                # Get the geometry and ID of the switch
                switch_centroid = shape(feature['geometry'])
                switch_fid = feature['properties']['FID']
                nearby_switches.append([switch_centroid, switch_fid])
    return nearby_switches