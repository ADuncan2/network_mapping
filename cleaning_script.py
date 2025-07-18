"""
This script should produce a database called 
graph.sqlite. It will contain an incident list table, 
where the first column (the index) is the edge id. The 
second and third are the IDs of the two terminal nodes
of the edge. The fourth column is the ID of the substation
underneath which this edge exists. Another table in this
file is called mapped_substations, which is simply a list
of substation IDs that have been mapped. E.g. you could iterate
over this list to get the IDs of mapped substations, and then 
filter the incidence_list table by its fourth column to get 
only the incidence data for each substation's distribution
network. Finally there is an edge_list and node_list table
which contains data about each mapped edge and node. It has
the fields fid, Geometry, Asset_Type, Voltage, Material,
Conductors_Per_Phase, Cable_Size, Insulation, 
Installation_Date, Phases_Connecected, Sleeve_Type, 
Associated_Cable.

This script works by first simplifying substations by
altering the conn table in the network_data.sqlite file
so that substation boundaries appear as a node and all
lv lines crossing the boundary from the substation are
adjusted so that they begin at the substation node rather 
than at whatever component inside the substation. This 
stage also creates another temporary file called flux_lines.sqlite
that records a list of edges that are directly incident on 
substations. This will be of use later in the depth-first-search
stage in order to determine when a substation is located and
whether we've been there before.

The script also simplifies switch boxes and service points
in a similar manner to substations. 

The script then creates a temporary database file called 
lv_assets. This is just a subset of the original assets.gpkg 
that contains only things of interest to this lv 
network mapping. This done so that things can be quickly queried 
during the depth first search.

Then, a new, empty database called graph.sqlite is created. 
This will be populated with the data from the depth
first search. This is the final output file.

the above steps are preprocessing. They only need to be 
run once (hopefully). Once they've been run the lines in this 
script can be commented out down to the substation_fid = 11391343
line.

The script then moves on to the depth first search. To begin, 
a data container obect called net_data is created. This is 
an instance of the NetworkData class, as defined in 
gridstock/recorder.py. As the recursive search goes on, 
things will be saved in this object (it is passed down within the 
function). At the end of the search, you can call the 
NetworkData class' to_sql() method to dump the recorded
network data into the graph.sqlite database.

This script then picks an arbitrary substation and does the 
depth first search on it. One can then write code to iterate
over all substation curtilages in the general boundary data
in assets and map all of the them in the same way. Each time, after
completion of the search, do to_sql().

Once the graph.sqlite database is formed, networkx graphs can be
formed by simply creating an empty graph and doing 
add_edge(node1, node2, extra edge properties = ...) for
each edge, where node1 and node2 are the node_from and 
node_to values from graph.sqlite's incidence_list table
and the edge's ID and other properties can be entered by looking
up the edges properties in the node list table. Node properties
can also be added simply.
"""

import sqlite3
from gridstock.dbcleaning import *
from gridstock.recorder import NetworkData
from gridstock.mapfuncs import DFS
from gridstock.plotting import plot_net_data
from gridstock.creating_conn_new import creating_new_conn

# # # Creating conn_new tables with indexing for computation of following steps
# print("Creating and indexing conn_new SQL table...")
# creating_new_conn()
# print("done")

# # # Simplify switch boxes
# print("Simplifiying switch boxes...")
# collapse_switch_boxes("data/assets.gpkg")
# print("done")

# # Simplify service points
# print("Simplifiying service points...")
# merge_service_points("data/assets.gpkg")
# print("done")

# # Simplify substations
# print("Simplifying substations...")
# simplify_substations()
# print("done")

# # Set up the LV sql file - this is a temporary
# print("Creating data temporary base...")
# create_lv_db()
# print("done")

# # Create the data base to store the results in.
# # This will be called graph.sqlite.
# # This will contain an incidence list and list of 
# # node and edge properties
# print("Creating new data base...")
# create_graph_db()
# print("done")

# Create flux_lines.sqlite table 
print("Creating flux_lines database...")
create_station_flux_lines_table()
print("done")

# # Pick a substation and find its initial edges
# substation_fid = 11391343 # test substation
# connection_net = sqlite3.connect('data/network_data.sqlite')
# cursor_net = connection_net.cursor()

# # Find all edge connected to it
# cursor_net.execute(
#         f"""
#         SELECT fid_to, fid_from FROM conn_comp WHERE fid_from = {substation_fid} OR fid_to = {substation_fid}
#         """
#         )
# rows = cursor_net.fetchall()
# incident_edges = [
#     (x[0] if x[1] == substation_fid else x[1]) for x in rows
# ]

# # Populate the network recorder with this data
# net_data = NetworkData()
# net_data.counter = 0
# net_data.visited_nodes.append(substation_fid)
# net_data.substation = substation_fid

# reset_station_flux_lines_table()

# # Start the depth first search
# conn_lv_assets = sqlite3.connect('data/lv_assets.sqlite')
# cursor_lv_assets = conn_lv_assets.cursor()
# connection_flux = sqlite3.connect('data/flux_lines.sqlite')
# cursor_flux = connection_flux.cursor()

# # Find the substation's geom
# cursor_lv_assets.execute(
#         f"""
#         SELECT * FROM lv_assets where fid = {substation_fid}
#         """
#     )
# row_lv = cursor_lv_assets.fetchone()
# net_data.substation_geom = row_lv[1]

# print("Starting depth first search...")
# for e in incident_edges:
#     print(e)
#     DFS(net_data, e, cursor_net, cursor_lv_assets,
#         cursor_flux, connection_flux, 0, 
#         max_recursion_depth=20000)


# # Close down all database connections
# cursor_lv_assets.close()
# conn_lv_assets.close()
# cursor_flux.close()
# connection_flux.close()
# cursor_net.close()
# connection_net.close()

# # Plot the results
# fig, ax = plt.subplots(figsize=(10,10))
# plot_net_data(net_data, ax, alpha=0.5)
# plt.show()

# # Optionally call net_data.to_sql() to save 
# # the results to the graph.sqlite file.


