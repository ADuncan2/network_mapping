from shapely.geometry import Point, LineString, shape, Polygon
import networkx as nx
import pandapower as pp
import os
import sqlite3
from shapely.wkb import loads, dumps


import numpy as np
import pandas as pd

from pyproj import Transformer

import multiprocessing as mp

import logging


class DistributionNetwork:

    def __init__(self,fid,log_batch) -> None:
        """
        Make the path to the network data and empty lists
        to service points, switches, and lamp posts.
        This object is a modifed version of the one used in the network simulation scripts 
        (main difference is getting network data direct from net_data rather than reading it from graph.sqlite)
        """
        self.service_points = []
        self.switches = []
        self.lamp_posts = []

        # Create empty pandapower and networkx networks
        self.net = nx.Graph()
        self.ppnet = pp.create_empty_network()
        self.fid = fid
        self.log_batch = log_batch

    def get_substation_networkx(self, net_data) -> None:
        """
        Function that reads the data stored in NetworkData and then creates
        a networkx graph object. This is a modified version of the 
        """
        

        # get the contextual data for the substation, which is excluded from graph.sql so needs to be sourced from lv_assets.sql
        lv_con = sqlite3.connect(r"data\lv_assets.sqlite")
        lv_curs = lv_con.cursor()
        lv_curs.execute(f"SELECT Geometry, Asset_Type, Voltage, Installation_Date FROM lv_assets WHERE fid = {self.fid}")
        sub_data = lv_curs.fetchone()
        # Substation geometry is a boundary polygon so must be converted to a point to match the other nodes.
        sub_point = loads(sub_data[0]).centroid
        sub_point_wkb = dumps(sub_point)
        sub_data = (sub_point_wkb,)+sub_data[1:]
        
        
        try:
            # Add substation to node_list
            self.net.add_node(
                self.fid,
                geometry=sub_data[0],
                asset_type="Substation",
                voltage="",
                install_date=""
            )
        except Exception as e:
            self.log_batch.add_log(logging.ERROR, f"Could not add substation because: {e}")



        # Get the info for each node and add it to self.net (a networkx graph)
        # and maintain a list of those nodes that are switches and service points
        # Iterate over the FIDs of the nodes that we found in this substation's domain
        

        if net_data.node_list:
            for node in net_data.node_list:
                # print(f"node fid: {node[0]}")
                # If the node is the substation itself then 
                # add it to the graph appropriately
                try:
                    # Add to the graph
                    self.net.add_node(
                        node[0],
                        geometry=node[1],
                        asset_type=node[2],
                        voltage=node[3],
                        install_date=node[4]
                    )
                    
                    # Add the fid of the node to one of the lists 
                    # if the node is a switch or service point
                    if node[2] == "Service Point":
                        self.service_points.append(node[0])
                    if node[2] == "Switch":
                        self.switches.append(node[0])
                except Exception as e:
                    self.log_batch.add_log(logging.WARNING, f"Could not add a node in the node_list to networkx format because: {e}")

        # # Get the number of nodes. This will be useful for checking against later
        # # to ensure that we don't accidentally add extra nodes
        # num_nodes_after_node_initialisation = self.net.number_of_nodes()

        
        # Iterate over the edges in the incidence list
        if net_data.edge_list:
            for row in net_data.edge_list:

                # Calculate the edge's length
                geom = loads(row[1])
                length = geom.length

                for incident_row in net_data.incidence_list:
                    edge_fid_inc, node_to_inc, node_from_inc = incident_row
                    if edge_fid_inc == row[0]:
                        edge_fid = edge_fid_inc
                        node_to = node_to_inc
                        node_from = node_from_inc


                # Add the edge to the graph
                self.net.add_edge(
                    node_from,
                    node_to,
                    edge_fid=edge_fid,
                    length=length,
                    geometry=geom,
                    asset_type=row[2],
                    voltage=row[3],
                    cable_size=row[6]
                )
        
        # # Check that adding edges did not add extra nodes
        # nodes_after_edges_added = self.net.number_of_nodes() 
        # if  nodes_after_edges_added != num_nodes_after_node_initialisation:
        #     print(f"num_of_nodes: {nodes_after_edges_added}")
        #     print(f"initial_num_of_nodes: {num_nodes_after_node_initialisation}")
        #     raise Warning("Spurious nodes added during edge creation.")
        
        # Check the nodes are all accounted for; i.e. check that all nodes of 
        # degree 1 are either the substation, a service point, or a switch.
        # If they are not, then, for now, label them as a lamp post!
        for node in self.net.nodes:
            deg = self.net.degree[node]
            if deg == 0:
                self.log_batch.add_log(logging.WARNING, f"One of the nodes is not connected to anything")
            elif deg == 1:
                # If degree 1, check it is accounted for
                if node not in self.service_points and node not in self.switches and node != self.fid:
                    self.lamp_posts.append(node)

        # Check that the grid is connected
        if not nx.is_connected(self.net):
            self.log_batch.add_log(logging.ERROR, f"The networkx graph created is not connected")


    def create_ppnetwork(self, config) -> None:
        """
        Function to turn the networkx graph into 
        a pandapower network
        """
        # Get the geometry info for the substation
        # create the transformer to transform the projection to one that pandapower can plot
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4277", always_xy=True)
        
        #first get the data of the substation node with fid == self.fid
        target_tuple = next((node_tuple for node_tuple in self.net.nodes(data=True) if node_tuple[0] == self.fid), None)
        #load the geometry from the bits
        sub_geom = loads(target_tuple[1]["geometry"])
        longitude, latitude = transformer.transform(sub_geom.x, sub_geom.y)
        #put it in a format that pandapower will understand
        sub_geom_coord = (longitude, latitude)

        # Make the external grid and transformer associated
        # with the fid of the substation. First we create the
        # external grid bus

        self.external_grid = pp.create_bus(self.ppnet, vn_kv=11., name="grid", geodata= sub_geom_coord)

        # Then attach a pandapower external grid object to that bus
        pp.create_ext_grid(
            self.ppnet, bus=self.external_grid, vm_pu=config.get("create_ppnetwork","ext_grid_params", "vm_pu"), name="grid connection"
            )
        
        # Now create a bus for the substation and add it to the bus list
        # so that it is the 0th bus in the list
        pp.create_bus(self.ppnet, vn_kv=0.4, name=self.fid,geodata= sub_geom_coord)

        # Now specify that transformer settings
        pp.create_transformer_from_parameters(
            self.ppnet,
            hv_bus= config.get("create_ppnetwork","transformer_parameters", "hv_bus"),
            lv_bus=config.get("create_ppnetwork","transformer_parameters", "lv_bus"),
            sn_mva=config.get("create_ppnetwork","transformer_parameters", "sn_mva"),
            vn_hv_kv=config.get("create_ppnetwork","transformer_parameters", "vn_hv_kv"),
            vn_lv_kv=config.get("create_ppnetwork","transformer_parameters", "vn_lv_kv"),
            vkr_percent=config.get("create_ppnetwork","transformer_parameters", "vkr_percent"),
            vk_percent=config.get("create_ppnetwork","transformer_parameters", "vk_percent"),
            pfe_kw=config.get("create_ppnetwork","transformer_parameters", "pfe_kw"),
            i0_percent=config.get("create_ppnetwork","transformer_parameters", "i0_percent"),
            name="Trafo"
        )

        # Need to maintain a map between bus pp bus numbers and node fids
        # The substation bus will have label 1. This is because the 0th bus
        # is the external grid as that was added first. The external grid 
        # has no corresponding entity in the network data
        bus_num = 1
        bus_num_dict = {self.fid: bus_num}
        self.bus_num_dict = bus_num_dict
        


        # Go through all of the nodes in the networkx graph
        # and add corresponding buses in pandapower
        for node_id, data in self.net.nodes(data=True):
            
            # The substation node has already been dealt with, so
            # we skip it here. Recall that self.fid is the fid 
            # of the substation

            if node_id != self.fid:
                try:
                    # Keep the relationship between the bus fid and the number that
                    # pandapower will store it as
                    bus_num += 1
                    bus_num_dict[node_id] = bus_num

                    # Add this bus to the pandapower network and handle any cases with missing
                    # data by assuming that they are 400V.

                    # read the geometry from wkb format and convert into (x,y) format required for pandapower geomdata
                    geom_data = loads(data["geometry"])
                    # Using the transformer defined above to convert projections into one compatible with pandapower
                    longitude, latitude = transformer.transform(geom_data.x, geom_data.y)
                    coordinates = (longitude, latitude)

                    ## Create a bus in the pandapower network
                    pp.create_bus(self.ppnet, vn_kv=0.4, name=node_id, geodata= coordinates)
                except Exception as e:
                    self.log_batch.add_log(logging.WARNING, f"Node could not be made into pandpower bus because: {e}")
        
        # Go through all of the edges in the networkx graph
        # and add in the lines in pandapower. Also maintain a dictionary mapping
        # the edge fids to the line numbers in pandapower
        self.edge_num_dict = {}
        edge_num = 0
        

        # Take each edge and assign it the closest coresponding pandapower data type.
        for u, v, data in self.net.edges(data=True):
            # Get the cross section data in its raw form
            cross_sec = data['cable_size']

            # Account for the fact that cross section is stored as a mm2 or in2 string and we want it as a numeric for calculation
            if "n2" in cross_sec:
                cross_sec = float(cross_sec.replace("in2", ""))
                cross_sec_mm = cross_sec*645.16
            elif not any(char.isdigit() for char in cross_sec):
                ## BIG ASSUMPTION HERE, any cables that are not given a cross section are assumed to be 300mm2, is this fair??
                cross_sec_mm = config.get("create_ppnetwork","default_cross_section")
            else:
                cross_sec = float(cross_sec.replace("mm2", ""))
                cross_sec_mm = cross_sec
            
            # BIG ASSUMPTION HERE: assumes that all the wires are aluminium and uses the resistivity of Al to match
            resistivity_of_al = config.get("create_ppnetwork","resistivity_of_al") #ohms m


            res_per_km = resistivity_of_al*1000*1000*1000/cross_sec_mm


            # Read the cleaned_cables.csv file (can be found on pandapower/doc/std_types/linetypes.csv on the pandapower github)
            cables_df = pd.read_csv(os.path.join("data", "cleaned_cables.csv"))

            # Find the 'Cable Name' with the 'Resistance per km' closest to res_per_km
            closest_cable = cables_df.iloc[(cables_df['Resistance per km'] - res_per_km).abs().argmin()]['Cable Name']

            # Assign the closest cable name to the edge data
            data['std_type'] = closest_cable
        
        # # Summarize the different std_types and their frequencies
        # std_type_counts = {}
        # for _, _, data in self.net.edges(data=True):
        #     std_type = data.get("std_type", "Unknown")
        #     if std_type in std_type_counts:
        #         std_type_counts[std_type] += 1
        #     else:
        #         std_type_counts[std_type] = 1

        # # Print the std_type counts
        # for std_type, count in std_type_counts.items():
        #     print(f"Standard Type: {std_type}, Count: {count}")

        for u, v, data in self.net.edges(data=True):
            try:
                pp.create_line(self.ppnet, from_bus=bus_num_dict[u], to_bus=bus_num_dict[v], length_km=data["length"]/1000.0, name=data["edge_fid"], std_type=data["std_type"]) 
                self.edge_num_dict[data["edge_fid"]] = edge_num
                edge_num += 1
            except Exception as e:
                self.log_batch.add_log(logging.WARNING, f"Could not add an edge to pandapower because: {e}")
        # Got through service points and lamp posts and add loads
        # NOTE: this puts the loads as 1kW by default, ready to be scaled by whatever load profile is applied. q_mvar is just guessed at this stage, as is lampost load size
        for service_point in self.service_points:
            try:
                pp.create_load(self.ppnet, bus=bus_num_dict[service_point], p_mw=0.0, q_mvar=config.get("create_ppnetwork","default_reactive_power"), name=service_point)
            except Exception as e:
                self.log_batch.add_log(logging.WARNING, f"Could not add an service point to pandapower because: {e}")
            # pp.create_storage(self.ppnet, bus=bus_num_dict[service_point], p_mw = -0.003,max_e_mwh = 0.013, soc_percent = 1.0, min_e_mwh = 0, controllable = False)
        for lamp_post in self.lamp_posts:
            try:
                pp.create_load(self.ppnet, bus_num_dict[lamp_post], p_mw=config.get("create_ppnetwork", "lamp_post_load_size", "p_mw"), q_mvar=config.get("create_ppnetwork", "lamp_post_load_size", "q_mvar"), name=lamp_post)
            except Exception as e:
                self.log_batch.add_log(logging.WARNING, f"Could not add an lamp post to pandapower because: {e}")
        # Set all indices to be fids
        self.ppnet.load.set_index('name', inplace=True)







