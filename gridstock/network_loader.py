"""
Load mapped networks from graph.sqlite into NetworkX graphs.

Used by building_matching.py to reconstruct network topology
for spatial matching of service points to buildings.
"""

import networkx as nx
import sqlite3
from shapely.wkb import loads, dumps


class MappedNetwork:
    """A mapped distribution network loaded from graph.sqlite as a NetworkX graph."""

    def __init__(self) -> None:
        self.service_points = []
        self.switches = []
        self.lamp_posts = []
        self.net = nx.Graph()

    def load_from_sqlite(self, fid: int, sql_fname="results/graph.sqlite") -> None:
        """
        Read a substation's mapped network from graph.sqlite and
        build a NetworkX graph with node/edge attributes.
        """
        self.fid = fid

        connection = sqlite3.connect(sql_fname)
        cursor = connection.cursor()

        # Get all unique nodes from the incidence list
        cursor.execute(f"SELECT DISTINCT node_from FROM incidence_list WHERE parent_substation = {fid}")
        unique_node_from_rows = cursor.fetchall()
        cursor.execute(f"SELECT DISTINCT node_to FROM incidence_list WHERE parent_substation = {fid}")
        unique_node_to_rows = cursor.fetchall()

        set_node_from = set([row[0] for row in unique_node_from_rows])
        set_node_to = set([row[0] for row in unique_node_to_rows])
        nodes = set_node_from.union(set_node_to)

        # Get substation data from lv_assets (excluded from graph.sqlite)
        lv_con = sqlite3.connect(r"data\lv_assets.sqlite")
        lv_curs = lv_con.cursor()
        lv_curs.execute(f"SELECT Geometry, Asset_Type, Voltage, Installation_Date FROM lv_assets WHERE fid = {fid}")
        sub_data = lv_curs.fetchone()
        sub_point = loads(sub_data[0]).centroid
        sub_point_wkb = dumps(sub_point)
        sub_data = (sub_point_wkb,) + sub_data[1:]

        # Add nodes to the NetworkX graph
        for node in nodes:
            cursor.execute(f"SELECT Geometry, Asset_Type, Voltage, Installation_Date FROM node_list WHERE fid = {node}")
            node_data = cursor.fetchone()

            try:
                if node == fid:
                    self.net.add_node(
                        node,
                        geometry=sub_data[0],
                        asset_type="Substation",
                        voltage="",
                        install_date=""
                    )
                else:
                    self.net.add_node(
                        node,
                        geometry=node_data[0],
                        asset_type=node_data[1],
                        voltage=node_data[2],
                        install_date=node_data[3]
                    )

                    if node_data[1] == "Service Point":
                        self.service_points.append(node)
                    if node_data[1] == "Switch":
                        self.switches.append(node)
            except Exception:
                pass  # node missing from node_list — skipped

        num_nodes_after_node_initialisation = self.net.number_of_nodes()

        # Add edges from the incidence list
        cursor.execute(f"SELECT edge_fid, node_from, node_to FROM incidence_list WHERE parent_substation = {fid}")
        incidence_rows = cursor.fetchall()

        for row in incidence_rows:
            edge_fid, node_from, node_to = row

            cursor.execute(f"SELECT Geometry, Asset_Type, Voltage, Cable_Size FROM edge_list WHERE fid = {edge_fid}")
            edge_data = cursor.fetchone()

            geom = loads(edge_data[0])
            length = geom.length

            self.net.add_edge(
                node_from,
                node_to,
                edge_fid=edge_fid,
                length=length,
                geometry=edge_data[0],
                asset_type=edge_data[1],
                voltage=edge_data[2],
                cable_size=edge_data[3]
            )

        # Check for spurious nodes added by edges
        nodes_after_edges_added = self.net.number_of_nodes()
        if nodes_after_edges_added != num_nodes_after_node_initialisation:
            self._spurious_nodes = nodes_after_edges_added - num_nodes_after_node_initialisation

        cursor.close()
        connection.close()
        lv_curs.close()
        lv_con.close()

        # Classify degree-1 nodes not already accounted for as lamp posts
        for node in self.net.nodes:
            deg = self.net.degree[node]
            if deg == 0:
                raise Warning(f"Node {node} is isolated!")
            elif deg == 1:
                if node not in self.service_points and node not in self.switches and node != fid:
                    self.lamp_posts.append(node)
