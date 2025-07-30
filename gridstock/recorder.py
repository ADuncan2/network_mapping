"""
File containing a class used to record data
during depth first searching.

Each NetworkData class instance should be used
to store the data for one distribution network,
i.e. the mapping of a singgle substation.
"""

import sqlite3
from shapely import wkb
from shapely.ops import linemerge
from shapely.geometry import LineString
import csv
import os


class NetworkData:
    """
    Parameters
    ----------
    counter : int
        The number of times the depth first search
        function has been called.

    substation : int
        The FID of the substation that we're searching.

    visited_edges : list[int]
        List of edges visited by DFS.

    visited_nodes : list[int]
        List of nodes visted by DFS.

    incidence_list : list[list[Any]]
        The incidence list of the network so far. Contains
        Each element in the list is itself a list with four 
        entries: the ID of the edge, the IDs of the to and
        from nodes, and the ID of the parent substation.

    node_list : list[list[Any]]
        List whose rows contain a list containing all the 
        data grabed from assets pertaining to each node.
    
    edge_list : list[list[Any]]
        List whose rows contain a list containing all the 
        data grabed from assets pertaining to each edge.

    summary_stats : dict[str, Any]
        Dictionary containing summary statistics of the network for saving as summary CSV
    """
    def __init__(self) -> None:
        self.counter = 0
        self.substation = None
        self.switch = None
        self.substation_geom = None
        self.visited_edges = []
        self.visited_nodes = []
        self.incidence_list = []
        self.node_list = []
        self.edge_list = []
        self.summary_stats = {}

    def __str__(self) -> str:
        msg = f"""
        NetworkData recorder object containing:
        {len(self.node_list)} nodes, {len(self.edge_list)} edges and {len(self.substation)} substations.
        """
        return msg
    
    def modify_edge(
            self,
            edge_fid: str,
            new_geom: LineString,
            new_edge_fid: str,
            new_terminus: str
            ) -> None:
        """
        Function to merge lines that have a 
        line-line connection with no intermmediate 
        node into a single line.
        """
        
        # Update line geometry
        for edge_row in self.edge_list:
            if edge_row[0] == edge_fid:

                # Merge edge geometries
                edge_geom = wkb.loads(edge_row[1])
                merged_line = linemerge([edge_geom, new_geom])
                if merged_line.geom_type == "LineString":
                    edge_row[1] = wkb.dumps(merged_line)
                else:
                    raise TypeError("Could not merge lines.")
            break
        
        # Modify incidence
        for row in self.incidence_list:
            if row[0] == edge_fid:
                if row[1] == new_edge_fid:
                    row[2] = new_terminus
                else:
                    row[1] = new_terminus

    def to_sql(
            self,
            fname: str = "data/graph.sqlite"
            ) -> None:
        """
        Function to write the network data 
        to the sql file.
        """
        #print("Full edge list:")
        #print(self.edge_list)

        if len(self.edge_list) > 1:
        
            connection = sqlite3.connect(fname)
            cursor = connection.cursor() 
            connection_lv = sqlite3.connect("data/lv_assets.sqlite")
            cursor_lv = connection_lv.cursor()

            if self.substation != None:
                parent_substation = int(self.substation)
                cursor.execute("INSERT INTO mapped_substations (substation_fid) VALUES (?)", (parent_substation,))
            if self.switch != None:
                parent_switch = int(self.switch)
                cursor.execute("INSERT INTO mapped_switches (switch_fid) VALUES (?)", (parent_switch,))


            for entry in self.incidence_list:
                line, node_from, node_to = entry
                #print(line)
                try:
                    if self.substation != None:
                        cursor.execute("INSERT INTO incidence_list (edge_fid, node_from, node_to, parent_substation) VALUES (?, ?, ?, ?)", (int(line), int(node_from), int(node_to), int(parent_substation)))
                    if self.switch != None:
                        cursor.execute("INSERT INTO incidence_list (edge_fid, node_from, node_to, parent_switch) VALUES (?, ?, ?, ?)", (int(line), int(node_from), int(node_to), int(parent_switch)))
                except sqlite3.IntegrityError as e:
                    #print("incident list error:")
                    #print("IntegrityError:", e)
                    pass

            # Now need to insert into edge_list by copying from self.edge_list
            for entry in self.edge_list:
                fid, wkb_geom, asset_type, voltage, material, conductors_per_phase, cable_size, insulation, install_date, phases_connected, sleeve_type, associated_cable, switch_status, is_substation, is_switch, cat = entry
                #print(fid)
                try:
                    cursor.execute('INSERT INTO edge_list (fid, Geometry, Asset_Type, Voltage, Material, Conductors_Per_Phase, Cable_Size, Insulation, Installation_Date, Phases_Connected, Sleeve_Type, Associated_Cable, Switch_Status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (fid, wkb_geom, asset_type, voltage, material, conductors_per_phase, cable_size, insulation, install_date, phases_connected, sleeve_type, associated_cable, switch_status))
                    #print(fid)
                except sqlite3.Error as e:
                    #print("SQLite Error:", e)
                    #print("Error saving node to node_list in graph.sql")
                    pass
                
                connection.commit()

            # Now need to insert into node_list by copying from self.node_list
            for entry in self.node_list:
                fid, wkb_geom, asset_type, voltage, material, conductors_per_phase, cable_size, insulation, install_date, phases_connected, sleeve_type, associated_cable, switch_status, is_substation, is_switch, cat = entry

                try:
                    cursor.execute('INSERT INTO node_list (fid, Geometry, Asset_Type, Voltage, Material, Conductors_Per_Phase, Cable_Size, Insulation, Installation_Date, Phases_Connected, Sleeve_Type, Associated_Cable, Switch_Status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (fid, wkb_geom, asset_type, voltage, material, conductors_per_phase, cable_size, insulation, install_date, phases_connected, sleeve_type, associated_cable, switch_status))
                except sqlite3.IntegrityError as e:
                    #print("error saving node to node_list in graph.sql")
                    #print("IntegrityError:", e)
                    pass
                connection.commit()

            cursor.close()
            connection.close()
            cursor_lv.close()
            connection_lv.close()

    def to_csv(self, fname: str = "data/summary.csv") -> None:
        """
        Save the summary_stats dictionary to a CSV as a new row.
        """
        if not hasattr(self, "summary_stats") or not isinstance(self.summary_stats, dict):
            pass

        # Check if the CSV already exists
        file_exists = os.path.exists(fname)

        with open(fname, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.summary_stats)

