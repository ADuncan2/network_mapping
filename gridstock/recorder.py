"""
File containing a class used to record data
during depth first searching.

Each NetworkData class instance should be used
to store the data for one distribution network,
i.e. the mapping of a singgle substation.
"""

import csv
import os
import sqlite3

from shapely import wkb
from shapely.geometry import LineString
from shapely.ops import linemerge


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
        self.visited_edges = set()
        self.visited_nodes = set()
        self.incidence_list = []
        self.node_list = []
        self.edge_list = []
        self.summary_stats = {}
        self.substation_fid = None
        self.substation_coord = None

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
            fname: str = "results/graph.sqlite",
            connection: sqlite3.Connection = None,
            ) -> None:
        """
        Write the network data to graph.sqlite using batched inserts.

        If *connection* is provided it is reused (caller owns the lifecycle).
        Otherwise a new connection is opened and closed per call.
        """
        if len(self.edge_list) > 1:

            own_conn = connection is None
            if own_conn:
                connection = sqlite3.connect(fname, timeout=30)
            cursor = connection.cursor()

            if self.substation is not None:
                parent_substation = int(self.substation)
                cursor.execute(
                    "INSERT OR IGNORE INTO mapped_substations (substation_fid) VALUES (?)",
                    (parent_substation,))
            if self.switch is not None:
                parent_switch = int(self.switch)
                cursor.execute(
                    "INSERT OR IGNORE INTO mapped_switches (switch_fid) VALUES (?)",
                    (parent_switch,))

            # Batch insert incidence list
            if self.substation is not None:
                inc_rows = []
                for line, node_from, node_to in self.incidence_list:
                    inc_rows.append((
                        int(line), int(node_from),
                        int(node_to), int(parent_substation)
                    ))
                cursor.executemany(
                    "INSERT OR IGNORE INTO incidence_list "
                    "(edge_fid, node_from, node_to, parent_substation) "
                    "VALUES (?, ?, ?, ?)",
                    inc_rows)
            elif self.switch is not None:
                inc_rows = []
                for line, node_from, node_to in self.incidence_list:
                    inc_rows.append((int(line), int(node_from), int(node_to), int(parent_switch)))
                cursor.executemany(
                    "INSERT OR IGNORE INTO incidence_list "
                    "(edge_fid, node_from, node_to, parent_switch) "
                    "VALUES (?, ?, ?, ?)",
                    inc_rows)

            # Batch insert edges — first 13 columns (fid..Switch_Status)
            cursor.executemany(
                'INSERT OR IGNORE INTO edge_list (fid, Geometry, Asset_Type, Voltage, '
                'Material, Conductors_Per_Phase, Cable_Size, Insulation, '
                'Installation_Date, Phases_Connected, Sleeve_Type, '
                'Associated_Cable, Switch_Status) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [entry[:13] for entry in self.edge_list])

            # Batch insert nodes — same first-13-columns approach
            cursor.executemany(
                'INSERT OR IGNORE INTO node_list (fid, Geometry, Asset_Type, Voltage, '
                'Material, Conductors_Per_Phase, Cable_Size, Insulation, '
                'Installation_Date, Phases_Connected, Sleeve_Type, '
                'Associated_Cable, Switch_Status) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [entry[:13] for entry in self.node_list])

            connection.commit()

            cursor.close()
            if own_conn:
                connection.close()

    def to_csv(self, fname: str = "results/summary.csv") -> None:
        """
        Save the summary_stats dictionary to a CSV as a new row.
        """
        if not hasattr(self, "summary_stats") or not isinstance(self.summary_stats, dict):
            pass

        # Check if the CSV already exists
        file_exists = os.path.exists(fname)

        with open(fname, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.summary_stats)
