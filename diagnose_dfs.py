"""
DFS diagnostic script.
Runs problem substations + control substations with full instrumentation
to identify root causes of memory blowup and runaway DFS.

Usage:
    python diagnose_dfs.py
"""

import os
import time
import csv
import sqlite3
import logging
import tracemalloc
from datetime import datetime
from collections import Counter

from gridstock.recorder import NetworkData
from gridstock.mapfuncs import DFS, DFSStats, DFSBudgetExceeded
from gridstock.dbcleaning import create_station_flux_lines_table, reset_station_flux_lines_table

# Known problem substations (hardcoded exclusion list + benchmark crasher)
PROBLEM_SUBS = [11375446, 11375106, 11375790, 11375939, 11376031, 20641038]

# Known-good substations for comparison (small / medium / large)
CONTROL_SUBS = [11374876, 11377484, 11381596]

# Benchmark substations (13 ground-mounted + 5 PMT)
BENCHMARK_SUBS = [
    11375741, 11376677, 11377424, 11377527, 11381369, 11381596,
    11381873, 11383667, 11384273, 11389576, 11390927,
    20622972, 20625444, 20630799, 20642174, 68518209,
]

# Random substations for wider coverage
RANDOM_SUBS = [11382294, 20645582, 20635558, 11384262, 20654077, 20633799]


class LogBatch:
    """Minimal LogBatch for diagnostics."""
    def __init__(self, fid):
        self.fid = fid
        self.messages = []
    def add_log(self, level, message):
        self.messages.append({'level': level, 'message': message})


def dfs_only_instrumented(substation_fid, flux_db_path, log_batch, stats, max_recursion_depth=2000, verbose=True):
    """Run DFS with full instrumentation. Returns NetworkData."""
    connection_net = sqlite3.connect('data/network_data.sqlite', timeout=120)
    cursor_net = connection_net.cursor()

    cursor_net.execute(
        f"SELECT fid_to, fid_from FROM conn_comp "
        f"WHERE fid_from = {substation_fid} OR fid_to = {substation_fid}")
    rows = cursor_net.fetchall()
    incident_edges = [(x[0] if x[1] == substation_fid else x[1]) for x in rows]

    net_data = NetworkData()
    net_data.counter = 0
    net_data.visited_nodes.add(substation_fid)

    conn_lv = sqlite3.connect('data/lv_assets.sqlite', timeout=120)
    cursor_lv = conn_lv.cursor()

    if not os.path.exists(flux_db_path):
        create_station_flux_lines_table(flux_db_path)
    if os.path.exists(flux_db_path):
        reset_station_flux_lines_table(flux_db_path)

    connection_flux = sqlite3.connect(flux_db_path)
    cursor_flux = connection_flux.cursor()

    # Get substation geom
    cursor_lv.execute(f"SELECT * FROM lv_assets where fid = {substation_fid}")
    row_lv = cursor_lv.fetchone()
    if row_lv is not None:
        net_data.substation_geom = row_lv[1]

    # Filter to wire edges only
    incident_edges_filter = []
    for edge in incident_edges:
        cursor_lv.execute(f"SELECT * FROM lv_assets WHERE fid = {edge}")
        edge_details = cursor_lv.fetchone()
        if edge_details is not None and edge_details[-1] == 'edge':
            incident_edges_filter.append(edge)

    # Run DFS with stats
    for e in incident_edges_filter:
        DFS(net_data, log_batch, e, cursor_net, cursor_lv,
            cursor_flux, connection_flux, 0,
            max_recursion_depth=max_recursion_depth, hyper_edges=None,
            stats=stats, verbose=verbose)

    net_data.substation = substation_fid

    # Clean up
    cursor_lv.close(); conn_lv.close()
    cursor_flux.close(); connection_flux.close()
    cursor_net.close(); connection_net.close()

    return net_data


def format_branching(branching_list):
    """Summarise hyper edge branching as a distribution string."""
    if not branching_list:
        return "none"
    counts = Counter(branching_list)
    return ", ".join(f"{k}-way: {v}" for k, v in sorted(counts.items()))


def print_report(results):
    """Print human-readable diagnostic report."""
    print("\n" + "=" * 90)
    print(f"DFS DIAGNOSTIC REPORT  --  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Group results
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    for group_name, group_results in groups.items():
        print(f"\n--- {group_name.upper()} ({len(group_results)}) ---\n")
        for r in group_results:
            print(f"  FID {r['fid']}: {r['status'].upper()} in {r['t_total']:.2f}s")
            print(f"    Calls: {r['total_calls']} total "
                  f"({r['node_calls']} node, {r['edge_calls']} edge, "
                  f"{r['hyper_edge_calls']} hyper)")
            if r['status'] != 'error':
                print(f"    Network: {r['nodes']} nodes, {r['edges']} edges, "
                      f"{r['service_points']} SPs")
                print(f"    Max depth: {r['max_depth']}, "
                      f"depth limit hits: {r['depth_limit_hits']}")
            print(f"    Memory: {r['peak_memory_mb']:.1f} MB peak")
            if r['redundant_node_visits'] > 0 or r['both_nodes_visited'] > 0:
                print(f"    Redundant: {r['redundant_node_visits']} node re-entries, "
                      f"{r['redundant_edge_visits']} edge re-entries, "
                      f"{r['both_nodes_visited']} both-visited")
            if r['hyper_edge_calls'] > 0:
                print(f"    Hyper edges: {r['hyper_edge_calls']} "
                      f"({format_branching(r['branching_raw'])})")
            if r['error']:
                print(f"    Error: {r['error']}")
            print()

    # Aggregate summary
    ok = [r for r in results if r['status'] == 'ok']
    errs = [r for r in results if r['status'] != 'ok']

    print("=" * 90)
    print(f"SUMMARY: {len(ok)}/{len(results)} ok, {len(errs)} failed")
    if ok:
        times = [r['t_total'] for r in ok]
        nodes = [r['nodes'] for r in ok]
        calls = [r['total_calls'] for r in ok]
        mems = [r['peak_memory_mb'] for r in ok]
        print(f"  Time:   min={min(times):.2f}  median={sorted(times)[len(times)//2]:.2f}  "
              f"max={max(times):.2f}  total={sum(times):.1f}s")
        print(f"  Nodes:  min={min(nodes)}  median={sorted(nodes)[len(nodes)//2]}  "
              f"max={max(nodes)}  total={sum(nodes)}")
        print(f"  Calls:  min={min(calls)}  median={sorted(calls)[len(calls)//2]}  "
              f"max={max(calls)}  total={sum(calls)}")
        print(f"  Memory: min={min(mems):.1f}  median={sorted(mems)[len(mems)//2]:.1f}  "
              f"max={max(mems):.1f} MB")
    if errs:
        print(f"  Failures:")
        for r in errs:
            print(f"    {r['fid']} ({r['group']}): {r['status']} - {r['error'][:80]}")
    print("=" * 90)


def run_diagnostic():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/diagnostic_{timestamp}.csv"
    flux_db_path = os.path.join("data", "temp", "flux_lines_diag.sqlite")

    all_fids = ([(fid, "control") for fid in CONTROL_SUBS] +
                [(fid, "problem") for fid in PROBLEM_SUBS] +
                [(fid, "benchmark") for fid in BENCHMARK_SUBS] +
                [(fid, "random") for fid in RANDOM_SUBS])

    results = []
    print(f"DFS Diagnostic: {len(all_fids)} substations "
          f"({len(CONTROL_SUBS)} control, {len(PROBLEM_SUBS)} problem, "
          f"{len(BENCHMARK_SUBS)} benchmark, {len(RANDOM_SUBS)} random)")
    print(f"Budget: 50,000 calls / 120s per substation")
    print("-" * 90)

    for i, (fid, group) in enumerate(all_fids):
        log_batch = LogBatch(fid)
        stats = DFSStats()
        stats.started_at = time.perf_counter()

        row = {
            "fid": fid, "group": group, "status": "error",
            "nodes": 0, "edges": 0, "service_points": 0, "incidence_rows": 0,
            "t_total": 0.0,
            "total_calls": 0, "node_calls": 0, "edge_calls": 0,
            "hyper_edge_calls": 0, "max_depth": 0, "depth_limit_hits": 0,
            "redundant_node_visits": 0, "redundant_edge_visits": 0,
            "both_nodes_visited": 0, "flux_returns": 0, "substation_returns": 0,
            "peak_memory_mb": 0.0, "branching_raw": [], "error": "",
        }

        tracemalloc.start()
        try:
            t0 = time.perf_counter()
            net_data = dfs_only_instrumented(fid, flux_db_path, log_batch, stats, verbose=False)
            t_total = time.perf_counter() - t0

            row["status"] = "ok"
            row["nodes"] = len(net_data.node_list)
            row["edges"] = len(net_data.edge_list)
            row["service_points"] = len([n for n in net_data.node_list if n[2] == 'Service Point'])
            row["incidence_rows"] = len(net_data.incidence_list)
            row["t_total"] = round(t_total, 3)

            tag = f"{row['nodes']:>5d} nodes  {row['edges']:>5d} edges  {t_total:.2f}s"
            print(f"  [{i+1:2d}/{len(all_fids)}] {fid} ({group:>7s})  OK  {tag}")

        except DFSBudgetExceeded:
            t_total = time.perf_counter() - t0
            row["status"] = "budget_exceeded"
            row["t_total"] = round(t_total, 3)
            print(f"  [{i+1:2d}/{len(all_fids)}] {fid} ({group:>7s})  "
                  f"BUDGET EXCEEDED  {stats.total_calls} calls  {t_total:.1f}s  "
                  f"{stats.hyper_edge_calls} hyper")

        except Exception as e:
            t_total = time.perf_counter() - t0
            row["status"] = "error"
            row["t_total"] = round(t_total, 3)
            row["error"] = str(e)[:200]
            print(f"  [{i+1:2d}/{len(all_fids)}] {fid} ({group:>7s})  ERROR: {e}")

        finally:
            # Capture final memory snapshot
            try:
                _, peak = tracemalloc.get_traced_memory()
                stats.peak_memory_mb = max(stats.peak_memory_mb, peak / 1024 / 1024)
            except RuntimeError:
                pass
            tracemalloc.stop()

        # Copy stats into row
        row["total_calls"] = stats.total_calls
        row["node_calls"] = stats.node_calls
        row["edge_calls"] = stats.edge_calls
        row["hyper_edge_calls"] = stats.hyper_edge_calls
        row["max_depth"] = stats.max_depth_seen
        row["depth_limit_hits"] = stats.depth_limit_hits
        row["redundant_node_visits"] = stats.redundant_node_visits
        row["redundant_edge_visits"] = stats.redundant_edge_visits
        row["both_nodes_visited"] = stats.both_nodes_visited
        row["flux_returns"] = stats.flux_returns
        row["substation_returns"] = stats.substation_returns
        row["peak_memory_mb"] = round(stats.peak_memory_mb, 1)
        row["branching_raw"] = stats.hyper_edge_branching

        results.append(row)

    # Write CSV (exclude branching_raw list, add summary string instead)
    csv_fields = [
        "fid", "group", "status", "nodes", "edges", "service_points",
        "incidence_rows", "t_total", "total_calls", "node_calls", "edge_calls",
        "hyper_edge_calls", "max_depth", "depth_limit_hits",
        "redundant_node_visits", "redundant_edge_visits", "both_nodes_visited",
        "flux_returns", "substation_returns", "peak_memory_mb",
        "branching_summary", "error",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            r["branching_summary"] = format_branching(r["branching_raw"])
            writer.writerow(r)

    print(f"\nCSV: {out_path}")

    # Print full report
    print_report(results)


if __name__ == "__main__":
    run_diagnostic()
