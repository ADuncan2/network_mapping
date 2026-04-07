"""
Benchmark script for the mapping pipeline.
Measures: DB setup, way check, DFS, stats logging, NetworkX graph.
Run before and after code changes to compare performance.

Usage:
    python benchmark_dfs.py [label]

    label: optional tag for the output file (e.g. "pre" or "post")
           Output goes to results/benchmark_{label}_{timestamp}.csv
"""

import sys
import os
import time
import csv
import sqlite3
import logging
from datetime import datetime

from gridstock.mapfuncs import (
    DFSStats, DFSBudgetExceeded,
    log_stats_of_dfs, flux_way_check, DFS,
)
from gridstock.recorder import NetworkData
from gridstock.network_loader import MappedNetwork
from gridstock.dbcleaning import create_station_flux_lines_table, reset_station_flux_lines_table

# 40 substations: 18 benchmark + 6 former problem + 16 random
BENCHMARK_FIDS = [
    # Original benchmark set (18)
    11374876, 11375741, 11376677, 11377424, 11377484, 11377527,
    11381369, 11381596, 11381873, 11383667, 11384273, 11389576,
    11390927, 20622972, 20625444, 20630799, 20642174, 68518209,
    # Former problem substations (6)
    11375446, 11375106, 11375790, 11375939, 11376031, 20641038,
    # Random substations (16)
    11382294, 20645582, 20635558, 11384262, 20654077, 20633799,
    11382980, 11396242, 11397095, 20639212, 11382676, 11382065,
    20652781, 11388297, 11393910, 20627691,
]


class LogBatch:
    """Minimal LogBatch for benchmark (matches single_fid_mapping.py interface)."""
    def __init__(self, fid):
        self.fid = fid
        self.process_name = "Benchmark"
        self.messages = []
        self.start_time = datetime.now()
        self.end_time = None

    def add_log(self, level, message):
        self.messages.append({'level': level, 'message': message})

    def finalize(self):
        self.end_time = datetime.now()

    def get_duration(self):
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time


def benchmark_pipeline(substation_fid, flux_db_path, log_batch):
    """Run the mapping pipeline with per-stage timing.

    Returns (net_data, dfs_stats, timings).
    """
    timings = {}

    # --- Stage 1: DB setup & edge filtering ---
    t0 = time.perf_counter()

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

    cursor_lv.execute(f"SELECT * FROM lv_assets where fid = {substation_fid}")
    row_lv = cursor_lv.fetchone()
    if row_lv is not None:
        net_data.substation_geom = row_lv[1]

    incident_edges_filter = []
    for edge in incident_edges:
        cursor_lv.execute(f"SELECT * FROM lv_assets WHERE fid = {edge}")
        edge_details = cursor_lv.fetchone()
        if edge_details is not None and edge_details[-1] == 'edge':
            incident_edges_filter.append(edge)

    timings["t_setup"] = time.perf_counter() - t0

    # --- Stage 2: Way check ---
    t0 = time.perf_counter()
    for edge in incident_edges_filter:
        flux_way_check(edge, cursor_net)
    timings["t_way_check"] = time.perf_counter() - t0

    # --- Stage 3: DFS ---
    t0 = time.perf_counter()
    dfs_stats = DFSStats()
    dfs_stats.started_at = time.perf_counter()

    for e in incident_edges_filter:
        DFS(net_data, log_batch, e, cursor_net, cursor_lv,
            cursor_flux, connection_flux, 0,
            max_recursion_depth=2000, hyper_edges=None,
            stats=dfs_stats)

    net_data.substation = substation_fid
    net_data.dfs_stats = dfs_stats
    timings["t_dfs"] = time.perf_counter() - t0

    # --- Stage 4: DFS stats logging ---
    t0 = time.perf_counter()
    log_stats_of_dfs(net_data, log_batch)
    timings["t_dfs_stats"] = time.perf_counter() - t0

    # Clean up DB connections used by DFS
    cursor_lv.close(); conn_lv.close()
    cursor_flux.close(); connection_flux.close()
    cursor_net.close(); connection_net.close()

    # --- Stage 5: Write to graph.sqlite and reload as NetworkX ---
    # This tests the full round-trip: DFS results → SQLite → NetworkX
    t0 = time.perf_counter()
    net_data.to_sql("data/temp/bench_graph.sqlite")
    mapped = MappedNetwork()
    mapped.load_from_sqlite(substation_fid, "data/temp/bench_graph.sqlite")
    timings["t_networkx"] = time.perf_counter() - t0

    return net_data, dfs_stats, timings


def run_benchmark(label=""):
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)

    # Create a temp graph.sqlite for NetworkX round-trip test
    from gridstock.dbcleaning import create_graph_db
    if os.path.exists("data/temp/bench_graph.sqlite"):
        os.remove("data/temp/bench_graph.sqlite")
    create_graph_db("data/temp/bench_graph.sqlite")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{label}" if label else ""
    out_path = f"results/benchmark{tag}_{timestamp}.csv"

    flux_db_path = os.path.join("data", "temp", "flux_lines_bench.sqlite")

    stage_names = ["t_setup", "t_way_check", "t_dfs", "t_dfs_stats", "t_networkx"]

    results = []
    print(f"Mapping benchmark: {len(BENCHMARK_FIDS)} substations")
    print(f"Output: {out_path}")
    print("-" * 90)
    print(f"{'#':>3}  {'FID':>10}  {'Status':>7}  {'Total':>6}  "
          f"{'Setup':>5}  {'Ways':>5}  {'DFS':>6}  {'Stats':>5}  "
          f"{'NX':>5}  {'Nodes':>5}  {'Edges':>5}")
    print("-" * 90)

    t_run_start = time.perf_counter()

    for i, fid in enumerate(BENCHMARK_FIDS):
        log_batch = LogBatch(fid)
        row = {"fid": fid, "status": "error", "nodes": 0, "edges": 0,
               "service_points": 0, "total_dfs_calls": 0,
               "hyper_edge_count": 0, "max_depth": 0,
               "t_total": 0.0, "error": ""}
        for s in stage_names:
            row[s] = 0.0

        try:
            t0 = time.perf_counter()
            net_data, dfs_stats, timings = benchmark_pipeline(
                fid, flux_db_path, log_batch)
            t_total = time.perf_counter() - t0

            row["status"] = "ok"
            row["t_total"] = round(t_total, 3)
            row["nodes"] = len(net_data.node_list)
            row["edges"] = len(net_data.edge_list)
            row["service_points"] = len([n for n in net_data.node_list if n[2] == 'Service Point'])
            row["total_dfs_calls"] = dfs_stats.total_calls
            row["hyper_edge_count"] = dfs_stats.hyper_edge_calls
            row["max_depth"] = dfs_stats.max_depth_seen
            for s in stage_names:
                row[s] = round(timings.get(s, 0), 3)

            print(f"{i+1:3d}  {fid:>10d}  {'ok':>7s}  {t_total:>6.2f}  "
                  f"{timings['t_setup']:>5.2f}  {timings['t_way_check']:>5.2f}  "
                  f"{timings['t_dfs']:>6.2f}  {timings['t_dfs_stats']:>5.2f}  "
                  f"{timings['t_networkx']:>5.2f}  "
                  f"{row['nodes']:>5d}  {row['edges']:>5d}")

        except DFSBudgetExceeded as e:
            t_total = time.perf_counter() - t0
            row["t_total"] = round(t_total, 3)
            row["status"] = "budget"
            row["total_dfs_calls"] = e.stats.total_calls
            row["error"] = str(e)[:200]
            print(f"{i+1:3d}  {fid:>10d}  {'budget':>7s}  {t_total:>6.1f}  BUDGET EXCEEDED")

        except Exception as e:
            t_total = time.perf_counter() - t0
            row["t_total"] = round(t_total, 3)
            row["error"] = str(e)[:200]
            print(f"{i+1:3d}  {fid:>10d}  {'error':>7s}  {t_total:>6.1f}  {str(e)[:80]}")

        results.append(row)
        sys.stdout.flush()

    t_run_total = time.perf_counter() - t_run_start

    # Write CSV
    fieldnames = ["fid", "status", "nodes", "edges", "service_points",
                  "total_dfs_calls", "hyper_edge_count", "max_depth",
                  "t_total"] + stage_names + ["error"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    errs = [r for r in results if r["status"] != "ok"]

    print("-" * 90)
    print(f"\nSUMMARY: {len(ok)}/{len(results)} ok, {len(errs)} errors, "
          f"wall time {t_run_total:.1f}s")

    if ok:
        print(f"\n  Stage totals (ok only, {len(ok)} substations):")
        for s in stage_names:
            vals = [r[s] for r in ok]
            label_s = s.replace("t_", "")
            print(f"    {label_s:>12s}:  total={sum(vals):>7.1f}s  "
                  f"mean={sum(vals)/len(vals):>5.2f}s  "
                  f"median={sorted(vals)[len(vals)//2]:>5.2f}s  "
                  f"max={max(vals):>6.2f}s")

        total_times = [r["t_total"] for r in ok]
        print(f"\n  Overall timing:")
        print(f"    min={min(total_times):.2f}s  "
              f"median={sorted(total_times)[len(total_times)//2]:.2f}s  "
              f"max={max(total_times):.2f}s  "
              f"sum={sum(total_times):.1f}s")

    if errs:
        print(f"\n  Errors:")
        for r in errs:
            print(f"    {r['fid']}: {r['status']} - {r['error'][:80]}")

    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    run_benchmark(label)
