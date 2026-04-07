"""
Large-scale DFS diagnostic script.
Runs the full mapping pipeline on all substations, writing results
incrementally to a SQLite database. Supports resume on restart.

Usage:
    python diagnose_dfs.py              # run all substations (shuffled)
    python diagnose_dfs.py --limit 500  # run at most 500 substations
    python diagnose_dfs.py --db results/my_run.sqlite  # custom output path

Monitor progress from another terminal:
    python -c "
    import sqlite3, sys
    c = sqlite3.connect('results/diagnostic.sqlite')
    for r in c.execute('SELECT status, COUNT(*), ROUND(AVG(t_total),2) FROM results GROUP BY status'):
        print(f'  {r[0]:>15s}: {r[1]:>5d}  avg {r[2]}s')
    total = c.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    print(f'  Total: {total}')
    "
"""

import os
import sys
import time
import random
import sqlite3
import logging
import argparse
import tracemalloc
from datetime import datetime

from gridstock.recorder import NetworkData
from gridstock.mapfuncs import (
    DFS, DFSStats, DFSBudgetExceeded,
    log_stats_of_dfs, flux_way_check,
    ConfigManager, DistributionNetwork,
)
from gridstock.dbcleaning import create_station_flux_lines_table, reset_station_flux_lines_table


# ---------------------------------------------------------------------------
# Results database
# ---------------------------------------------------------------------------

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    fid             INTEGER PRIMARY KEY,
    status          TEXT NOT NULL,
    nodes           INTEGER DEFAULT 0,
    edges           INTEGER DEFAULT 0,
    service_points  INTEGER DEFAULT 0,
    incidence_rows  INTEGER DEFAULT 0,
    total_calls     INTEGER DEFAULT 0,
    node_calls      INTEGER DEFAULT 0,
    edge_calls      INTEGER DEFAULT 0,
    hyper_edge_calls INTEGER DEFAULT 0,
    max_depth       INTEGER DEFAULT 0,
    depth_limit_hits INTEGER DEFAULT 0,
    redundant_node_visits INTEGER DEFAULT 0,
    redundant_edge_visits INTEGER DEFAULT 0,
    both_nodes_visited INTEGER DEFAULT 0,
    flux_returns    INTEGER DEFAULT 0,
    substation_returns INTEGER DEFAULT 0,
    t_total         REAL DEFAULT 0,
    t_setup         REAL DEFAULT 0,
    t_way_check     REAL DEFAULT 0,
    t_dfs           REAL DEFAULT 0,
    t_dfs_stats     REAL DEFAULT 0,
    t_networkx      REAL DEFAULT 0,
    t_pandapower    REAL DEFAULT 0,
    t_simulation    REAL DEFAULT 0,
    peak_memory_mb  REAL DEFAULT 0,
    error           TEXT DEFAULT '',
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event       TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    detail      TEXT DEFAULT ''
);
"""


def init_results_db(db_path):
    """Create or open the results database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.executescript(RESULTS_SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def get_completed_fids(conn):
    """Return set of FIDs already in the results table."""
    rows = conn.execute("SELECT fid FROM results").fetchall()
    return {r[0] for r in rows}


def write_result(conn, row):
    """Insert one result row immediately."""
    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    conn.execute(
        f"INSERT OR REPLACE INTO results ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols]
    )
    conn.commit()


def log_event(conn, event, detail=""):
    """Write a timestamped event to the run_log table."""
    conn.execute(
        "INSERT INTO run_log (event, timestamp, detail) VALUES (?, ?, ?)",
        (event, datetime.now().isoformat(), str(detail)[:500])
    )
    conn.commit()


# ---------------------------------------------------------------------------
# FID loading
# ---------------------------------------------------------------------------

def load_all_substation_fids():
    """Load all substation FIDs from lv_assets."""
    conn = sqlite3.connect('data/lv_assets.sqlite', timeout=120)
    cur = conn.cursor()
    cur.execute("SELECT fid FROM lv_assets WHERE Is_Substation = 1")
    fids = [r[0] for r in cur.fetchall()]
    conn.close()
    return fids


# ---------------------------------------------------------------------------
# LogBatch (minimal, matches single_fid_mapping interface)
# ---------------------------------------------------------------------------

class LogBatch:
    def __init__(self, fid):
        self.fid = fid
        self.process_name = "Diagnostic"
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


# ---------------------------------------------------------------------------
# Full pipeline with per-stage timing
# ---------------------------------------------------------------------------

def run_full_pipeline(substation_fid, flux_db_path, log_batch, stats):
    """Run the full mapping pipeline. Returns (net_data, stats, timings)."""
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
    stats.started_at = time.perf_counter()

    for e in incident_edges_filter:
        DFS(net_data, log_batch, e, cursor_net, cursor_lv,
            cursor_flux, connection_flux, 0,
            max_recursion_depth=2000, hyper_edges=None,
            stats=stats)

    net_data.substation = substation_fid
    net_data.dfs_stats = stats
    timings["t_dfs"] = time.perf_counter() - t0

    # --- Stage 4: DFS stats ---
    t0 = time.perf_counter()
    log_stats_of_dfs(net_data, log_batch)
    timings["t_dfs_stats"] = time.perf_counter() - t0

    # --- Stage 5: NetworkX ---
    t0 = time.perf_counter()
    config = ConfigManager('gridstock/config.yaml')
    pnet = DistributionNetwork(substation_fid, log_batch)
    pnet.get_substation_networkx(net_data)
    timings["t_networkx"] = time.perf_counter() - t0

    # --- Stage 6: Pandapower ---
    t0 = time.perf_counter()
    pnet.create_ppnetwork(config)
    pnet.check_pandapower_network()
    timings["t_pandapower"] = time.perf_counter() - t0

    # --- Stage 7: Simulation ---
    t0 = time.perf_counter()
    pnet.simulate_ppnetwork(config)
    timings["t_simulation"] = time.perf_counter() - t0

    # Clean up
    cursor_lv.close(); conn_lv.close()
    cursor_flux.close(); connection_flux.close()
    cursor_net.close(); connection_net.close()

    return net_data, timings


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results_conn, n_total):
    """Print aggregate summary from the results DB."""
    cur = results_conn.cursor()

    counts = cur.execute(
        "SELECT status, COUNT(*) FROM results GROUP BY status"
    ).fetchall()
    done = sum(c for _, c in counts)

    print(f"\n  Progress: {done}/{n_total} "
          f"({100*done/n_total:.1f}%)" if n_total else "")
    for status, count in counts:
        avg = cur.execute(
            "SELECT ROUND(AVG(t_total), 2) FROM results WHERE status=?",
            (status,)
        ).fetchone()[0]
        print(f"    {status:>15s}: {count:>5d}  avg {avg}s")

    ok_stats = cur.execute("""
        SELECT COUNT(*), ROUND(MIN(t_total),2), ROUND(AVG(t_total),2),
               ROUND(MAX(t_total),2), ROUND(SUM(t_total),1),
               SUM(nodes), ROUND(MAX(peak_memory_mb),1),
               ROUND(AVG(t_dfs),2), ROUND(AVG(t_pandapower),2)
        FROM results WHERE status='ok'
    """).fetchone()

    if ok_stats and ok_stats[0] > 0:
        n, tmin, tavg, tmax, tsum, nodes, mem, dfs_avg, pp_avg = ok_stats
        print(f"    OK timing:  min={tmin}s  avg={tavg}s  max={tmax}s  total={tsum}s")
        print(f"    OK nodes:   {nodes} total across {n} substations")
        print(f"    Peak mem:   {mem} MB")
        print(f"    Avg stage:  DFS={dfs_avg}s  pandapower={pp_avg}s")

    # Show top 5 errors by frequency
    errors = cur.execute("""
        SELECT error, COUNT(*) as cnt FROM results
        WHERE status != 'ok' AND error != ''
        GROUP BY error ORDER BY cnt DESC LIMIT 5
    """).fetchall()
    if errors:
        print(f"    Top errors:")
        for err, cnt in errors:
            print(f"      [{cnt:>3d}x] {err[:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Large-scale DFS diagnostic")
    parser.add_argument("--db", default="results/diagnostic.sqlite",
                        help="Output SQLite path (default: results/diagnostic.sqlite)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max substations to process (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle order")
    parser.add_argument("--summary-every", type=int, default=100,
                        help="Print summary every N substations")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)

    # Load FIDs and shuffle
    print("Loading substation FIDs from lv_assets...")
    all_fids = load_all_substation_fids()
    random.seed(args.seed)
    random.shuffle(all_fids)
    print(f"  Found {len(all_fids)} substations")

    # Open results DB and check for resume
    results_conn = init_results_db(args.db)
    completed = get_completed_fids(results_conn)
    fids_to_run = [f for f in all_fids if f not in completed]

    if args.limit > 0:
        fids_to_run = fids_to_run[:args.limit]

    print(f"  Already completed: {len(completed)}")
    print(f"  Remaining to run:  {len(fids_to_run)}")
    n_total = len(completed) + len(fids_to_run)

    if not fids_to_run:
        print("Nothing to do.")
        print_summary(results_conn, n_total)
        results_conn.close()
        return

    log_event(results_conn, "run_started",
              f"{len(fids_to_run)} substations, limit={args.limit}")

    flux_db_path = os.path.join("data", "temp", "flux_lines_diag.sqlite")

    print(f"\nStarting at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results: {args.db}")
    print("-" * 90)
    sys.stdout.flush()

    run_start = time.perf_counter()

    for i, fid in enumerate(fids_to_run):
        log_batch = LogBatch(fid)
        stats = DFSStats()

        row = {
            "fid": fid, "status": "error",
            "nodes": 0, "edges": 0, "service_points": 0, "incidence_rows": 0,
            "total_calls": 0, "node_calls": 0, "edge_calls": 0,
            "hyper_edge_calls": 0, "max_depth": 0, "depth_limit_hits": 0,
            "redundant_node_visits": 0, "redundant_edge_visits": 0,
            "both_nodes_visited": 0, "flux_returns": 0, "substation_returns": 0,
            "t_total": 0.0, "t_setup": 0.0, "t_way_check": 0.0,
            "t_dfs": 0.0, "t_dfs_stats": 0.0, "t_networkx": 0.0,
            "t_pandapower": 0.0, "t_simulation": 0.0,
            "peak_memory_mb": 0.0, "error": "",
            "completed_at": "",
        }

        tracemalloc.start()
        t0 = time.perf_counter()

        try:
            net_data, timings = run_full_pipeline(fid, flux_db_path, log_batch, stats)
            t_total = time.perf_counter() - t0

            row["status"] = "ok"
            row["nodes"] = len(net_data.node_list)
            row["edges"] = len(net_data.edge_list)
            row["service_points"] = len([n for n in net_data.node_list if n[2] == 'Service Point'])
            row["incidence_rows"] = len(net_data.incidence_list)
            row["t_total"] = round(t_total, 3)
            for k, v in timings.items():
                row[k] = round(v, 3)

            print(f"  [{len(completed)+i+1:>5d}/{n_total}] {fid}  OK  "
                  f"{t_total:>6.2f}s  {row['nodes']:>5d}n  {row['edges']:>5d}e  "
                  f"dfs={timings['t_dfs']:.2f}  pp={timings['t_pandapower']:.2f}")

        except DFSBudgetExceeded:
            t_total = time.perf_counter() - t0
            row["status"] = "budget_exceeded"
            row["t_total"] = round(t_total, 3)
            row["error"] = f"{stats.total_calls} calls, {stats.hyper_edge_calls} hyper"
            print(f"  [{len(completed)+i+1:>5d}/{n_total}] {fid}  BUDGET  "
                  f"{t_total:>6.1f}s  {stats.total_calls} calls")

        except Exception as e:
            t_total = time.perf_counter() - t0
            row["status"] = "error"
            row["t_total"] = round(t_total, 3)
            row["error"] = str(e)[:200]
            print(f"  [{len(completed)+i+1:>5d}/{n_total}] {fid}  ERROR  "
                  f"{t_total:>6.1f}s  {str(e)[:60]}")

        finally:
            try:
                _, peak = tracemalloc.get_traced_memory()
                stats.peak_memory_mb = max(stats.peak_memory_mb, peak / 1024 / 1024)
            except RuntimeError:
                pass
            tracemalloc.stop()

        # Copy DFS stats into row
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
        row["completed_at"] = datetime.now().isoformat()

        # Write immediately
        write_result(results_conn, row)
        sys.stdout.flush()

        # Periodic summary
        if (i + 1) % args.summary_every == 0:
            elapsed = time.perf_counter() - run_start
            rate = (i + 1) / elapsed
            remaining = (len(fids_to_run) - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- After {i+1} substations ({elapsed:.0f}s elapsed, "
                  f"~{remaining/3600:.1f}h remaining) ---")
            print_summary(results_conn, n_total)
            print()
            log_event(results_conn, "checkpoint", f"{i+1} done, {elapsed:.0f}s elapsed")

    # Final summary
    total_elapsed = time.perf_counter() - run_start
    log_event(results_conn, "run_finished",
              f"{len(fids_to_run)} substations in {total_elapsed:.0f}s")

    print(f"\n{'='*90}")
    print(f"FINISHED at {datetime.now().strftime('%H:%M:%S')}  "
          f"({total_elapsed:.0f}s elapsed)")
    print_summary(results_conn, n_total)
    print(f"\nResults: {args.db}")
    print(f"{'='*90}")

    results_conn.close()


if __name__ == "__main__":
    main()
