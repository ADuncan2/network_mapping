"""
End-to-end pipeline test: runs network mapping and building matching
for 10 random substations, writing results to a dedicated test output
directory so the real results/ folder is untouched.
"""

import time
import random
import sqlite3
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gridstock.mapfuncs import map_substation
from gridstock.dbcleaning import create_graph_db
from building_matching import process_fid

# Re-use LogBatch from the mapping script
from single_fid_mapping import LogBatch

# ── Configuration ──────────────────────────────────────────────────
N_SUBSTATIONS = 50
SEED = 42                         # Set to None for truly random
TEST_DIR = Path("results/test")
TEST_GRAPH_DB = TEST_DIR / "graph.sqlite"
TEST_SUMMARY_CSV = TEST_DIR / "summary.csv"
TEST_MATCHES_CSV = TEST_DIR / "building_matches.csv"
TEST_MATCH_SUMMARY = TEST_DIR / "building_match_summary.csv"
FLUX_DB_PATH = TEST_DIR / "flux_lines_test.sqlite"


def select_random_fids(n=N_SUBSTATIONS, seed=SEED):
    """Pick n random substation FIDs from lv_assets."""
    conn = sqlite3.connect("data/lv_assets.sqlite", timeout=120)
    cur = conn.cursor()
    # Fetch a small pool from the DB directly to avoid loading all FIDs
    pool_size = max(n, 100)
    cur.execute(
        f"SELECT fid FROM lv_assets WHERE Is_Substation = 1 "
        f"ORDER BY RANDOM() LIMIT {pool_size}"
    )
    pool = [r[0] for r in cur.fetchall()]
    conn.close()

    if seed is not None:
        random.seed(seed)
    return random.sample(pool, min(n, len(pool)))


def setup_test_outputs():
    """Create test output directory and a fresh graph.sqlite."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs from previous test runs
    for f in [TEST_GRAPH_DB, TEST_SUMMARY_CSV, TEST_MATCHES_CSV, TEST_MATCH_SUMMARY]:
        if f.exists():
            f.unlink()
    # Also clean WAL/SHM if present
    for suffix in ["-wal", "-shm"]:
        wal = Path(str(TEST_GRAPH_DB) + suffix)
        if wal.exists():
            wal.unlink()

    create_graph_db(str(TEST_GRAPH_DB))
    print(f"Created test database at {TEST_GRAPH_DB}")


def run_mapping(fids):
    """
    Stage 1: Network mapping via DFS for each substation.
    Returns a list of result dicts.
    """
    results = []

    for fid in tqdm(fids, desc="Stage 1 — Network mapping"):
        log_batch = LogBatch(fid, "test-runner")
        log_batch.add_log(logging.INFO, "Starting mapping")

        t0 = time.perf_counter()
        try:
            net_data, log_batch = map_substation(
                fid, str(FLUX_DB_PATH), log_batch
            )
            elapsed = round(time.perf_counter() - t0, 2)

            # Write to test graph.sqlite and summary.csv
            net_data.to_sql(str(TEST_GRAPH_DB))
            if hasattr(net_data, "summary_stats") and net_data.summary_stats:
                net_data.to_csv(str(TEST_SUMMARY_CSV))

            stats = net_data.summary_stats
            results.append({
                "fid": fid,
                "status": "success",
                "time_s": elapsed,
                "nodes": stats.get("nodes", 0),
                "edges": stats.get("edges", 0),
                "service_points": stats.get("service_points", 0),
                "dfs_calls": stats.get("dfs_calls", 0),
                "budget_exceeded": stats.get("budget_exceeded", False),
            })

        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            results.append({
                "fid": fid,
                "status": "error",
                "time_s": elapsed,
                "error": str(e),
            })
            tqdm.write(f"  ERROR mapping FID {fid}: {e}")

    return results


def run_building_matching(fids):
    """
    Stage 2: Building matching for each successfully mapped substation.
    Returns a list of result dicts.
    """
    results = []

    for fid in tqdm(fids, desc="Stage 2 — Building matching"):
        result = process_fid(
            fid,
            sql_fname=str(TEST_GRAPH_DB),
            matches_path=str(TEST_MATCHES_CSV),
        )
        results.append(result)

        if result["status"] == "failed":
            tqdm.write(f"  ERROR matching FID {fid}: {result.get('error')}")

    return results


def print_report(fids, mapping_results, matching_results, total_time):
    """Print a concise summary table."""
    print("\n" + "=" * 80)
    print(f"PIPELINE TEST REPORT — {len(fids)} substations")
    print(f"Total wall-clock time: {total_time:.1f}s")
    print("=" * 80)

    # Merge mapping and matching results by fid
    map_by_fid = {r["fid"]: r for r in mapping_results}
    match_by_fid = {r["fid"]: r for r in matching_results}

    header = (
        f"{'FID':>12}  {'Map':>5}  {'Time':>6}  {'Nodes':>6}  {'Edges':>6}  "
        f"{'SvcPts':>6}  {'Match':>5}  {'Rate':>6}  {'Bldgs':>6}"
    )
    print(header)
    print("-" * len(header))

    map_ok = map_fail = match_ok = match_fail = 0

    for fid in fids:
        mr = map_by_fid.get(fid, {})
        br = match_by_fid.get(fid, {})

        map_status = mr.get("status", "—")
        map_time = f"{mr.get('time_s', 0):.1f}s"
        nodes = mr.get("nodes", "—")
        edges = mr.get("edges", "—")
        svc = mr.get("service_points", "—")

        match_status = br.get("status", "—")
        match_rate = f"{br.get('match_rate', 0):.0%}" if br.get("status") == "success" else "—"
        bldgs = br.get("buildings_matched", "—") if br.get("status") == "success" else "—"

        map_icon = "OK" if map_status == "success" else "FAIL"
        match_icon = "OK" if match_status == "success" else "FAIL"

        if map_status == "success":
            map_ok += 1
        else:
            map_fail += 1
        if match_status == "success":
            match_ok += 1
        elif match_status == "failed":
            match_fail += 1

        print(
            f"{fid:>12}  {map_icon:>5}  {map_time:>6}  {nodes:>6}  {edges:>6}  "
            f"{svc:>6}  {match_icon:>5}  {match_rate:>6}  {bldgs:>6}"
        )

    print("-" * len(header))
    print(
        f"Mapping:  {map_ok} OK / {map_fail} failed  |  "
        f"Matching: {match_ok} OK / {match_fail} failed"
    )
    print(f"\nOutputs written to {TEST_DIR}/")
    print(f"  graph.sqlite          — mapped network topology")
    print(f"  summary.csv           — per-substation mapping stats")
    print(f"  building_matches.csv  — per-service-point match results")
    print()


def main():
    t_start = time.perf_counter()

    # 1. Pick random substations
    fids = select_random_fids()
    print(f"Selected {len(fids)} random substations: {fids}")

    # 2. Prepare test output directory
    setup_test_outputs()

    # 3. Stage 1 — Network mapping
    print()
    mapping_results = run_mapping(fids)

    # Identify successfully mapped FIDs for stage 2
    mapped_fids = [r["fid"] for r in mapping_results if r["status"] == "success"]
    print(f"\nMapped {len(mapped_fids)}/{len(fids)} substations successfully.")

    # 4. Stage 2 — Building matching (only for successfully mapped)
    matching_results = []
    if mapped_fids:
        print()
        matching_results = run_building_matching(mapped_fids)

    # 5. Report
    total_time = time.perf_counter() - t_start
    print_report(fids, mapping_results, matching_results, total_time)

    # 6. Save combined summary
    combined = []
    map_by_fid = {r["fid"]: r for r in mapping_results}
    match_by_fid = {r["fid"]: r for r in matching_results}
    for fid in fids:
        row = {"fid": fid}
        row.update({f"map_{k}": v for k, v in map_by_fid.get(fid, {}).items() if k != "fid"})
        row.update({f"match_{k}": v for k, v in match_by_fid.get(fid, {}).items() if k != "fid"})
        combined.append(row)
    pd.DataFrame(combined).to_csv(str(TEST_MATCH_SUMMARY), index=False)


if __name__ == "__main__":
    main()
