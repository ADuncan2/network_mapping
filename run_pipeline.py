"""
Production pipeline: runs network mapping and building matching
for all ~40,000 substations in ENWL.

Both stages have built-in resume support — safe to interrupt and restart.

Prerequisites:
    - cleaning_script.py has been run (one-time)
    - data/ folder contains lv_assets.sqlite, network_data.sqlite,
      assets.gpkg, all_buildings_in_enwl_27700_bbox.parquet

Usage:
    uv run python run_pipeline.py              # both stages
    uv run python run_pipeline.py --mapping    # stage 1 only
    uv run python run_pipeline.py --matching   # stage 2 only
"""

import argparse
import sqlite3
import time
from datetime import timedelta


def print_banner():
    print()
    print("=" * 60)
    print("  GRIDSTOCK — Full Production Pipeline")
    print("=" * 60)

    conn = sqlite3.connect("data/lv_assets.sqlite", timeout=120)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM lv_assets WHERE Is_Substation = 1")
    total = cur.fetchone()[0]
    conn.close()

    conn = sqlite3.connect("results/graph.sqlite", timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM mapped_substations")
    mapped = cur.fetchone()[0]
    try:
        cur.execute("SELECT COUNT(DISTINCT substation_fid) FROM building_matches")
        matched = cur.fetchone()[0]
    except sqlite3.OperationalError:
        matched = 0
    conn.close()

    print(f"  Total substations:     {total:,}")
    print(f"  Already mapped:        {mapped:,}")
    print(f"  Already matched:       {matched:,}")
    print("=" * 60)
    print()


def run_stage_1():
    """Stage 1: Network mapping (multiprocessing DFS)."""
    print("=" * 60)
    print("  STAGE 1 — Network Mapping")
    print("=" * 60)

    t0 = time.perf_counter()

    import single_fid_mapping
    single_fid_mapping.main()

    elapsed = time.perf_counter() - t0
    print(f"\nStage 1 completed in {timedelta(seconds=int(elapsed))}")
    return elapsed


def run_stage_2():
    """Stage 2: Building matching (multiprocessing spatial join)."""
    print()
    print("=" * 60)
    print("  STAGE 2 — Building Matching")
    print("=" * 60)

    t0 = time.perf_counter()

    import building_matching
    building_matching.main()

    elapsed = time.perf_counter() - t0
    print(f"\nStage 2 completed in {timedelta(seconds=int(elapsed))}")
    return elapsed


def print_summary(t_mapping, t_matching):
    total = (t_mapping or 0) + (t_matching or 0)

    conn = sqlite3.connect("results/graph.sqlite", timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM mapped_substations")
    mapped = cur.fetchone()[0]
    for table in ["incidence_list", "edge_list", "node_list"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count:,} rows")
    try:
        cur.execute("SELECT COUNT(DISTINCT substation_fid) FROM building_matches")
        matched = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM building_matches")
        match_rows = cur.fetchone()[0]
        print(f"  building_matches: {match_rows:,} rows ({matched:,} substations)")
    except sqlite3.OperationalError:
        matched = 0
    conn.close()

    import os
    db_size = os.path.getsize("results/graph.sqlite") / 1024 / 1024

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Substations mapped:    {mapped:,}")
    print(f"  Substations matched:   {matched:,}")
    print(f"  graph.sqlite size:     {db_size:.1f} MB")
    if t_mapping is not None:
        print(f"  Stage 1 time:          {timedelta(seconds=int(t_mapping))}")
    if t_matching is not None:
        print(f"  Stage 2 time:          {timedelta(seconds=int(t_matching))}")
    print(f"  Total time:            {timedelta(seconds=int(total))}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Run the Gridstock production pipeline.")
    parser.add_argument("--mapping", action="store_true", help="Run stage 1 (network mapping) only")
    parser.add_argument("--matching", action="store_true", help="Run stage 2 (building matching) only")
    args = parser.parse_args()

    # If neither flag is set, run both stages
    run_map = args.mapping or not (args.mapping or args.matching)
    run_match = args.matching or not (args.mapping or args.matching)

    print_banner()

    t_mapping = None
    t_matching = None

    if run_map:
        t_mapping = run_stage_1()

    if run_match:
        t_matching = run_stage_2()

    print_summary(t_mapping, t_matching)


if __name__ == "__main__":
    main()
