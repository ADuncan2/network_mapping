"""
Wipe all mapping output data so the pipeline can be re-run from scratch.

Clears:
  - results/graph.sqlite      (all tables emptied, WAL truncated)
  - results/summary.csv       (deleted)
  - results/building_matches.csv        (deleted)
  - results/building_match_summary.csv  (deleted)
  - results/temp/              (all files deleted)
  - logs/                      (all .log files deleted)
"""

import os
import glob
import sqlite3


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

GRAPH_DB = os.path.join(RESULTS_DIR, "graph.sqlite")
CSV_FILES = [
    os.path.join(RESULTS_DIR, "summary.csv"),
    os.path.join(RESULTS_DIR, "building_matches.csv"),
    os.path.join(RESULTS_DIR, "building_match_summary.csv"),
]
TEMP_DIR = os.path.join(RESULTS_DIR, "temp")


def clear_graph_db():
    """Empty all tables in graph.sqlite and truncate the WAL."""
    if not os.path.exists(GRAPH_DB):
        print(f"  graph.sqlite not found — skipping")
        return

    conn = sqlite3.connect(GRAPH_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        print(f"  Cleared table '{table}'")

    conn.commit()
    cursor.close()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"  WAL truncated")


def delete_csvs():
    """Delete output CSV files."""
    for path in CSV_FILES:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted {os.path.basename(path)}")


def clear_temp():
    """Delete all files in results/temp/."""
    if not os.path.isdir(TEMP_DIR):
        return
    for f in os.listdir(TEMP_DIR):
        fpath = os.path.join(TEMP_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"  Deleted temp/{f}")


def clear_logs():
    """Delete all .log files in logs/."""
    for f in glob.glob(os.path.join(LOGS_DIR, "*.log")):
        os.remove(f)
        print(f"  Deleted {os.path.basename(f)}")


def main():
    print("This will delete ALL mapping results:\n")
    print("  - Empty all tables in results/graph.sqlite")
    print("  - Delete results/summary.csv")
    print("  - Delete results/building_matches.csv")
    print("  - Delete results/building_match_summary.csv")
    print("  - Delete all files in results/temp/")
    print("  - Delete all log files in logs/")
    print()

    answer = input("Are you sure you want to delete the results? (yes/no): ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    print()
    print("Clearing graph.sqlite...")
    clear_graph_db()

    print("Deleting CSV files...")
    delete_csvs()

    print("Clearing temp directory...")
    clear_temp()

    print("Clearing log files...")
    clear_logs()

    print("\nDone. All mapping outputs have been cleared.")


if __name__ == "__main__":
    main()
