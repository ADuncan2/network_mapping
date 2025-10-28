import sqlite3

with sqlite3.connect("data/graph.sqlite") as conn:
    cursor = conn.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    print(f"Journal mode is: {mode}")
