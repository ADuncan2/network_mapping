import sqlite3

def creating_new_conn() -> None:
    # Database file name and connection
    db_file = "data/network_data.sqlite"  # Replace with the actual database file path
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    # Table names
    source_table_name = "conn"
    new_table_name = "conn_new"

    try:
        # Drop the new table if it already exists
        print("Creating conn_new table")
        cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")

        # Create a new table with the same schema as the source table
        cursor.execute(f"CREATE TABLE {new_table_name} AS SELECT * FROM {source_table_name}")

        # Add an index to all columns in the new table
        cursor.execute(f"PRAGMA table_info({new_table_name})")
        print("table created")
        columns = [column[1] for column in cursor.fetchall()]
        print("Beginning indexing...")
        for column in columns:
            cursor.execute(f"CREATE INDEX idx_{new_table_name}_{column} ON {new_table_name}({column})")

        # Commit the changes
        connection.commit()

        print(f"Table '{new_table_name}' created as a clone of '{source_table_name}' with indexes added.")

    except sqlite3.Error as e:
        print("SQLite error:", e)

    finally:
        # Close the database connection
        connection.close()

