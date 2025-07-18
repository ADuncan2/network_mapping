import sqlite3

# Define the file path to the SQLite database
fname = "data/network_data.sqlite"  # Replace with your actual file path

try:
    # Open a connection to the SQLite database
    connection = sqlite3.connect(fname)

    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()

    # Define the SQL query to select the 'FID' values from the 'Switch' table
    switch_query = "SELECT FID FROM Switch WHERE switch_type = 'Link Box'"

    # Execute the SQL query to get 'FID' values from the 'Switch' table
    cursor.execute(switch_query)
    switch_fid_values = cursor.fetchall()

    # Close the cursor and the database connection
    cursor.close()
    connection.close()

    # Check if 'switch_fid_values' is not empty
    if switch_fid_values:

        first_10_switch_fid_values = [fid[0] for fid in switch_fid_values[:10]]
        # Open a new connection to the SQLite database
        connection = sqlite3.connect(fname)
        cursor = connection.cursor()

        # Iterate over each switch FID and filter the 'conn' table
        for switch_fid in first_10_switch_fid_values:
            # Initialize the 'all_fids' list for this switch
            all_fids = []

            # Define the SQL query to filter 'conn' table for the current switch
            conn_query = f"SELECT fid_from,fid_to FROM conn WHERE fid_from = {switch_fid} OR fid_to = {switch_fid}"

            # Execute the SQL query for the current switch
            cursor.execute(conn_query)
            filtered_conn_rows = cursor.fetchall()

            # Add the 'FID' values to the 'all_fids' list
            all_fids.extend([row[0] for row in filtered_conn_rows] + [row[1] for row in filtered_conn_rows])

            print("FIDs for switch ",switch_fid,": ",all_fids)

            ## perform a DFS of 1 level to get the FIDs of all wires connected to each Way
            # Query the 'conn' table again to retrieve rows with matching 'FID' values in 'all_fids'
            expanded_conn_query = f"SELECT * FROM conn WHERE fid_from IN ({','.join(map(str, all_fids))}) OR fid_to IN ({','.join(map(str, all_fids))} )"
            cursor.execute(expanded_conn_query)
            expanded_conn_rows = cursor.fetchall()

            for row in expanded_conn_rows:
                print(row)
        # Close the cursor and the database connection
        cursor.close()
        connection.close()
    else:
        print("No 'Switch' FID values found.")

except sqlite3.Error as e:
    print("SQLite error:", e)


