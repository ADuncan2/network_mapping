def collapse_switch_boxes(
        fname: str
    ):
    """
    Function to simplify switch boxes into 
    single nodes.
    """

    # Get the switch geometries
    switches = []
    with fiona.open(fname, layer='Switch') as src:
        for feature in src:
            if feature['properties']['Switch Type'] == 'Link Box':
                centroid = shape(feature['geometry'])
                theta = feature['properties']['style_icon_rotation']
                box = square_from_point(centroid, float(theta))
                results = [
                    str(feature['properties']['FID']),
                    box
                    ]
                switches.append(results)

    # Open the SQLite database connection
    database_path = 'data/network_data.sqlite'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # # Get the contained features from the "Way" table
    with fiona.open(fname, layer='Way') as src2:

        ll = len(switches)
        for i, switch in enumerate(switches):

            print(f"{i} of switch {ll}...")

            switch_fid, switch_geom = switch

            buffered_geometry = switch_geom.buffer(5)
            filtered_features = src2.filter(bbox=buffered_geometry.bounds)

            for f in filtered_features:
                f_fid = str(f['properties']['FID'])

                cursor.execute(f"UPDATE conn_comp SET fid_to = {switch_fid} WHERE fid_to = {f_fid}")

                # Update rows where fid_from is equal to f_fid
                cursor.execute(f"UPDATE conn_comp SET fid_from = {switch_fid} WHERE fid_from = {f_fid}")
                connection.commit()

    # Delete rows where both fid_to and fid_from are equal to fid_switch
    cursor.execute("DELETE FROM conn_comp WHERE fid_to = fid_from")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()
