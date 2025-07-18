from gridstock.mapfuncs import *
from gridstock.dbcleaning import (
    reset_station_flux_lines_table,
    get_mapped_substations_data
)
from gridstock.plotting import plot_net_data
from gridstock.recorder import NetworkData
#import matplotlib.pyplot as plt

reset_station_flux_lines_table()

print("database reset")

def extract_all_fids(gpkg_file='data/assets.gpkg', layer_name='General Boundary'):
    """
    Extracts a list of all 'fids' from the specified Geopackage file and layer.

    Parameters:
        gpkg_file (str): The path to the Geopackage file.
        layer_name (str): The name of the layer containing the 'fids'.

    Returns:
        list: A list containing all 'fids' found in the specified layer.
    """
    all_fids = []

    with fiona.open(gpkg_file, layer=layer_name, crs="osgb36") as src:
        for feature in src:
            fid = feature['properties'].get('FID')  # Get the 'fid' value from the properties dictionary
            if fid is not None:
                all_fids.append(fid)

    return all_fids

# all_fids_list_raw = extract_all_fids()

# # removing substations that cause as it causes an infinite loop - investigate! (something in the point overlap point)
# problem_subs = [11375446, 11375106, 11375790, 11375939, 11376031]
# all_fids_list = [item for item in all_fids_list_raw if item not in problem_subs]


mapped_subs = get_mapped_substations_data()

print(f"substations mapped: {len(mapped_subs)}")

all_fids_list = [11374885]

for fid in all_fids_list:
    print(f"Current substation: {fid}")
    perc_prog = round(all_fids_list.index(fid) * 100 / len(all_fids_list),2)
    print(f"Percentage progress: {perc_prog}%")

    print("Starting mapping...")

    is_fid_present = any(fid == row[0] for row in mapped_subs)

    if is_fid_present:
        print(f"fid {fid} is already in mapped_substations.")
    else:
        try:
            print("trying")
            nd = map_substation(fid, max_recursion_depth=20000)
            print("mapped")
            nd.to_sql()
            print("saved")
        except Exception as e:
            print(f"An error occurred: {e}")


print("finishes the script")