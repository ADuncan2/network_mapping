from gridstock.mapfuncs import *
from gridstock.dbcleaning import (
    reset_station_flux_lines_table,
    get_mapped_substations_data
)
from gridstock.plotting import plot_net_data
from gridstock.recorder import NetworkData

from multiprocessing import Pool, Queue, Process, cpu_count, Event
from tqdm import tqdm
import os
from queue import Empty


#reset_station_flux_lines_table()

# print("database reset")

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





# def map_script(fid):
#     pid = os.getpid()
#     flux_db_path = f"./data/temp/flux_lines_{pid}.sqlite"
    
#     # Perform the mapping
#     try:
#         netdata = map_substation(fid, flux_db_path)
#         return netdata
#     except Exception as e:
#         print(f"Error in worker {pid} for fid {fid}: {e}")
#         return None  # or return (fid, str(e)) if you want to track errors


# def main():
#     mapped_subs = get_mapped_substations_data()

#     print(f"substations mapped: {len(mapped_subs)}")

#     fids = [11374879, 11374881, 11374883, 11374885, 11374888, 11374889]

#     fids = [fid for fid in fids if fid not in mapped_subs]

#     with Pool(processes=cpu_count()-4) as pool:
#         for netdata in tqdm(pool.imap_unordered(map_script, fids), total=len(fids)):
#             if netdata is not None:
#                 try:
#                     netdata.to_sql()
#                 except Exception as e:
#                     print(f"[!] Error saving netdata: {e}")

def worker(fid_list, result_queue, index):
    flux_db_path = os.path.join(".","data", "temp", f"flux_lines_{index}.sqlite")
    # flux_db_path = f".\data\temp\flux_lines_{index}.sqlite"
    for fid in fid_list:
        try:
            # Clear flux db if needed here
            netdata = map_substation(fid, str(flux_db_path))
            result_queue.put(netdata)
        except Exception as e:
            print(f"[!] Error on FID {fid}: {e}")

def writer(result_queue):
    while True:
        netdata = result_queue.get()
        if netdata is None:
            break
        try:
            netdata.to_sql()
        except Exception as e:
            print(f"[!] Error saving netdata: {e}")

def main():

    all_fids_list_raw = extract_all_fids()

    # removing substations that cause as it causes an infinite loop - investigate! (something in the point overlap point)
    problem_subs = [11375446, 11375106, 11375790, 11375939, 11376031]
    all_fids_list = [item for item in all_fids_list_raw if item not in problem_subs]
    all_fids_list[0:40]

    # Check to see which substations have already been mapped
    mapped_subs = get_mapped_substations_data()
    fids = [fid for fid in all_fids_list if fid not in mapped_subs]

    num_of_workers = min(cpu_count() - 1, len(fids))
    chunk_size = (len(fids) + num_of_workers - 1) // num_of_workers
    fid_chunks = [fids[i:i+chunk_size] for i in range(0, len(fids), chunk_size)]

    result_queue = Queue()
    writer_process = Process(target=writer, args=(result_queue,))
    writer_process.start()

    workers = []
    for i, fid_chunk in enumerate(fid_chunks):
        p = Process(target=worker, args=(fid_chunk, result_queue, i))
        p.start()
        workers.append(p)

    for p in tqdm(workers, desc="Mapping processes"):
        p.join()

    result_queue.put(None)
    writer_process.join()

if __name__ == '__main__':
    print("Starting mapping:")
    main()
    print("Finished script")


