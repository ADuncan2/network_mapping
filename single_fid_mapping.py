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
import logging
import logging.handlers
from datetime import datetime
import traceback



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


class LogMessage:
    """Simple class to structure log messages for the queue"""
    def __init__(self, level, message, fid=None, process_name=None, timestamp=None):
        self.level = level
        self.message = message
        self.fid = fid
        self.process_name = process_name
        self.timestamp = timestamp or datetime.now()

def setup_logging():
    """Set up the main logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger('mapping')
    logger.setLevel(logging.INFO)
    
    # File handler for all logs
    file_handler = logging.FileHandler(f'logs/mapping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - FID:%(fid)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_writer(log_queue, result_queue, num_expected):
    """Combined writer that handles both logging and results"""
    logger = setup_logging()
    
    pbar = tqdm(total=num_expected, desc="Substations mapped")
    completed = 0
    failed = 0
    
    # Keep track of statistics
    stats = {
        'completed': 0,
        'failed': 0,
        'warnings': 0,
        'start_time': datetime.now()
    }
    
    while True:
        # Check for log messages (non-blocking)
        try:
            log_msg = log_queue.get_nowait()
            if log_msg is None:  # Shutdown signal for logging
                break
                
            # Log the message with FID context
            extra = {'fid': log_msg.fid or 'N/A'}
            if log_msg.level == logging.INFO:
                logger.info(log_msg.message, extra=extra)
            elif log_msg.level == logging.WARNING:
                logger.warning(log_msg.message, extra=extra)
                stats['warnings'] += 1
            elif log_msg.level == logging.ERROR:
                logger.error(log_msg.message, extra=extra)
                stats['failed'] += 1
                
        except:
            pass  # No log messages available
        
        # Check for result data
        try:
            network_data = result_queue.get_nowait()
            if network_data is None:  # Shutdown signal for results
                break
                
            try:
                network_data.to_sql()
                completed += 1
                stats['completed'] += 1
                pbar.update(1)
                
                # Log successful completion
                fid = getattr(network_data, 'fid', 'Unknown')
                logger.info(f"Successfully saved network data", extra={'fid': fid})
                
            except Exception as e:
                failed += 1
                stats['failed'] += 1
                error_msg = f"Error saving network data: {str(e)}"
                logger.error(error_msg, extra={'fid': getattr(network_data, 'fid', 'Unknown')})
                
        except:
            pass  # No result data available
        
        # Check if we're done with results
        if completed + failed >= num_expected:
            break
    
    pbar.close()
    
    # Log final statistics
    duration = datetime.now() - stats['start_time']
    logger.info(f"Mapping completed - Total: {completed + failed}, "
               f"Success: {completed}, Failed: {failed}, Warnings: {stats['warnings']}, "
               f"Duration: {duration}", extra={'fid': 'SUMMARY'})
    
    print(f"\n=== Mapping Summary ===")
    print(f"Total processed: {completed + failed}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Warnings: {stats['warnings']}")
    print(f"Duration: {duration}")
    print(f"Log file: logs/mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def worker(fid_list, result_queue, log_queue, index):
    """Enhanced worker with logging integration"""
    process_name = f"Worker-{index}"
    flux_db_path = os.path.join(".", "data", "temp", f"flux_lines_{index}.sqlite")
    
    # Log worker startup
    log_queue.put(LogMessage(
        logging.INFO, 
        f"{process_name} started with {len(fid_list)} substations",
        process_name=process_name
    ))
    
    for i, fid in enumerate(fid_list, 1):
        try:
            # Log start of processing
            log_queue.put(LogMessage(
                logging.INFO,
                f"Starting mapping ({i}/{len(fid_list)})",
                fid=fid,
                process_name=process_name
            ))
            
            # Clear flux db if needed here
            netdata = map_substation(fid, str(flux_db_path))
            
            # Add FID to network data for logging purposes
            if hasattr(netdata, '__dict__'):
                netdata.fid = fid
            
            result_queue.put(netdata)
            
            # Log successful completion
            log_queue.put(LogMessage(
                logging.INFO,
                f"Mapping completed successfully ({i}/{len(fid_list)})",
                fid=fid,
                process_name=process_name
            ))
            
        except Exception as e:
            # Log detailed error information
            error_msg = f"Mapping failed: {str(e)}"
            log_queue.put(LogMessage(
                logging.ERROR,
                error_msg,
                fid=fid,
                process_name=process_name
            ))
            
            # Also log the full traceback for debugging
            log_queue.put(LogMessage(
                logging.ERROR,
                f"Full traceback: {traceback.format_exc()}",
                fid=fid,
                process_name=process_name
            ))
    
    # Log worker completion
    log_queue.put(LogMessage(
        logging.INFO,
        f"{process_name} completed processing {len(fid_list)} substations",
        process_name=process_name
    ))

def main():
    all_fids_list_raw = extract_all_fids()

    # removing substations that cause issues - investigate! 
    problem_subs = [11375446, 11375106, 11375790, 11375939, 11376031]
    all_fids_list = [item for item in all_fids_list_raw if item not in problem_subs]

    # Check to see which substations have already been mapped
    mapped_subs = get_mapped_substations_data()
    fids = [fid for fid in all_fids_list if fid not in mapped_subs]

    print(f"Total substations to process: {len(fids)}")
    
    if len(fids) == 0:
        print("No substations to process!")
        return

    # Calculate optimal number of workers
    num_of_workers = min(cpu_count() - 1, len(fids))
    chunk_size = (len(fids) + num_of_workers - 1) // num_of_workers
    fid_chunks = [fids[i:i+chunk_size] for i in range(0, len(fids), chunk_size)]

    print(f"Using {num_of_workers} workers with chunks of ~{chunk_size} substations each")

    # Create queues
    result_queue = Queue()
    log_queue = Queue()
    
    # Start the combined writer process
    writer_process = Process(
        target=log_writer, 
        args=(log_queue, result_queue, len(fids))
    )
    writer_process.start()

    # Start worker processes
    workers = []
    for i, fid_chunk in enumerate(fid_chunks):
        p = Process(
            target=worker, 
            args=(fid_chunk, result_queue, log_queue, i),
            name=f"Worker-{i}"
        )
        p.start()
        workers.append(p)

    # Wait for all workers to complete
    for p in tqdm(workers, desc="Waiting for workers"):
        p.join()

    # Send shutdown signals
    result_queue.put(None)
    log_queue.put(None)
    
    # Wait for writer to finish
    writer_process.join()

    print("All processes completed successfully!")

if __name__ == "__main__":
    main()


