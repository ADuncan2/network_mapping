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
from datetime import datetime, timedelta
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

class LogBatch:
    """Container for all log messages from processing one FID"""
    def __init__(self, fid, process_name):
        self.fid = fid
        self.process_name = process_name
        self.messages = []
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_log(self, level, message):
        self.messages.append({
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        })
    
    def finalize(self):
        self.end_time = datetime.now()
        
    def get_duration(self):
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time

def setup_logging():
    """Set up the main logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger('mapping')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler for all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/mapping_{timestamp}.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - FID:%(fid)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Force immediate flushing to disk for real-time logging
    file_handler.flush = lambda: file_handler.stream.flush()
    
    # Set stream to unbuffered mode
    if hasattr(file_handler, 'stream'):
        file_handler.stream.reconfigure(line_buffering=True)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - FID:%(fid)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, timestamp

def log_writer(log_queue, result_queue, num_expected):
    """Combined writer that handles both logging and results"""
    logger, log_timestamp = setup_logging()
    
    completed = 0
    failed = 0
    
    # Keep track of statistics
    stats = {
        'completed': 0,
        'failed': 0,
        'warnings': 0,
        'errors': 0,
        'start_time': datetime.now(),
        'total_processing_time': timedelta()
    }
    
    while completed + failed < num_expected:
        log_processed = False
        result_processed = False
        
        # Check for log batches (non-blocking)
        try:
            log_item = log_queue.get_nowait()
            if log_item is None:  # Shutdown signal for logging
                break
                
            log_processed = True
            
            # Handle log batch
            if isinstance(log_item, LogBatch):
                duration = log_item.get_duration()
                stats['total_processing_time'] += duration
                
                # Log all messages for this FID together
                for msg in log_item.messages:
                    extra = {'fid': log_item.fid}
                    
                    if msg['level'] == logging.INFO:
                        logger.info(msg['message'], extra=extra)
                    elif msg['level'] == logging.WARNING:
                        logger.warning(msg['message'], extra=extra)
                        stats['warnings'] += 1
                    elif msg['level'] == logging.ERROR:
                        logger.error(msg['message'], extra=extra)
                        stats['errors'] += 1
                
                # Add duration summary for this FID
                logger.info(f"Processing duration: {duration}", extra={'fid': log_item.fid})
                
                # Force flush to disk immediately
                for handler in logger.handlers:
                    handler.flush()
                
        except:
            pass  # No log messages available
        
        # Check for result data (non-blocking)
        try:
            network_data = result_queue.get_nowait()
            if network_data is None:  # Shutdown signal for results
                break
                
            result_processed = True
            
            try:
                network_data.to_sql()
                completed += 1
                stats['completed'] += 1
                
                # Log successful save
                fid = getattr(network_data, 'fid', 'Unknown')
                logger.info(f"Successfully saved network data", extra={'fid': fid})
                
                # Force flush after database saves
                for handler in logger.handlers:
                    handler.flush()
                
            except Exception as e:
                failed += 1
                stats['failed'] += 1
                error_msg = f"Error saving network data: {str(e)}"
                fid = getattr(network_data, 'fid', 'Unknown')
                logger.error(error_msg, extra={'fid': fid})
                
                # Force flush after errors (critical for debugging)
                for handler in logger.handlers:
                    handler.flush()
                
        except:
            pass  # No result data available
        
        # If neither queue had data, sleep briefly to avoid busy waiting
        if not log_processed and not result_processed:
            import time
            time.sleep(0.01)
    
    # Process any remaining log messages
    remaining_logs = True
    while remaining_logs:
        try:
            log_item = log_queue.get_nowait()
            if log_item is None:
                break
                
            # Handle remaining log batch
            if isinstance(log_item, LogBatch):
                duration = log_item.get_duration()
                stats['total_processing_time'] += duration
                
                for msg in log_item.messages:
                    extra = {'fid': log_item.fid}
                    logger.log(msg['level'], msg['message'], extra=extra)
                
                logger.info(f"Processing duration: {duration}", extra={'fid': log_item.fid})
        except:
            remaining_logs = False
    
    # Log final statistics
    total_duration = datetime.now() - stats['start_time']
    avg_processing_time = stats['total_processing_time'] / max(completed + failed, 1)
    
    summary_msg = (f"Mapping completed - Total: {completed + failed}, "
                  f"Success: {completed}, Failed: {failed}, "
                  f"Warnings: {stats['warnings']}, Errors: {stats['errors']}, "
                  f"Total Duration: {total_duration}, "
                  f"Avg Processing Time: {avg_processing_time}")
    
    logger.info(summary_msg, extra={'fid': 'SUMMARY'})
    
    print(f"\n=== Mapping Summary ===")
    print(f"Total processed: {completed + failed}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Warnings: {stats['warnings']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total duration: {total_duration}")
    print(f"Average processing time per substation: {avg_processing_time}")
    print(f"Log file: logs/mapping_{log_timestamp}.log")
    
    return stats

def worker_single(args):
    """Process a single FID and return batched logs"""
    fid, worker_id = args
    process_name = f"Worker-{worker_id}"
    flux_db_path = os.path.join(".", "data", "temp", f"flux_lines_{worker_id}.sqlite")
    
    # Create log batch for this FID
    log_batch = LogBatch(fid, process_name)
    
    try:
        log_batch.add_log(logging.INFO, "Starting mapping")
        
        # Clear flux db if needed here
        netdata, log_batch = map_substation(fid, str(flux_db_path),log_batch)
        
        # Add FID to network data for logging purposes
        if hasattr(netdata, '__dict__'):
            netdata.fid = fid
        elif netdata is not None:
            # If it's not an object with attributes, try to add fid another way
            try:
                netdata.fid = fid
            except:
                pass  # If we can't add fid, that's okay
        
        

        log_batch.finalize()
        
        return ('success', netdata, log_batch)
        
    except Exception as e:
        error_msg = f"Mapping failed: {str(e)}"
        log_batch.add_log(logging.ERROR, error_msg)
        
        # Add full traceback for debugging
        traceback_msg = f"Full traceback: {traceback.format_exc()}"
        log_batch.add_log(logging.ERROR, traceback_msg)
        
        log_batch.finalize()
        
        return ('error', None, log_batch)

def main():
    print("Starting GIS mapping process...")
    
    # Extract all FIDs
    all_fids_list_raw = extract_all_fids()

    # Remove problematic substations - investigate these later!
    problem_subs = [11375446, 11375106, 11375790, 11375939, 11376031]
    all_fids_list = [item for item in all_fids_list_raw if item not in problem_subs]
    
    if problem_subs:
        print(f"Excluding {len(problem_subs)} problematic substations: {problem_subs}")

    # Check which substations have already been mapped
    print("Checking for previously mapped substations...")
    mapped_subs = get_mapped_substations_data()
    fids = [fid for fid in all_fids_list if fid not in mapped_subs]
    fids = fids[15:35]

    print(f"Total substations available: {len(all_fids_list)}")
    print(f"Already mapped: {len(mapped_subs)}")
    print(f"Remaining to process: {len(fids)}")
    
    if len(fids) == 0:
        print("No substations to process!")
        return

    # Calculate optimal number of workers
    num_of_workers = min(cpu_count() - 1, len(fids), 8)  # Cap at 8 to avoid too many temp files
    print(f"Using {num_of_workers} worker processes")

    # Prepare arguments for pool workers
    # Each worker gets a worker_id for temp file naming
    worker_args = [(fid, i % num_of_workers) for i, fid in enumerate(fids)]

    # Create queues
    result_queue = Queue()
    log_queue = Queue()
    
    # Start the combined writer process
    writer_process = Process(
        target=log_writer, 
        args=(log_queue, result_queue, len(fids)),
        name="LogWriter"
    )
    writer_process.start()

    print("Starting mapping processes...")
    
    # Use Pool for better load balancing
    successful_mappings = 0
    failed_mappings = 0
    
    try:
        with Pool(processes=num_of_workers) as pool:
            # Process results as they complete (unordered for better performance)
            for result in tqdm(pool.imap_unordered(worker_single, worker_args), 
                              total=len(fids), desc="Processing substations"):
                
                status, netdata, log_batch = result
                
                # Send log batch to queue first
                log_queue.put(log_batch)
                
                # Send result data if successful
                if status == 'success' and netdata is not None:
                    result_queue.put(netdata)
                    successful_mappings += 1
                else:
                    failed_mappings += 1
                    
    except KeyboardInterrupt:
        print("\nMapping interrupted by user!")
    except Exception as e:
        print(f"\nUnexpected error in main process: {e}")
        
    finally:
        # Send shutdown signals
        print("Shutting down processes...")
        result_queue.put(None)
        log_queue.put(None)
        
        # Wait for writer to finish with timeout
        print("Waiting for log writer to finish...")
        writer_process.join(timeout=30)
        
        # Force terminate if it doesn't finish gracefully
        if writer_process.is_alive():
            print("Force terminating log writer...")
            writer_process.terminate()
            writer_process.join(timeout=5)
        
        print(f"\nImmediate results: {successful_mappings} successful, {failed_mappings} failed")
        print("Check the log file for detailed information about each substation.")
        print("Mapping process completed!")

if __name__ == "__main__":
    main()