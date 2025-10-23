import deeplake
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import time
import concurrent.futures # Import the concurrent module

# --- Configuration ---
DATASET_PATH = "hub://activeloop/ffhq"
NUM_IMAGES_TO_LOAD = 10000
TENSOR_PATH = "images_1024/image"
MAX_WORKERS = 1 # Start with the number of CPU cores, adjust as needed (e.g., * 2 for I/O bound)
# We need the total size to calculate original indices correctly
try:
    # Reduce timeout for faster check if connection is slow/failing
    ds_full = deeplake.load(DATASET_PATH, read_only=True, creds=None) # Use load for < v4
    TOTAL_IMAGES_IN_DS = len(ds_full)
    print(f"Detected total dataset size: {TOTAL_IMAGES_IN_DS}")
    del ds_full # Free up memory
except Exception as e:
    print(f"Warning: Could not get exact dataset length, assuming 70000. Error: {e}")
    TOTAL_IMAGES_IN_DS = 70000

START_INDEX = TOTAL_IMAGES_IN_DS - NUM_IMAGES_TO_LOAD

OUTPUT_DIR = "./ffhq_test_images" # New output dir
FILENAME_PREFIX = "ffhq_test"
IMAGE_FORMAT = "png"
# --- --- --- --- ---

# --- Verify Deep Lake Version ---
try:
    print(f"Using Deep Lake version: {deeplake.__version__}")
    time.sleep(0.1)
except AttributeError:
    print("Could not determine Deep Lake version automatically.")
# --- --- --- --- ---

# --- Worker Function ---
# This function will be executed by each thread
def save_image_worker(args):
    """Saves a single image based on provided arguments."""
    i, sample, start_idx, tensor_path, out_dir, prefix, img_format = args

    # Calculate the original index
    original_index = start_idx + i
    
    # Create filename
    filename = f"{prefix}_{original_index:05d}.{img_format}"
    output_path = os.path.join(out_dir, filename)
    if os.path.exists(output_path):
        return None # Success
        
    try:
        # *** Access the tensor using its FULL PATH ***
        image_np = sample[tensor_path].numpy()

        # Convert numpy array to PIL image
        img = Image.fromarray(image_np)

        # Save the image file
        img.save(output_path, format=img_format.upper())
        return None # Success

    except KeyError:
        original_index = start_idx + i
        # Return error info instead of printing directly from worker
        return (original_index, f"KeyError: Tensor '{tensor_path}' not found in sample.")
    except Exception as e:
        original_index = start_idx + i
        # Return error info
        return (original_index, f"Error processing/saving: {e}")
# --- --- --- --- ---

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Will save images from tensor '{TENSOR_PATH}' to: {OUTPUT_DIR}")
print(f"Using up to {MAX_WORKERS} worker threads.")

try:
    print("Opening Deeplake dataset view for the last {} images using deeplake.load()...".format(NUM_IMAGES_TO_LOAD))

    # *** USE deeplake.load() for versions < 4.0 ***
    ds_test_view = deeplake.load(DATASET_PATH, read_only=True)[-NUM_IMAGES_TO_LOAD:]

    print(f"Starting to save {len(ds_test_view)} images using thread pool...")

    futures = []
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Iterate through the dataset view sample by sample IN THE MAIN THREAD
        # Submit tasks to the executor
        for i, sample in enumerate(tqdm(ds_test_view, desc="Submitting tasks", total=len(ds_test_view))):
            # Package arguments for the worker function
            args_for_worker = (i, sample, START_INDEX, TENSOR_PATH, OUTPUT_DIR, FILENAME_PREFIX, IMAGE_FORMAT)
            # Submit the worker function with its arguments
            future = executor.submit(save_image_worker, args_for_worker)
            futures.append(future)

        # Optional: Wait for all tasks to complete and check for errors
        print("\nTasks submitted. Waiting for completion and checking results...")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result is not None: # Check if the worker returned error info
                original_index, error_message = result
                print(f"\nError processing image for original index {original_index}: {error_message}")


    print(f"\nFinished saving images process to {OUTPUT_DIR}")

except AttributeError as ae:
     print(f"\nAttributeError: {ae}")
     print("This might happen if deeplake.load() doesn't support an argument like 'read_only' in your specific v3 version.")
     print("Try removing 'read_only=True' from the deeplake.load() call if this occurs during loading.")
except Exception as e:
    print(f"\nAn error occurred during dataset loading or processing: {e}")