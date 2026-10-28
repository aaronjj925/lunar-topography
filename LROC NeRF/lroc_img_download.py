import os
import requests
import numpy as np

# numpy 2.0 merged np.product and np.prod, but planetaryimage works with older versions
if not hasattr(np, 'product'):
    np.product = np.prod

from PIL import Image
from planetaryimage import PDS3Image


def find_img_url(data):
    """Recursively scans the JSON response for .IMG or .LBL files."""
    if isinstance(data, dict):
        for k, v in data.items():
            result = find_img_url(v)
            if result: return result
    elif isinstance(data, list):
        for item in data:
            result = find_img_url(item)
            if result: return result
    elif isinstance(data, str):
        if data.upper().endswith('.IMG'):
            return data
        elif data.upper().endswith('.LBL'):
            return data[:-4] + '.IMG'
    return None


def download_and_convert_lroc(product_ids, save_folder="lroc_processed"):
    os.makedirs(save_folder, exist_ok=True)
    
    base_api_url = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/select"
    
    for pid in product_ids:
        print(f"\n{'='*40}")
        print(f"Processing: {pid}")
        print(f"{'='*40}")
        
        search_id = pid[:-1] if (pid.endswith('E') or pid.endswith('C')) else pid
            
        params = {
            "q": f"PRODUCT_ID:*{search_id}*",
            "wt": "json"
        }
        
        try:
            print("Searching NASA PDS Index...")
            response = requests.get(base_api_url, params=params, timeout=15)
            docs = response.json().get("response", {}).get("docs", [])
            
            if not docs:
                print("  [!] Not found in the PDS index. Skipping.")
                continue
                
            doc = docs[0]
            img_url = find_img_url(doc)
                        
            if not img_url:
                print("  [!] Still no .IMG link found.")
                continue
                
            # --- PROPER URL CONSTRUCTION ---
            if not img_url.startswith("http"):
                clean_path = img_url.lstrip("/")
                img_url = f"https://pds.lroc.asu.edu/data/{clean_path}"
                
            # --- DOWNLOAD PHASE ---
            img_filename = img_url.split("/")[-1]
            img_filepath = os.path.join(save_folder, img_filename)
            png_filename = pid + ".png" 
            png_filepath = os.path.join(save_folder, png_filename)
            
            if os.path.exists(png_filepath):
                print(f"  [SKIPPED] {png_filename} already exists.")
                continue
                
            print(f"Downloading raw data from: {img_url}")
            img_data = requests.get(img_url, stream=True, timeout=30)
            
            if img_data.status_code != 200:
                print(f"  [ERROR] Server returned {img_data.status_code}. File might be missing.")
                continue
            
            with open(img_filepath, 'wb') as file:
                for chunk in img_data.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"  [SUCCESS] Downloaded raw {img_filename}")
            
# --- CONVERSION PHASE ---
            print("Converting to PNG...")
            pds_img = PDS3Image.open(img_filepath)
            image_array = np.squeeze(pds_img.image).astype(np.float32)

            # Mask out NaNs, Infs, and PDS special negative sentinel values (< -1e30 or < 0)
            valid_mask = np.isfinite(image_array) & (image_array > 0.0)

            if np.any(valid_mask):
                valid_pixels = image_array[valid_mask]

                # Percentile stretch: clips extreme outliers to maximize lunar surface contrast
                vmin, vmax = np.percentile(valid_pixels, (0.5, 99.5))

                if vmax > vmin:
                    clipped = np.clip(image_array, vmin, vmax)
                    normalized = (clipped - vmin) / (vmax - vmin) * 255.0
                else:
                    normalized = np.zeros_like(image_array)

                # Set masked/NULL background pixels to black (0)
                normalized[~valid_mask] = 0.0
            else:
                normalized = np.zeros_like(image_array)

            final_image = normalized.astype(np.uint8)

            img = Image.fromarray(final_image)
            img.save(png_filepath)
            print(f"  [SUCCESS] Converted and saved as {png_filename}")
            
            # --- CLEANUP PHASE ---
            os.remove(img_filepath)
            print("  [CLEANUP] Deleted the raw .IMG file.")
            
        except Exception as e:
            print(f"  [ERROR] Pipeline failed for {pid}: {e}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    my_product_ids = [
        "M150314875LC",
        "M150314875RC",
        "M152669226LC",
        "M152669226RC",
        "M155030843LC",
        "M155030843RC",
        "M155037640RC",
        "M157391825LC",
        "M157391825RC",
        "M159746292LC",
        "M159746292RC",
        "M162094243LC",
        "M162094243RC",
        "M162094272LC",
        "M162094272RC",
        "M162107819LC",
        "M162107819RC",
        "M165631933LC",
        "M165631933RC",
        "M165645494LC",
        "M165645494RC",
        "M167993567LC",
        "M167993567RC",
        "M168000370LC",
        "M168000370RC",
        "M168007168LC",
        "M168007168RC",
        "M170361668LC",
        "M170361668RC",
        "M172717067LC",
        "M172717067RC",
        "M172717111LC",
        "M172717111RC",
    ]
    
    download_and_convert_lroc(my_product_ids)