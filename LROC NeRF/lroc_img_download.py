import os
import requests
import cv2
import numpy as np

# numpy 2.0 merged np.product and np.prod, but planetaryimage works with older version, so it needs to product=prod
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
            print(f"Searching NASA PDS Index...")
            response = requests.get(base_api_url, params=params, timeout=15)
            docs = response.json().get("response", {}).get("docs", [])
            
            if not docs:
                print(f"  [!] Not found in the PDS index. Skipping.")
                continue
                
            doc = docs[0]
            img_url = find_img_url(doc)
                        
            if not img_url:
                print(f"  [!] Still no .IMG link found.")
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
            
            # --- CONVERSION PHASE (TRUE ALBEDO INTEGRATION) ---
            print(f"Converting to PNG (Applying True Albedo Sigma Stretch)...")
            pds_img = PDS3Image.open(img_filepath)
            
            # Squeeze the array into a flat 2D grid
            image_array = np.squeeze(pds_img.image).astype(np.float32)
            valid_pixels = image_array[image_array > 0]
            
            if len(valid_pixels) > 0:
                # 1. Find the true average brightness of the specific terrain
                mean_val = np.mean(valid_pixels)
                std_val = np.std(valid_pixels)
                
                # 2. Set limits based on Standard Deviation, NOT percentiles.
                # Black point: slightly below the average dirt color
                p_low = mean_val - (1.5 * std_val) 
                # White point: pushed far up so dark rocks don't get stretched to pure white
                p_high = mean_val + (4.0 * std_val) 
                
                # 3. Clip and scale to the new anchored limits
                image_array_clipped = np.clip(image_array, p_low, p_high)
                
                if p_high > p_low:
                    normalized_float = (image_array_clipped - p_low) / (p_high - p_low)
                else:
                    normalized_float = np.zeros_like(image_array_clipped)
                
                # 4. Apply standard sRGB Gamma (2.2) for natural contrast weighting
                gamma = 2.2 
                gamma_corrected = np.power(normalized_float, 1.0 / gamma)
                
                img_8bit = (gamma_corrected * 255.0).astype(np.uint8)
                
                # 5. Extremely mild CLAHE (clipLimit 1.2) just to preserve COLMAP edges
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(32, 32))
                final_image = clahe.apply(img_8bit)
                
            else:
                final_image = image_array.astype(np.uint8)
            
            img = Image.fromarray(final_image)
            img.save(png_filepath)
            print(f"  [SUCCESS] Converted and saved as {png_filename}")
            
            # --- CLEANUP PHASE ---
            os.remove(img_filepath)
            print(f"  [CLEANUP] Deleted the raw .IMG file.")
            
        except Exception as e:
            print(f"  [ERROR] Pipeline failed for {pid}: {e}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    my_product_ids = [
        "M162107819RC",
        "M172717111RC"
    ]
    
    download_and_convert_lroc(my_product_ids)