import os
from pathlib import Path
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200000000
# ==========================================
# CONFIGURATION
# ==========================================
# downscales from 5064x52224=264,462,336 to 16,528,896, where the default PIL max pixel count is 89,478,485
SCALE_FACTOR = 0.25 
CSV_PATH = "/home/aajung/lunar-project/lunar-topography/LROC_NeRF/lroc_data.csv"
OUTPUT_IMG_DIR = "/home/aajung/lunar-project/lroc_pngs"


def downscale_and_update_dataset(
    csv_dir: str,
    output_dir: str,
    scale: float,
):
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_dir)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    new_image_paths = []
    new_widths = []
    new_heights = []

    print(f"Starting batch downscale for {len(df)} images...")
    print(f"Resampling mode: Image.Resampling.BOX (Area Averaging)")
    print(f"Output directory: {output_dir}\n")

    for idx, row in df.iterrows():
        # orig_path = Path(str(row["png_path"]))
        id = str(row["Product ID"])
        orig_path = Path(f"{output_dir}/{id}.png")
        
        dest_filename = f"{orig_path.stem}.png"
        dest_path = out_dir_path / dest_filename

        if not orig_path.exists():
            print(f"  [!] Missing file: {orig_path}. Keeping target path in CSV.")
            new_image_paths.append(str(dest_path.resolve()))
            new_widths.append(int(round(row["Pixel Width"] * scale)))
            new_heights.append(int(round(row["Pixel Height"] * scale)))
            continue

        # Open and resize using Box / Area Averaging
        with Image.open(orig_path) as img:
            new_w = int(round(img.width * scale))
            new_h = int(round(img.height * scale))

            resized_img = img.resize((new_w, new_h), resample=Image.Resampling.BOX)
            resized_img.save(dest_path)

        new_image_paths.append(str(dest_path.resolve()))
        new_widths.append(new_w)
        new_heights.append(new_h)

        print(
            f"  [{idx+1}/{len(df)}] Scaled {orig_path.name} "
            f"({row['Pixel Width']}x{row['Pixel Height']} -> {new_w}x{new_h})"
        )

    # Update CSV columns to match the scaled coordinate system
    df["Pixel Width"] = new_widths
    df["Pixel Height"] = new_heights
    df["Focal Length X (pix)"] = new_widths
    df["Focal Length Y (pix)"] = new_heights
    df["Cx"] = df["Cx"] * scale
    df["Cy"] = df["Cy"] * scale

    # Scale focal length to keep pinhole projection angular field identical
    if "Focal Length" in df.columns:
        df["Focal Length"] = df["Focal Length"] * scale

    # delete old and replace with new
    os.remove(csv_dir)
    df.to_csv(csv_dir, index=False)
    print(f"\n[SUCCESS] Updated dataset written to: {csv_dir}")


if __name__ == "__main__":
    downscale_and_update_dataset(
        csv_dir=CSV_PATH,
        output_dir=OUTPUT_IMG_DIR,
        scale=SCALE_FACTOR,
    )