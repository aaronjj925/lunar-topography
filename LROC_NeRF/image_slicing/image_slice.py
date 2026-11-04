from pathlib import Path
from typing import Iterable, List, Union

from PIL import Image

def slice_png_into_horizontal_strips(
	png_files: Union[str, Path, Iterable[Union[str, Path]]],
	strip_height: int,
	output_dir: Union[str, Path, None] = None,
) -> List[Path]:
	"""Slice one or more PNG images into horizontal strips.

	The final strip may be shorter than ``strip_height``. The output files are
	named ``<input>_strip_000.png``, ``<input>_strip_001.png``, and so on.
	Returns the paths of all saved strips.
	"""
	if not isinstance(strip_height, int) or isinstance(strip_height, bool) or strip_height <= 0:
		raise ValueError("strip_height must be a positive integer")

	if isinstance(png_files, (str, Path)):
		png_files = [png_files]

	saved_paths: List[Path] = []
	for png_file in png_files:
		input_path = Path(png_file)
		if input_path.suffix.lower() != ".png":
			raise ValueError(f"Expected a PNG file: {input_path}")

		destination = Path(output_dir) if output_dir is not None else input_path.parent
		destination.mkdir(parents=True, exist_ok=True)

		with Image.open(input_path) as image:
			image.load()
			for index, top in enumerate(range(0, image.height, strip_height)):
				bottom = min(top + strip_height, image.height)
				strip = image.crop((0, top, image.width, bottom))
				output_path = destination / f"{input_path.stem}_strip_{index:03d}.png"
				strip.save(output_path, format="PNG")
				saved_paths.append(output_path)

	return saved_paths

if __name__ == "__main__":
	# Example usage
    df = pd.read_csv("/home/aajung/lunar-project/lunar-topography/LROC_NeRF/lroc_data.csv")
	input_files = df["png_path"] 
	strip_height = 100
 
	output_dir = "/home/aajung/lunar-project/lroc_sliced"
	saved_paths = slice_png_into_horizontal_strips(input_files, strip_height, output_dir)
	for path in saved_paths:
		print(path)