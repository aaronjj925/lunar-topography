"""Plot the image footprints in lroc_data.csv."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon


CORNER_COLUMNS = ["Upper Right", "Upper Left", "Lower Right", "Lower Left"]
PRODUCT_ID_COLUMN = "Product ID"
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_coordinate(value):
	"""Parse a coordinate cell into (latitude, longitude) decimal degrees.

	Cells may contain decimal coordinates separated by commas or spaces, with
	optional N/S/E/W hemisphere letters (for example, '12.3 N, 45.6 W').
	"""
	text = str(value).strip().upper()
	numbers = [float(item) for item in NUMBER_RE.findall(text)]
	if len(numbers) < 2:
		raise ValueError(f"not a latitude/longitude pair: {value!r}")

	latitude, longitude = numbers[:2]
	hemispheres = re.findall(r"[NSEW]", text)
	for direction in hemispheres:
		if direction == "S":
			latitude = -abs(latitude)
		elif direction == "N":
			latitude = abs(latitude)
		elif direction == "W":
			longitude = -abs(longitude)
		elif direction == "E":
			longitude = abs(longitude)
	return latitude, longitude


def main():
	data = pd.read_csv(Path("../lroc_data.csv"))
	missing = [column for column in [*CORNER_COLUMNS, PRODUCT_ID_COLUMN] if column not in data.columns]
	if missing:
		raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

	fig, ax = plt.subplots(figsize=(11, 8))
	count = 0
	color_map = plt.get_cmap("hsv")
	for row_index, row in data.iterrows():
		try:
			corners = {column: parse_coordinate(row[column]) for column in CORNER_COLUMNS}
		except ValueError as error:
			print(f"Skipping CSV row {row_index + 2}: {error}")
			continue

		# Plot longitude on x and latitude on y, joining the four corners.
		order = ["Upper Left", "Upper Right", "Lower Right", "Lower Left"]
		vertices = [(corners[name][1], corners[name][0]) for name in order]
		product_id = str(row[PRODUCT_ID_COLUMN])
		color = color_map(count / max(len(data), 1))
		ax.add_patch(
			Polygon(
				vertices,
				closed=True,
				fill=False,
				linewidth=0.8,
				alpha=0.6,
				label=f"{product_id} (row {row_index + 2})",
				color=color,
			)
		)
		count += 1

	ax.autoscale_view()
	ax.set_xlabel("Longitude (degrees)")
	ax.set_ylabel("Latitude (degrees)")
	ax.set_title(f"LROC image footprints ({count} plotted)")
	if count:
		ax.legend(title=PRODUCT_ID_COLUMN, bbox_to_anchor=(1.02, 1), loc="upper left")
	ax.grid(True, linestyle=":", alpha=0.5)
	fig.tight_layout()
	output_path = Path("lroc_image_footprints.png")
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	print(f"Saved footprint plot to {output_path.resolve()}")
	plt.show(block=True)


if __name__ == "__main__":
	main()
