import json
import argparse
import re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Path & Product ID Parser
# ---------------------------------------------------------------------------
def extract_product_id(path_str):
    """
    Extracts the portion of the path after '11_10_lunar/'.
    e.g., '/workspace/data/11_10_lunar/M170361668RC.png' -> 'M170361668RC.png'
    """
    pattern = r"11_10_lunar/(.+)$"
    match = re.search(pattern, str(path_str))
    if match:
        return match.group(1)
    # Fallback to the filename if the specific folder isn't in the path
    return Path(path_str).name


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def load_poses_from_transforms(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    poses = {}
    for frame in data.get("frames", []):
        raw_path = frame.get("file_path", "")
        pid = extract_product_id(raw_path)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        poses[pid] = c2w
    return poses


def load_poses_from_csv(csv_path):
    import csv

    poses = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = (
                row.get("file_path")
                or row.get("image_path")
                or row.get("product_id")
                or row.get("image_name")
                or ""
            )
            pid = extract_product_id(raw_path)

            tx = float(row.get("x", 0.0))
            ty = float(row.get("y", 0.0))
            tz = float(row.get("z", 0.0))

            qx = float(row.get("qx", 0.0))
            qy = float(row.get("qy", 0.0))
            qz = float(row.get("qz", 0.0))
            qw = float(row.get("qw", 1.0))

            r_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = r_mat
            c2w[:3, 3] = [tx, ty, tz]

            poses[pid] = c2w
    return poses


# ---------------------------------------------------------------------------
# Fast Geometry Builders (Batched to prevent lag)
# ---------------------------------------------------------------------------
def build_frustum_lines(poses, scale=0.2, fov_deg=45.0):
    aspect = 1.0
    half_angle = np.deg2rad(fov_deg / 2.0)
    w = scale * np.tan(half_angle)
    h = w / aspect
    d = scale

    corners_cam = np.array([
        [0, 0, 0],          # Apex
        [-w, -h, d],        # TL
        [ w, -h, d],        # TR
        [ w,  h, d],        # BR
        [-w,  h, d]         # BL
    ])

    lines_idx = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1)
    ]

    fx, fy, fz = [], [], []
    for c2w in poses.values():
        r_mat = c2w[:3, :3]
        t = c2w[:3, 3]
        corners_world = (r_mat @ corners_cam.T).T + t

        for start, end in lines_idx:
            fx.extend([corners_world[start, 0], corners_world[end, 0], None])
            fy.extend([corners_world[start, 1], corners_world[end, 1], None])
            fz.extend([corners_world[start, 2], corners_world[end, 2], None])

    return fx, fy, fz


def build_cartesian_axes(poses, scale=0.2, axis_ratio=0.8):
    axis_len = scale * axis_ratio
    axes = {
        "x": {"x": [], "y": [], "z": []},
        "y": {"x": [], "y": [], "z": []},
        "z": {"x": [], "y": [], "z": []}
    }

    for c2w in poses.values():
        r_mat = c2w[:3, :3]
        t = c2w[:3, 3]

        local_dirs = {
            "x": r_mat[:, 0],  # Camera Right
            "y": r_mat[:, 1],  # Camera Down/Up
            "z": r_mat[:, 2]   # Camera Optical Boresight Forward
        }

        for k, vec in local_dirs.items():
            end = t + vec * axis_len
            axes[k]["x"].extend([t[0], end[0], None])
            axes[k]["y"].extend([t[1], end[1], None])
            axes[k]["z"].extend([t[2], end[2], None])

    return axes


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
def plot_cameras(poses, scale=0.2, fov=45.0, axis_ratio=0.8):
    product_ids = list(poses.keys())
    num_cams = len(product_ids)
    print(f"Loaded {num_cams} cameras. Building visualization...")

    # Unique colors per product ID for markers
    palette = pcolors.qualitative.Alphabet + pcolors.qualitative.Dark24
    marker_colors = [palette[i % len(palette)] for i in range(num_cams)]

    # 1. Camera Centers + Floating Labels
    centers = np.array([poses[pid][:3, 3] for pid in product_ids])
    nodes_trace = go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers+text',
        marker=dict(size=5, color=marker_colors, line=dict(color='white', width=1)),
        text=product_ids,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>Product:</b> %{text}<br>" +
                      "<b>Pos:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        name='Camera Centers'
    )

    # 2. Camera View Frustums (wireframes)
    fx, fy, fz = build_frustum_lines(poses, scale=scale, fov_deg=fov)
    frustum_trace = go.Scatter3d(
        x=fx, y=fy, z=fz,
        mode='lines',
        line=dict(color='rgba(180, 180, 180, 0.35)', width=1.5),
        hoverinfo='none',
        name='Frustums'
    )

    # 3. Cartesian Axes for every camera (+X Red, +Y Green, +Z Blue)
    axes = build_cartesian_axes(poses, scale=scale, axis_ratio=axis_ratio)

    x_trace = go.Scatter3d(
        x=axes["x"]["x"], y=axes["x"]["y"], z=axes["x"]["z"],
        mode='lines', line=dict(color='red', width=3.5), name='+X (Right)'
    )
    y_trace = go.Scatter3d(
        x=axes["y"]["x"], y=axes["y"]["y"], z=axes["y"]["z"],
        mode='lines', line=dict(color='green', width=3.5), name='+Y (Down/Up)'
    )
    z_trace = go.Scatter3d(
        x=axes["z"]["x"], y=axes["z"]["y"], z=axes["z"]["z"],
        mode='lines', line=dict(color='deepskyblue', width=4.5), name='+Z (Boresight)'
    )

    fig = go.Figure(data=[frustum_trace, x_trace, y_trace, z_trace, nodes_trace])

    fig.update_layout(
        template="plotly_dark",
        title=f"LROC Cameras & Orientations ({num_cams} images)",
        scene=dict(
            aspectmode='data',
            xaxis=dict(title="World X", showgrid=True),
            yaxis=dict(title="World Y", showgrid=True),
            zaxis=dict(title="World Z", showgrid=True)
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(r=0, l=0, b=0, t=50)
    )

    fig.show()


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera poses with product ID labels and Cartesian frames.")
    parser.add_argument("--json", type=str, help="Path to transforms.json")
    parser.add_argument("--csv", type=str, help="Path to CSV metadata")
    parser.add_argument("--scale", type=float, default=0.2, help="Visual scale of frustums and axes")
    parser.add_argument("--fov", type=float, default=45.0, help="Camera visual FOV in degrees")
    parser.add_argument("--axis-ratio", type=float, default=0.8, help="Length of coordinate axes relative to frustum scale")

    args = parser.parse_args()

    poses = {}
    if args.json:
        poses = load_poses_from_transforms(args.json)
    elif args.csv:
        poses = load_poses_from_csv(args.csv)
    else:
        print("Please supply either --json <transforms.json> or --csv <metadata.csv>")
        exit(1)

    plot_cameras(poses, scale=args.scale, fov=args.fov, axis_ratio=args.axis_ratio)
    # python visualize_cameras.py --json /home/aajung/smurf/data/11_10_lunar/transforms.json --scale 0.5