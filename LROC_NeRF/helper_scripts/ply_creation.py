import json
from pathlib import Path
import numpy as np


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    num_verts = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    vertex_data = np.empty(
        num_verts,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    vertex_data["x"] = points[:, 0].astype(np.float32)
    vertex_data["y"] = points[:, 1].astype(np.float32)
    vertex_data["z"] = points[:, 2].astype(np.float32)
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())


def generate_lunar_planar_ply(
    transforms_path: str,
    output_ply_path: str = "sparse_pc.ply",
    lunar_radius: float = 1737400.0,
    grid_res: int = 200,
    margin_ratio: float = 0.5,
):
    transforms_file = Path(transforms_path)
    with open(transforms_file, "r") as f:
        meta = json.load(f)

    cam_positions = []
    ground_subpoints = []

    for frame in meta["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        pos = c2w[:3, 3]
        cam_positions.append(pos)

        # Drop nadir to the lunar radius
        dist = np.linalg.norm(pos)
        ground_pt = pos * (lunar_radius / dist)
        ground_subpoints.append(ground_pt)

    cam_positions = np.array(cam_positions)
    ground_subpoints = np.array(ground_subpoints)

    # Center of target surface area
    center = ground_subpoints.mean(axis=0)

    # Surface normal (pointing up from center of moon through surface)
    normal = center / np.linalg.norm(center)

    # Establish an orthogonal basis on the local tangent plane
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(normal[2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u_axis = np.cross(normal, ref)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    # Measure the swath footprint span of the camera path
    rel_pts = ground_subpoints - center
    u_vals = np.dot(rel_pts, u_axis)
    v_vals = np.dot(rel_pts, v_axis)

    # Minimum 15 km swath width if camera trajectory has little baseline
    u_span = max(float(np.ptp(u_vals)), 15000.0) * (1.0 + margin_ratio)
    v_span = max(float(np.ptp(v_vals)), 15000.0) * (1.0 + margin_ratio)

    u_lin = np.linspace(-u_span / 2, u_span / 2, grid_res)
    v_lin = np.linspace(-v_span / 2, v_span / 2, grid_res)
    uu, vv = np.meshgrid(u_lin, v_lin)

    # Generate flat plane points
    grid_points = (
        center[None, :]
        + uu.flatten()[:, None] * u_axis[None, :]
        + vv.flatten()[:, None] * v_axis[None, :]
    )

    colors = np.full((grid_points.shape[0], 3), 128, dtype=np.uint8)

    out_file = transforms_file.parent / output_ply_path
    write_ply(out_file, grid_points, colors)
    print(f"Successfully generated {len(grid_points)} points at: {out_file}")

    # Register into transforms.json
    meta["ply_file_path"] = output_ply_path
    with open(transforms_file, "w") as f:
        json.dump(meta, f, indent=4)
    print(
        f"Updated {transforms_file.name} with 'ply_file_path': '{output_ply_path}'"
    )


if __name__ == "__main__":
    generate_lunar_planar_ply(
        "/home/aajung/smurf/data/11_10_lunar/transforms.json"
    )