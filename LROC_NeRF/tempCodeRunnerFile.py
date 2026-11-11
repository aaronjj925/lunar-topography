   
import numpy as np
import spiceypy as spice
with open("kernelList.txt", "r") as f:
    kernelList = [line.strip() for line in f.readlines()]
# 1. Load kernels and get state at image start time
spice.furnsh(kernelList)
et = spice.str2et(time_utc)

# State vector [x, y, z, vx, vy, vz] of LRO relative to Moon center in IAU_MOON
state, _ = spice.spkezr("LRO", et, "IAU_MOON", "NONE", "MOON")
v_orbital = state[3:6]  # Velocity vector
v_orbital_unit = v_orbital / np.linalg.norm(v_orbital)
camera_frame = "LRO_LROCNACL"
# 2. Base camera-to-world matrix
r_c2w_raw = spice.pxform(camera_frame, "IAU_MOON", et)

# Define both candidate matrices
R_optA = np.array([
    [ 0, -1,  0],
    [-1,  0,  0],
    [ 0,  0, -1]
], dtype=np.float64)

R_optB = np.array([
    [ 0,  1,  0],
    [ 1,  0,  0],
    [ 0,  0, -1]
], dtype=np.float64)

# 3. In Nerfstudio / OpenCV convention:
# Column 0 is local +X (image right, dx)
# Column 1 is local +Y (image down / along scan, dy)
# Column 2 is local +Z (or -Z optical boresight)
c2w_A = r_c2w_raw @ R_optA
c2w_B = r_c2w_raw @ R_optB

# Local +Y in IAU_MOON coordinates
down_vec_A = c2w_A[:, 1]
down_vec_B = c2w_B[:, 1]

# 4. Compute alignment with flight vector
dot_A = np.dot(down_vec_A, v_orbital_unit)
dot_B = np.dot(down_vec_B, v_orbital_unit)

print(f"Alignment Option A (+90 roll): {dot_A:+.4f}")
print(f"Alignment Option B (-90 roll): {dot_B:+.4f}")

if dot_A > 0.8:
    print("\n--> Option A is CORRECT: Image top-to-bottom matches flight direction.")
elif dot_B > 0.8:
    print("\n--> Option B is CORRECT: Image top-to-bottom matches flight direction.")
else:
    print("\n--> Warning: Neither matches the flight vector closely. Check boresight axis.")

spice.kclear()