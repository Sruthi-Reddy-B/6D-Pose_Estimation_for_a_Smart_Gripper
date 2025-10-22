import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# --- File & Directory Helpers ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Load Pose Labels ---
def load_pose_labels(csv_path):
    return pd.read_csv(csv_path)

# --- Draw Coordinate Axes on Image ---
def draw_axes(img, origin, R_mat, length=50):
    cx, cy = int(origin[0]), int(origin[1])
    axes = np.array([[length,0,0], [0,length,0], [0,0,length]], dtype=float)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for vec, col in zip(axes, colors):
        pt = (int(cx+vec[0]), int(cy+vec[1]))
        cv2.line(img, (cx,cy), pt, col, 2)
    return img

# --- Convert Euler Angles to Rotation Matrix ---
def euler_to_rotmat(roll, pitch, yaw):
    return R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

