
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import draw_axes, euler_to_rotmat, ensure_dir

# --- Paths ---
#MODEL_PATH = "results/yolo_model/yolov8_6d_pose.pt"

MODEL_PATH = "yolov8n.pt"  # pre-trained YOLOv8 downloaded in repo root

#IMAGE_PATH = "./data/images/object_1.jpg"
#POSE_CSV = "./data/poses/pose_labels.csv"
#OUTPUT_DIR = "./results/visualization"

ensure_dir(OUTPUT_DIR)

# --- Load YOLO Model ---
model = YOLO(MODEL_PATH)

# --- Predict ---
results = model.predict(source=IMAGE_PATH, save=False)
img = cv2.imread(IMAGE_PATH)

# --- Load Ground-truth / Dummy Pose Labels ---
pose_data = pd.read_csv(POSE_CSV)
pose = pose_data.iloc[0]  # assume single image for demo

# --- Construct Pose Matrix ---
x, y, z = pose['x'], pose['y'], pose['z']
roll, pitch, yaw = pose['roll'], pose['pitch'], pose['yaw']
R_mat = euler_to_rotmat(roll, pitch, yaw)
pose_matrix = np.hstack((R_mat, np.array([[x],[y],[z]])))

print("Estimated 6D Pose Matrix:\n", pose_matrix)

# --- Draw Axes ---
img_pose = draw_axes(img.copy(), (x, y), R_mat)
out_path = os.path.join(OUTPUT_DIR, "pose_axes.jpg")
cv2.imwrite(out_path, img_pose)
print(f"Pose visualization saved at: {out_path}")
