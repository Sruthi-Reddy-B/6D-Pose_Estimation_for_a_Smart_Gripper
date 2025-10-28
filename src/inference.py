
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import os

# --- Load Pretrained YOLO ---
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# --- Input Paths ---
image_path = "./data/images/object_1.jpg"
pose_csv = "./data/poses/pose_labels.csv"
output_dir = "./results/visualization"

os.makedirs(output_dir, exist_ok=True)

# --- Run Object Detection ---
results = model.predict(source=image_path, save=True, project='./results/detections', name='pred', verbose=False)

# --- Load Image ---
img = cv2.imread(image_path)
height, width = img.shape[:2]

# --- Load Dummy Pose ---
pose_data = pd.read_csv(pose_csv)
x, y, z = pose_data.iloc[0][['x', 'y', 'z']]
roll, pitch, yaw = pose_data.iloc[0][['roll', 'pitch', 'yaw']]

# --- Compute Rotation Matrix ---
rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
pose_matrix = np.hstack((rotation_matrix, np.array([[x], [y], [z]])))
print("Estimated 6D Pose Matrix:\n", pose_matrix)

# --- Draw 3D Axes ---
def draw_axes(img, origin, R_mat, length=120):
    """
    Draw 3D coordinate axes on a 2D image.
    Red = X-axis, Green = Y-axis, Blue = Z-axis
    """
    cx, cy = int(origin[0]), int(origin[1])
    axes_3d = np.float32([[length,0,0],[0,length,0],[0,0,length]])
    colors = [(0,0,255), (0,255,0), (255,0,0)]  # X, Y, Z

    for vec, col in zip(axes_3d, colors):
        proj = R_mat @ vec
        pt = (int(cx + proj[0]), int(cy - proj[1]))  # flip Y-axis
        cv2.line(img, (cx, cy), pt, col, 3, cv2.LINE_AA)
    return img

# --- Choose Object Center ---
if len(results[0].boxes.xyxy) > 0:
    box = results[0].boxes.xyxy[0].cpu().numpy()
    label = results[0].names[int(results[0].boxes.cls[0])]
    conf = float(results[0].boxes.conf[0])
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    # Draw bounding box + label
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
    cv2.putText(img, f"{label} ({conf:.2f})", (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
else:
    cx, cy = width // 2, height // 2

# --- Draw Axes at Center ---
img_pose = draw_axes(img.copy(), (cx, cy), rotation_matrix)

# --- Save Output ---
save_path = os.path.join(output_dir, "pose_axes_box.jpg")
cv2.imwrite(save_path, img_pose)
print(f"Pose visualization with bounding box saved at: {save_path}")

'''

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import draw_axes, euler_to_rotmat, ensure_dir

# --- Paths ---
#MODEL_PATH = "results/yolo_model/yolov8_6d_pose.pt"

MODEL_PATH = "yolov8n.pt"  # pre-trained YOLOv8 downloaded in repo root

IMAGE_PATH = "./data/images/object_1.jpg"
POSE_CSV = "./data/poses/pose_labels.csv"
OUTPUT_DIR = "./results/visualization"

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
'''
