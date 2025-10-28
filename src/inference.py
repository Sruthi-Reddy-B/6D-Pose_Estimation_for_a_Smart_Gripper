import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# --- Paths ---
MODEL_PATH = "./yolov8n.pt"
IMAGE_PATH = "./data/images/object_1.jpg"
POSE_PATH = "./data/poses/pose_labels.csv"
SAVE_PATH = "./results/visualization/pose_axes.jpg"

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)
result = results[0]
img = cv2.imread(IMAGE_PATH)

# --- Pick the largest bounding box (main object) ---
if len(result.boxes) == 0:
    raise ValueError("No objects detected by YOLO!")

boxes = result.boxes.xyxy.cpu().numpy()
areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
largest_idx = np.argmax(areas)
x1, y1, x2, y2 = boxes[largest_idx]
center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# --- Load Pose Data ---
pose_data = pd.read_csv(POSE_PATH)
row = pose_data.iloc[0]

# --- Extract translation + orientation ---
t = np.array([row["x"], row["y"], row["z"]])
roll, pitch, yaw = np.deg2rad([row["roll"], row["pitch"], row["yaw"]])

# --- Rotation matrices ---
Rx = np.array([[1, 0, 0],
               [0, np.cos(roll), -np.sin(roll)],
               [0, np.sin(roll),  np.cos(roll)]])
Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
               [0, 1, 0],
               [-np.sin(pitch), 0, np.cos(pitch)]])
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
               [np.sin(yaw),  np.cos(yaw), 0],
               [0, 0, 1]])

# --- Combined rotation (roll -> pitch -> yaw) ---
R = Rz @ Ry @ Rx

# --- Pose matrix ---
pose_matrix = np.hstack((R, t.reshape(3,1)))
print("\nEstimated 6D Pose Matrix:\n", pose_matrix)

# --- Function to project 3D axis to 2D image ---
def project_axis(R_col, origin, length=100):
    ox, oy = origin
    # Flip Y axis for image coordinates
    x2 = int(ox + R_col[0]*length)
    y2 = int(oy - R_col[1]*length)
    return (x2, y2)

# --- Draw 3D axes ---
axis_length = 100
origin = (center_x, center_y)

x_axis = project_axis(R[:,0], origin, axis_length)
y_axis = project_axis(R[:,1], origin, axis_length)
z_axis = project_axis(R[:,2], origin, axis_length)

cv2.line(img, origin, x_axis, (0, 0, 255), 4)   # X - Red
cv2.line(img, origin, y_axis, (0, 255, 0), 4)   # Y - Green
cv2.line(img, origin, z_axis, (255, 0, 0), 4)   # Z - Blue

# --- Save result ---
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
cv2.imwrite(SAVE_PATH, img)
print(f"\nPose visualization saved at: {SAVE_PATH}")

'''import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# --- Paths ---
MODEL_PATH = "./yolov8n.pt"   # pre-trained YOLOv8 model
IMAGE_PATH = "./data/images/object_1.jpg"
POSE_PATH = "./data/poses/pose_labels.csv"
SAVE_PATH = "./results/visualization/pose_axes.jpg"

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)

# --- Run inference ---
results = model(IMAGE_PATH)
result = results[0]
img = cv2.imread(IMAGE_PATH)

# --- Draw 2D bounding boxes ---
for box in result.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- Load Pose Data ---
pose_data = pd.read_csv(POSE_PATH)
row = pose_data.iloc[0]

# --- Extract translation ---
t = np.array([row["x"], row["y"], row["z"]])

# --- Convert roll, pitch, yaw (in degrees) to radians ---
roll, pitch, yaw = np.deg2rad([row["roll"], row["pitch"], row["yaw"]])

# --- Compute rotation matrix (R = Rz * Ry * Rx) ---
Rx = np.array([[1, 0, 0],
               [0, np.cos(roll), -np.sin(roll)],
               [0, np.sin(roll),  np.cos(roll)]])
Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
               [0, 1, 0],
               [-np.sin(pitch), 0, np.cos(pitch)]])
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
               [np.sin(yaw),  np.cos(yaw), 0],
               [0, 0, 1]])

R = Rz @ Ry @ Rx

# --- Form 3x4 pose matrix ---
pose_matrix = np.hstack((R, t.reshape(3,1)))
print("\nEstimated 6D Pose Matrix:\n", pose_matrix)

# --- Draw 3D Axes from pose matrix ---
axis_length = 50  # adjust for visibility

# Origin in 2D image (approximate projection)
origin = (int(t[0]), int(t[1]))

# Endpoints for X, Y, Z axes (using rotation directions)
x_axis = (int(t[0] + R[0,0]*axis_length), int(t[1] + R[1,0]*axis_length))
y_axis = (int(t[0] + R[0,1]*axis_length), int(t[1] + R[1,1]*axis_length))
z_axis = (int(t[0] + R[0,2]*axis_length), int(t[1] + R[1,2]*axis_length))

# Draw axes on image
cv2.line(img, origin, x_axis, (0, 0, 255), 3)   # X - Red
cv2.line(img, origin, y_axis, (0, 255, 0), 3)   # Y - Green
cv2.line(img, origin, z_axis, (255, 0, 0), 3)   # Z - Blue

# --- Save visualization ---
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
cv2.imwrite(SAVE_PATH, img)
print(f"\nPose visualization saved at: {SAVE_PATH}")

'''
