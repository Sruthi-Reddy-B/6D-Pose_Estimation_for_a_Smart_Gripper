import cv2
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
