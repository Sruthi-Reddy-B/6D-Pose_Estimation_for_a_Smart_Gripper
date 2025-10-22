import os
from ultralytics import YOLO
from utils import ensure_dir

# --- Config ---
DATA_PATH = "../data/images"
MODEL_SAVE_PATH = "../results/yolo_model"

ensure_dir(MODEL_SAVE_PATH)

# --- Load Pre-trained YOLOv8 Model ---
model = YOLO("yolov8n.pt")  # start from pretrained weights

# --- Train ---
# Note: Using a tiny dataset, just for demo purposes
model.train(data="../data/dataset.yaml",
            imgsz=640,
            epochs=5,
            batch=2,
            project="../results",
            name="6d_pose_yolo_train",
            exist_ok=True)

# --- Save Trained Model ---
model_path = os.path.join(MODEL_SAVE_PATH, "yolov8_6d_pose.pt")
model.save(model_path)
print(f"Trained model saved at: {model_path}")

