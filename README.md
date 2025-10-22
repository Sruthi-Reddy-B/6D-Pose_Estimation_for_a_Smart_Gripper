# 6D Pose Estimation for Smart Gripper

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sruthi-Reddy-B/6d-pose-estimation-smart-gripper/blob/main/notebooks/pose_estimation_pipeline.ipynb)

##  Overview
This project implements a **vision-based 6D pose estimation pipeline** for a robotic gripper, inspired by my Master's thesis  
**“Vision-Driven Automation: Smart Gripper Development for Autonomous Material Manipulation.”**

It detects an object using **YOLOv8** and estimates its **6D pose (position + orientation)** to assist robotic manipulation.

---

##  Tech Stack
- **Frameworks:** PyTorch, Ultralytics YOLOv8  
- **Libraries:** OpenCV, NumPy, Matplotlib, SciPy, Pandas  
- **Environment:** Google Colab (GPU recommended)

---

##  Folder Structure
```text
6d-pose-estimation-smart-gripper/
├── data/           # RGB images and 6D pose labels
├── notebooks/      # Pipeline notebook
├── src/            # Training/inference scripts
├── results/        # Detections and pose visualizations
└── requirements.txt
```

---

##  How to Run

### 1️. Using Google Colab (Recommended)
1. Click the **Colab badge** at the top.
2. 2. Run all notebook cells sequentially.  
3. The notebook performs detection → pose estimation → visualization. Results are saved in /results/.

### 2️. Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook for training
jupyter notebook notebooks/pose_estimation_pipeline.ipynb
```

---

##  Results
Object detected successfully with YOLOv8.
6D pose matrix estimated (x, y, z + rotation R).
Visualization overlays local coordinate axes on detected object.

These outputs simulate the workflow of an industrial vision-gripper system.

---

##  Next Steps / Enhancements

Replace synthetic data with real RGB-D images.
Integrate with ROS 2 for real-time robot grasping.

Use YOLOv5-6D or PoseCNN for accurate orientation recovery.

Add point-cloud-based refinement (Gaussian splatting / NeRF).
