# Object-detection-with-pose-visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sruthi-Reddy-B/object-detection-with-pose-visualization/blob/main/notebooks/pose_estimation_pipeline.ipynb)

##  Overview
This project demonstrates a **vision-based workflow for robotic grasping**.  
It detects objects using **YOLOv8** and visualizes their **pose (position + orientation)** using sample/demo data.  

> Note: This is a **demo pipeline**. The pose is **synthetic / simulated**.  
> True 6D pose prediction requires training on datasets like LINEMOD or YCB-Video.

---

##  Tech Stack
- **Frameworks:** PyTorch, Ultralytics YOLOv8  
- **Libraries:** OpenCV, NumPy, Matplotlib, Pandas  
- **Environment:** Google Colab (GPU recommended)

---

##  Folder Structure
```text
Object-detection-with-pose-visualization/
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
Detected object with YOLOv8
Visualized axes for synthetic pose
Saved predicted pose matrix in CSV

---

##  Next Steps / Enhancements

-Train on real 6D pose datasets for true position + orientation prediction
-Integrate with ROS2 for real-time grasping
-Upgrade to YOLOv5-6D or PoseCNN for accurate orientation
-Use point-cloud / NeRF / Gaussian splatting for more advanced robotic perception
