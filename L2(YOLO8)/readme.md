<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg" />
<img src="https://img.shields.io/badge/Task-Object%20Detection-blueviolet.svg" />
<img src="https://img.shields.io/badge/Framework-Deep%20Learning-yellow.svg" />
<img src="https://img.shields.io/badge/Model-CNN%20Based-lightgrey.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# Object Detection using YOLOv8

## 📌 Overview

This project implements **object detection using YOLOv8 (You Only Look Once version 8)**.

The notebook demonstrates how to:

- Load and prepare a custom dataset
- Train a YOLOv8 detection model
- Perform inference on images
- Visualize bounding box predictions
- Evaluate detection results

YOLOv8 is a state-of-the-art real-time object detection model that performs detection in a single forward pass.

---

## 🎯 Objective

The goal of this project is to:

- Train a YOLOv8 model on a custom image dataset
- Detect objects with bounding boxes
- Generate confidence scores
- Visualize detected objects
- Understand real-time object detection workflow

---

## 🖼 Dataset

The dataset (`images.zip`) contains labeled images formatted for YOLO training.

Dataset structure typically includes:

```
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│
├── labels/
│   ├── train/
│   ├── val/
│
└── data.yaml
```

Each label file contains:

- Class ID
- Normalized bounding box coordinates
  - x_center
  - y_center
  - width
  - height

---

## 🏗 Project Workflow

The notebook follows a structured YOLO pipeline.

### 1️⃣ Install and Import YOLOv8

YOLOv8 is provided by the Ultralytics library.

```bash
pip install ultralytics
```

---

### 2️⃣ Load Pretrained Model

The model is initialized using:

- `yolov8n.pt` (Nano)
- Or another variant depending on usage

Pretrained weights provide transfer learning capabilities.

---

### 3️⃣ Train the Model

Training includes:

- Custom dataset configuration (`data.yaml`)
- Specifying number of epochs
- Batch size
- Image size
- Learning rate

YOLO performs:

- Feature extraction
- Bounding box regression
- Classification
- Confidence scoring

All in a single network.

---

### 4️⃣ Validation

The model is validated using:

- mAP (mean Average Precision)
- Precision
- Recall

These metrics evaluate detection performance.

---

### 5️⃣ Inference

After training:

- Images are passed through the model
- Bounding boxes are predicted
- Class labels and confidence scores are generated
- Results are visualized

---

## 🧠 Key Concepts Demonstrated

This project reinforces understanding of:

- One-stage object detection
- Bounding box regression
- Intersection over Union (IoU)
- Confidence scoring
- Non-Maximum Suppression (NMS)
- Transfer learning in detection models

---

## 🔍 Why YOLOv8?

YOLOv8 is:

- Fast (real-time capable)
- Accurate
- End-to-end trainable
- Efficient on GPU
- Suitable for custom datasets

Unlike classical sliding window detection, YOLO detects objects in one forward pass.

---

## 🛠 Technologies Used

- Python 3
- Ultralytics YOLOv8
- PyTorch (backend)
- NumPy
- OpenCV
- Jupyter Notebook

---

## 📂 Project Structure

```
YOLOv8-Object-Detection/
│
├── Yolo8.ipynb
├── images.zip
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 3️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Run all cells sequentially.

If using GPU, ensure CUDA is properly configured.

---

## 📊 Expected Output

- Trained YOLOv8 model
- Detection results with bounding boxes
- Confidence scores
- Evaluation metrics (mAP, Precision, Recall)
- Visualized prediction images

---

## ⚖ Classical Detection vs YOLO

| Classical Detection | YOLOv8 |
|--------------------|--------|
| Sliding window | Single forward pass |
| Manual features | Learned features |
| Slow | Real-time |
| Less robust | Highly accurate |

This project demonstrates modern deep learning-based object detection.

---

## 🚀 Possible Improvements

- Use larger YOLOv8 variants (s, m, l, x)
- Increase dataset size
- Perform hyperparameter tuning
- Deploy as real-time webcam detection
- Convert model to ONNX / TensorRT
- Deploy on edge devices

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- Object detection pipelines
- YOLO architecture fundamentals
- Training custom detection models
- Evaluation using mAP
- Real-world deep learning deployment concepts

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com  

---

## 📌 Final Note

This project demonstrates modern object detection using YOLOv8.

It bridges foundational CNN knowledge with real-world detection systems used in:

- Surveillance
- Autonomous driving
- Robotics
- Industrial automation
- Smart retail
