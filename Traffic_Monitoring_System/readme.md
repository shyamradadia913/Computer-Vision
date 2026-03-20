<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" />
<img src="https://img.shields.io/badge/Technique-Background%20Subtraction-blueviolet.svg" />
<img src="https://img.shields.io/badge/System-Traffic%20Monitoring-yellow.svg" />
<img src="https://img.shields.io/badge/Tracking-Centroid%20Tracker-orange.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# Traffic Monitoring System using Classical Computer Vision

## 📌 Overview

This project implements a **traffic monitoring and vehicle counting system** using classical computer vision techniques.

The system detects moving vehicles in a video stream and tracks them across frames to perform accurate counting.

Unlike deep learning-based approaches, this implementation relies on:

- Background subtraction  
- Morphological processing  
- Object tracking (Centroid Tracker)  

This makes it lightweight and suitable for real-time CPU-based systems.

---

## 🎯 Objective

The goal of this project is to:

- Detect moving vehicles from video input  
- Track vehicles across frames  
- Count vehicles crossing a defined region  
- Reduce false detections using noise filtering  
- Build a reliable classical vision pipeline  

---

## 🏗 System Pipeline

The system follows a structured pipeline:

### 1️⃣ Video Input
- Load traffic video stream  
- Process frame-by-frame  

---

### 2️⃣ Background Subtraction

- Uses **MOG2 (Mixture of Gaussians)**  
- Separates moving objects from background  

Purpose:
- Detect motion regions (vehicles)

---

### 3️⃣ Noise Reduction

Applies morphological operations:

- Erosion  
- Dilation  
- Closing  

Purpose:
- Remove noise  
- Fill gaps in detected objects  
- Improve contour quality  

---

### 4️⃣ Contour Detection

- Extract contours from foreground mask  
- Filter small objects using area threshold  

Only valid moving objects are retained.

---

### 5️⃣ Object Tracking (Centroid Tracker)

- Assign unique IDs to detected vehicles  
- Track centroids across frames  
- Maintain object identity  

This prevents duplicate counting.

---

### 6️⃣ Vehicle Counting Logic

- Define a virtual counting line  
- Detect when object crosses the line  
- Increment count only once per vehicle  

---

### 7️⃣ Visualization

- Draw bounding boxes around vehicles  
- Display object IDs  
- Show total vehicle count  
- Real-time video output  

---

## 🧠 Key Concepts Demonstrated

This project reinforces understanding of:

- Background subtraction (MOG2)  
- Motion detection  
- Morphological image processing  
- Contour filtering  
- Object tracking  
- Centroid-based tracking algorithms  
- Real-time video processing  

---

## ⚙️ Accuracy Improvements

The system includes optimizations such as:

- Noise reduction using morphological operations  
- Area-based filtering to remove small objects  
- Stable tracking to prevent double counting  
- Improved contour detection  

These improve detection reliability in real-world scenarios.

---

## 🛠 Technologies Used

- Python 3  
- OpenCV  
- NumPy  

---

## 📂 Project Structure

```
Traffic-Monitoring-System/
│
├── traffic_monitoring.ipynb
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install opencv-python numpy
```

---

### 2️⃣ Run notebook

```bash
jupyter notebook
```

Execute all cells.

---

## 📊 Expected Output

- Vehicles detected in video  
- Bounding boxes with IDs  
- Real-time tracking visualization  
- Accurate vehicle count displayed  

---

## ⚖ Classical vs Deep Learning Traffic Detection

| Classical Approach | Deep Learning |
|-------------------|--------------|
| Lightweight | Heavy models |
| CPU-friendly | Requires GPU |
| Fast setup | Needs training |
| Less robust in complex scenes | More accurate |

This project focuses on **efficient real-time processing**.

---

## 🚀 Possible Improvements

- Replace with YOLO-based detection  
- Add lane-wise counting  
- Implement speed estimation  
- Handle occlusion better  
- Deploy as real-time system (camera feed)  

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- Motion-based object detection  
- Tracking algorithms  
- Real-time video pipelines  
- Practical system design in computer vision  
- Trade-offs between classical and deep learning approaches  

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com 

---

## 📌 Final Note

This project demonstrates how a complete **real-time monitoring system** can be built using classical computer vision.

It highlights:

- Efficiency  
- Simplicity  
- Practical deployment potential  

A strong foundation before moving to deep learning-based tracking systems.
