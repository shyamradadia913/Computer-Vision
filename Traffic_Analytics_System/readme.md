<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" />
<img src="https://img.shields.io/badge/System-Video%20Analytics-blueviolet.svg" />
<img src="https://img.shields.io/badge/Tracking-Multi--Object-orange.svg" />
<img src="https://img.shields.io/badge/Output-Analytics%20%2B%20CSV-yellow.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# Advanced Traffic Monitoring & Video Analytics System

## 📌 Overview

This project implements a **complete real-time traffic analytics system** using classical computer vision techniques.

Unlike basic detection systems, this pipeline performs:

- Multi-object tracking  
- Directional vehicle counting  
- Traffic heatmap generation  
- Dwell time analysis (congestion measurement)  
- Data export for analytics  

This is a **full system-level implementation**, not just object detection.

---

## 🎯 Objective

The goal of this project is to build an intelligent traffic monitoring system that can:

- Detect moving vehicles  
- Track vehicles across frames  
- Count directional movement (entries & exits)  
- Analyze congestion patterns  
- Generate visual + numerical insights  

---

## 🚀 Key Features

### 1️⃣ Directional Counting
- Counts vehicles moving:
  - DOWN → Entries  
  - UP → Exits  
- Uses line-crossing logic with buffer zone  

---

### 2️⃣ Smart Debouncing
- Prevents duplicate counting  
- Uses:
  - Track maturity (`min_hits`)  
  - Unique ID tracking  
- Ensures accuracy even when vehicles pause  

---

### 3️⃣ Motion Heatmap
- Accumulates motion over time  
- Highlights high-traffic zones  
- Color-coded visualization:
  - Blue → Low activity  
  - Red → High congestion  

---

### 4️⃣ Dwell Time Analysis
- Measures how long vehicles stay on screen  
- Indicates:
  - Traffic density  
  - Congestion levels  

---

### 5️⃣ Data Export (CSV)
- Logs frame-by-frame analytics:
  - Active vehicles  
  - Entries  
  - Exits  

- Enables:
  - Excel analysis  
  - Dashboard integration  

---

## 🏗 System Pipeline

### 1️⃣ Video Input
- Load video stream  
- Resize for performance  

---

### 2️⃣ Background Subtraction (MOG2)
- Separates moving objects from static background  
- Learns scene over time  

---

### 3️⃣ Image Processing

- Median Blur → smooth noise  
- Thresholding → remove shadows  
- Morphological operations:
  - Opening → remove noise  
  - Closing → connect broken regions  

---

### 4️⃣ Object Detection
- Contour detection  
- Area filtering (ignore small objects)  
- Bounding box extraction  

---

### 5️⃣ Multi-Object Tracking

Custom tracker using:

- Centroid matching  
- Distance threshold  
- Track aging  
- Hit-based validation  

Each vehicle gets a **unique ID**.

---

### 6️⃣ Counting Logic

- Virtual counting line  
- Buffer zone for accuracy  
- Direction-based detection:
  - Above → Below → Entry  
  - Below → Above → Exit  

---

### 7️⃣ Analytics Layer

- Heatmap accumulation  
- Dwell time tracking  
- Vehicle count history  

---

### 8️⃣ Visualization

- Bounding boxes  
- Object IDs  
- Counting lines  
- Heatmap overlay  
- Live counters  

---

## 📊 Output

The system generates:

- 🎥 Processed video (`output_analytics.mp4`)  
- 📊 CSV data file (`traffic_data.csv`)  
- 📈 Congestion graph (`analytics_congestion.png`)  

---

## 🧠 Key Concepts Demonstrated

- Background subtraction (MOG2)  
- Morphological image processing  
- Contour detection  
- Multi-object tracking  
- Centroid tracking algorithm  
- Directional counting logic  
- Heatmap generation  
- Real-time analytics  

---

## 🛠 Technologies Used

- Python 3  
- OpenCV  
- NumPy  
- Matplotlib  
- CSV  

---

## 📂 Project Structure

```
Traffic-Analytics-System/
│
├── traffic_analytics.py
├── input_video.mp4
├── output_analytics.mp4
├── traffic_data.csv
├── analytics_congestion.png
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

### 2️⃣ Set video path

```python
VIDEO_PATH = "your_video.mp4"
```

---

### 3️⃣ Run script

```bash
python traffic_analytics.py
```

---

## ⚙️ Key Parameters

| Parameter | Description |
|----------|------------|
| `MIN_CONTOUR_AREA` | Filters small objects |
| `MAX_MATCH_DIST` | Tracking distance threshold |
| `LINE_BUFFER` | Prevents false crossings |
| `WARMUP_FRAMES` | Background learning phase |

---

## 📈 Expected Output

- Real-time tracking of vehicles  
- Direction-based counting  
- Heatmap overlay  
- CSV analytics data  
- Traffic congestion graph  

---

## ⚖ Classical vs Deep Learning Approach

| Classical CV | Deep Learning |
|--------------|--------------|
| Lightweight | Heavy models |
| Real-time CPU | GPU required |
| No training | Needs dataset |
| Less robust | More accurate |

This project focuses on **efficiency + analytics**.

---

## 🚀 Possible Improvements

- Replace detection with YOLOv8  
- Add speed estimation  
- Multi-lane tracking  
- License plate recognition  
- Deploy as live camera system  

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- End-to-end vision system design  
- Real-time tracking systems  
- Data-driven video analytics  
- Traffic pattern analysis  
- Practical deployment considerations  

---

## 👤 Author

**Shyam**  
AI & Data Science  

---

## 📌 Final Note

This project demonstrates how computer vision can move beyond detection into **analytics and decision-making systems**.

It combines:

- Detection  
- Tracking  
- Analytics  
- Visualization  

Into a single pipeline — closer to real-world industry applications.