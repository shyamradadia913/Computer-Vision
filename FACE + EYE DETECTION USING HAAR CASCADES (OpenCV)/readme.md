<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" />
<img src="https://img.shields.io/badge/Technique-Haar%20Cascade-blueviolet.svg" />
<img src="https://img.shields.io/badge/Type-Classical%20CV-yellow.svg" />
<img src="https://img.shields.io/badge/Detection-Face%20%2B%20Eye-orange.svg" />
<img src="https://img.shields.io/badge/Environment-Jupyter%20Notebook-lightgrey.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# Face + Eye Detection using Haar Cascades (OpenCV)

## 📌 Overview

This project implements real-time **Face and Eye Detection** using classical computer vision techniques with OpenCV’s Haar Cascade classifiers.

Unlike deep learning-based detection models, this approach uses:

- Pre-trained Haar features
- Integral images
- Cascade classifiers
- Sliding window detection

The system detects faces in an image or webcam feed and then detects eyes within each detected face region.

---

## 🎯 Objective

The goal of this project is to:

- Load pre-trained Haar cascade classifiers
- Detect faces in images or real-time webcam frames
- Detect eyes inside each detected face
- Draw bounding boxes around detected regions
- Visualize results in real-time

This project demonstrates how classical object detection was implemented before deep learning models like YOLO and RetinaNet became dominant.

---

## 🏗 Pipeline Structure

The notebook follows a structured detection workflow.

### 1️⃣ Load Haar Cascade Classifiers

Pre-trained XML classifiers are loaded:

- `haarcascade_frontalface_default.xml`
- `haarcascade_eye.xml`

These classifiers are trained using:

- Haar-like features
- AdaBoost
- Cascade architecture

---

### 2️⃣ Image Acquisition

Two possible modes:

- Static image detection
- Real-time webcam detection

Frames are captured and converted to grayscale since Haar cascades operate on intensity values.

---

### 3️⃣ Face Detection

Faces are detected using:

```
detectMultiScale()
```

Key parameters:

- `scaleFactor`
- `minNeighbors`
- `minSize`

The function scans the image at multiple scales to detect faces of different sizes.

---

### 4️⃣ Eye Detection (Within Face ROI)

For each detected face:

- Region of Interest (ROI) is extracted
- Eye detection is applied only inside the face area
- Bounding boxes are drawn for detected eyes

This hierarchical detection reduces false positives.

---

### 5️⃣ Visualization

Detected regions are highlighted:

- Blue rectangle → Face
- Green rectangle → Eyes

Real-time frames are displayed until user exits.

---

## 🧠 Key Concepts Demonstrated

This project reinforces understanding of:

- Haar-like features
- Integral images
- Cascade classifiers
- Multi-scale detection
- Region of Interest (ROI) processing
- Real-time video frame handling
- Classical object detection pipeline

---

## 🔍 How Haar Cascade Works (Conceptual)

Haar Cascade detection works by:

1. Computing simple rectangular features
2. Using integral images for fast computation
3. Selecting important features via AdaBoost
4. Organizing classifiers in cascade stages
5. Rejecting non-face regions quickly

Advantages:

- Lightweight
- No GPU required
- Real-time on CPU

Limitations:

- Sensitive to lighting
- Struggles with profile faces
- Not robust like modern CNN detectors

---

## 🛠 Technologies Used

- Python 3
- OpenCV
- NumPy
- Jupyter Notebook

---

## 📂 Project Structure

```
Face-Eye-Detection-Haar/
│
├── FACE + EYE DETECTION USING HAAR CASCADES (OpenCV).ipynb
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
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
pip install opencv-python numpy
```

### 3️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Run all cells sequentially.

If webcam mode is enabled, press `q` to exit.

---

## 📊 Expected Output

- Faces detected with bounding boxes
- Eyes detected within face regions
- Real-time webcam detection (if enabled)

---

## ⚖ Classical vs Deep Learning Detection

| Haar Cascade | Deep Learning (CNN-based) |
|--------------|---------------------------|
| No training required (pre-trained XML) | Requires large labeled dataset |
| Lightweight | Computationally expensive |
| CPU friendly | Often requires GPU |
| Less robust | Highly accurate & robust |

This project focuses on understanding the classical detection pipeline.

---

## 🚀 Possible Improvements

- Add smile detection
- Add face tracking
- Compare Haar vs DNN-based detector
- Implement face recognition pipeline
- Improve detection under low lighting
- Add FPS counter for performance evaluation

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- Classical object detection principles
- Real-time video processing
- ROI-based hierarchical detection
- Multi-scale sliding window approach
- Haar feature-based classification

---

## 👤 Author

**Shyam**  
AI & Data Science  

---

## 📌 Final Note

This project demonstrates foundational object detection using Haar cascades.

Although modern systems use deep learning, understanding classical methods strengthens your intuition about:

- Feature extraction
- Sliding window detection
- Real-time computer vision systems
