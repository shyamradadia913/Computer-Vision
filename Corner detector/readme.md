<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" />
<img src="https://img.shields.io/badge/Technique-Harris%20Corner%20Detection-blueviolet.svg" />
<img src="https://img.shields.io/badge/Type-Classical%20CV-yellow.svg" />
<img src="https://img.shields.io/badge/Environment-Jupyter%20Notebook-lightgrey.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# Corner Detection using Classical Computer Vision

## 📌 Overview

This project implements **corner detection** using classical computer vision techniques in OpenCV.

The notebook demonstrates how feature points (corners) can be detected in structured images such as:

- Chessboard patterns  
- Sudoku grids  

Corner detection is a fundamental building block in many vision systems including:

- Camera calibration  
- Object tracking  
- Image stitching  
- Structure from motion  
- Pattern recognition  

This implementation focuses on algorithmic understanding rather than deep learning.

---

## 🎯 Objective

The goal of this project is to:

- Load structured grid-based images  
- Apply preprocessing (grayscale conversion)  
- Detect corner features using classical algorithms  
- Visualize detected corner points  
- Analyze how structured geometry affects corner detection  

---

## 🖼 Images Used

### 1️⃣ Chessboard Image  
A high-contrast structured pattern ideal for detecting strong corners at square intersections.

### 2️⃣ Sudoku Grid Image  
A grid with numerical content used to observe corner detection behavior on real-world structured data.

---

## 🏗 Pipeline Structure

The notebook follows a structured computer vision workflow.

### 1️⃣ Image Loading
- Read image using OpenCV
- Convert to grayscale (corner detection operates on intensity gradients)

### 2️⃣ Preprocessing
- Image normalization (if required)
- Noise handling (optional smoothing)

### 3️⃣ Corner Detection Algorithm

Corner detection is performed using:

### 🔹 Harris Corner Detection

The Harris detector works by:

- Computing image gradients
- Constructing a second-moment matrix
- Measuring intensity variation in all directions
- Identifying points with strong variation (corners)

Corners are detected where intensity changes significantly in both x and y directions.

### 4️⃣ Thresholding
- Apply threshold to filter weak responses
- Keep only strong corner points

### 5️⃣ Visualization
- Mark detected corners on original image
- Display output using OpenCV or Matplotlib

---

## 🧠 Key Concepts Demonstrated

This project reinforces understanding of:

- Image gradients  
- Structure tensor  
- Eigenvalue interpretation in corner detection  
- Feature point extraction  
- Structured geometric patterns  
- Spatial intensity variation  

It builds intuition for why grid patterns (like chessboards) produce strong corner responses.

---

## 📊 Why Chessboard Works Well

Chessboards are commonly used in:

- Camera calibration  
- Pose estimation  

Because:

- They contain sharp intensity transitions  
- Corners are well-defined  
- Patterns are repetitive and structured  

Sudoku grids provide a more realistic test case where text and varying intensity exist.

---

## 🛠 Technologies Used

- Python 3  
- OpenCV  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

---

## 📂 Project Structure

```
Corner-Detection/
│
├── Corner detector.ipynb
├── chess-board-with-chess-figures-black-white-vector-illustration.jpg
├── sudoku.jpg
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
pip install opencv-python numpy matplotlib
```

### 3️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Run all cells sequentially.

---

## 📈 Expected Output

- Chessboard image with detected corner points highlighted  
- Sudoku grid with detected grid intersections marked  
- Visual comparison of corner density between images  

---

## 🔍 Classical Vision Insight

Corner detection is one of the earliest feature extraction techniques in computer vision.

Unlike deep learning approaches:

- No training data is required  
- No model weights are learned  
- Detection is purely mathematical  

It is lightweight, interpretable, and deterministic.

---

## 🚀 Possible Improvements

- Implement Shi-Tomasi corner detection  
- Compare Harris vs FAST detector  
- Add sub-pixel corner refinement  
- Perform camera calibration using chessboard corners  
- Use detected corners for homography estimation  

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- How classical feature detectors work  
- Why gradients matter in vision  
- How geometric patterns influence detection  
- Practical implementation of Harris detector  
- Visualization of feature points  

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com

---

## 📌 Final Note

This project demonstrates foundational feature detection techniques used in classical computer vision.

Understanding corner detection strengthens your grasp of:

- Image geometry  
- Feature extraction  
- Spatial intensity analysis  

These concepts remain relevant even in modern deep learning-based vision systems.
