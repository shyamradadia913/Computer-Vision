<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" />
<img src="https://img.shields.io/badge/Model-Faster%20R--CNN-blueviolet.svg" />
<img src="https://img.shields.io/badge/Backbone-ResNet50-yellow.svg" />
<img src="https://img.shields.io/badge/Dataset-COCO-lightgrey.svg" />
<img src="https://img.shields.io/badge/Task-Object%20Detection-success.svg" />

</p>

# Faster R-CNN Object Detection using TensorFlow Hub

## 📌 Overview

This project performs **object detection using a pre-trained Faster R-CNN model** with a ResNet50 backbone.

The model is loaded from **TensorFlow Hub** and applied to custom images for real-time inference.

The system detects objects from **90 COCO classes** and outputs:

- Bounding boxes  
- Class labels  
- Confidence scores  

---

## 🎯 Objective

The goal of this project is to:

- Perform object detection using a state-of-the-art pretrained model  
- Understand Faster R-CNN pipeline (RPN → ROI → Classification)  
- Filter detections based on confidence threshold  
- Visualize detection results with bounding boxes  
- Build a complete inference pipeline  

---

## 🏗 Detection Pipeline

The project follows a structured object detection workflow:

### 1️⃣ Load Model
- Faster R-CNN (ResNet50 backbone)
- Loaded from TensorFlow Hub
- Pretrained on COCO dataset

---

### 2️⃣ Image Preprocessing
- Load image using OpenCV  
- Convert BGR → RGB  
- Convert to tensor `[1, H, W, 3]`  
- Ensure correct datatype (`uint8`)  

---

### 3️⃣ Model Inference

The model internally performs:

- Region Proposal Network (RPN)  
- ROI Align  
- Bounding box regression  
- Classification  
- Non-Maximum Suppression (NMS)  

---

### 4️⃣ Filtering Detections

- Apply confidence threshold (`MIN_CONF`)  
- Remove weak predictions  
- Keep only relevant detections  

---

### 5️⃣ Visualization

- Draw bounding boxes  
- Assign unique colors per object  
- Display labels + confidence scores  
- Save output image  

---

## 🧠 Model Details

- Model: Faster R-CNN  
- Backbone: ResNet50  
- Input Size: Dynamic  
- Dataset: COCO 2017  
- Classes: 90 object categories  

Examples of detectable objects:

- Person, Car, Dog, Cat  
- Laptop, Phone, Chair  
- Bottle, Cup, Table  
- And many more  

---

## 📊 Output

The model produces:

- Bounding box coordinates (normalized)  
- Class IDs → mapped to labels  
- Confidence scores  
- Annotated output image  

---

## 🛠 Technologies Used

- Python 3  
- TensorFlow  
- TensorFlow Hub  
- OpenCV  
- NumPy  
- Matplotlib  

---

## 📂 Project Structure

```
Faster-RCNN-Detection/
│
├── L3-2(R-CNN).py
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install tensorflow tensorflow-hub opencv-python matplotlib numpy
```

---

### 2️⃣ Set image path

Update:

```python
IMAGE_PATH = "your_image.jpg"
```

---

### 3️⃣ Run script

```bash
python L3-2(R-CNN).py
```

---

## ⚙️ Configuration

| Parameter | Description |
|----------|------------|
| `IMAGE_PATH` | Input image path |
| `MIN_CONF` | Minimum confidence threshold |
| `OUTPUT_PATH` | Output image file |

---

## 📈 Expected Output

- Console output of detected objects  
- Bounding boxes drawn on image  
- Labels with confidence scores  
- Saved annotated image  

---

## ⚖ Faster R-CNN vs YOLO

| Faster R-CNN | YOLO |
|--------------|------|
| Two-stage detector | One-stage detector |
| Higher accuracy | Faster inference |
| Slower | Real-time |
| Region-based | Grid-based |

This project focuses on **accuracy-oriented detection**.

---

## 🚀 Possible Improvements

- Add real-time webcam detection  
- Compare with YOLOv8 results  
- Add batch image processing  
- Implement detection on video streams  
- Add IoU-based filtering  

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- Region-based object detection  
- Pretrained model usage (TF Hub)  
- COCO dataset structure  
- Bounding box processing  
- Confidence filtering  
- Visualization of detection results  

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com   

---

## 📌 Final Note

This project demonstrates practical usage of a **production-grade object detection model**.

It highlights how complex pipelines like Faster R-CNN can be:

- Loaded directly  
- Applied to real-world images  
- Interpreted and visualized effectively  

A critical step toward real-world AI deployment.
