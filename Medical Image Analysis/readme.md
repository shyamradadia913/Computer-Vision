<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg" />
<img src="https://img.shields.io/badge/Model-EfficientNetB0-blueviolet.svg" />
<img src="https://img.shields.io/badge/Task-Medical%20Image%20Analysis-red.svg" />
<img src="https://img.shields.io/badge/Explainability-GradCAM-yellow.svg" />
<img src="https://img.shields.io/badge/Status-Production--Level-success.svg" />

</p>

# 🩺 Breast Cancer Detection using EfficientNetB0 + Grad-CAM

## 📌 Overview

This project implements a **production-level medical image classification system** for detecting breast cancer using ultrasound images.

It goes beyond simple classification by integrating:

- Transfer Learning (EfficientNetB0)  
- Fine-tuning strategies  
- Class imbalance handling  
- Advanced evaluation metrics  
- Model explainability using Grad-CAM  

This pipeline simulates a **real-world AI-assisted diagnostic system**.

---

## 🎯 Objective

The goal of this project is to:

- Classify ultrasound images into:
  - **Benign**
  - **Malignant**
  - **Normal**
- Improve diagnostic performance using transfer learning  
- Handle imbalanced medical datasets  
- Provide interpretable predictions using Grad-CAM  
- Build a robust and reproducible training pipeline  

---

## 🧠 Model Architecture

- Backbone: **EfficientNetB0 (ImageNet pretrained)**
- Custom Head:
  - Global Average Pooling  
  - Dense layers (512 → 256)  
  - Batch Normalization  
  - Dropout regularization  
- Output: Softmax (3 classes)

---

## 🏗 Training Strategy

### Phase 1 — Feature Extraction
- Freeze base EfficientNet layers  
- Train only classification head  
- Faster convergence  

---

### Phase 2 — Fine-Tuning
- Unfreeze last **30 layers** of backbone  
- Keep BatchNorm layers frozen  
- Lower learning rate  
- Improves feature specialization  

---

## ⚙️ Key Techniques Used

### 1️⃣ Data Augmentation
- Rotation  
- Zoom  
- Brightness variation  
- Horizontal flipping  

---

### 2️⃣ Class Imbalance Handling
- Computed class weights  
- Penalizes misclassification of minority classes  

---

### 3️⃣ Regularization
- Dropout  
- Label smoothing  

---

### 4️⃣ Callbacks
- EarlyStopping  
- ModelCheckpoint  
- ReduceLROnPlateau  
- TensorBoard  

---

## 📊 Performance

| Metric | Value |
|------|------|
| Validation AUC | **0.9643** |
| Accuracy | **~85%** |
| Malignant AUC | **0.9282** |

👉 Focus is on **AUC**, not just accuracy (critical in medical tasks)

---

## 🔬 Model Explainability (Grad-CAM)

This project includes a **fully fixed and advanced Grad-CAM implementation**.

### Why it matters:
- Shows **where the model is looking**
- Builds trust in AI predictions
- Essential in healthcare applications

### Key Engineering Achievement:
- Solved **TensorFlow graph boundary issues**
- Implemented **forward-hook based Grad-CAM**
- Works reliably even after `load_model()`

---

## 📈 Outputs Generated

- 📊 Confusion Matrix  
- 📉 ROC Curves  
- 📊 Training History Graphs  
- 🔥 Grad-CAM Heatmaps  
- 💾 Saved models (`.keras`)  

---

## 🧠 Key Concepts Demonstrated

- Transfer Learning  
- Fine-Tuning Strategy  
- Medical Image Classification  
- Class Imbalance Handling  
- AUC-based evaluation  
- Explainable AI (XAI)  
- TensorFlow debugging & graph mechanics  

---

## 🛠 Technologies Used

- Python 3  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📂 Project Structure

```
Breast-Cancer-Detection/
│
├── model_training.ipynb
├── best_model.keras
├── final_model.keras
├── confusion_matrix.png
├── roc_curves.png
├── training_history.png
├── gradcam_samples.png
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Enable GPU (Colab recommended)

```
Runtime → Change runtime type → GPU
```

---

### 2️⃣ Install dependencies

```bash
pip install tensorflow opencv-python seaborn scikit-learn
```

---

### 3️⃣ Add Kaggle credentials

```python
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"
```

---

### 4️⃣ Run notebook

Execute all cells.

---

## 📊 Expected Output

- High AUC performance  
- Balanced classification  
- Visual explanations via Grad-CAM  
- Saved trained models  

---

## ⚠ Important Notes

- This is a **research/demo system**, not a medical device  
- Dataset limitations may affect generalization  
- Clinical validation is required for real-world use  

---

## 🚀 Possible Improvements

- Use larger datasets (e.g., multi-hospital data)  
- Add segmentation before classification  
- Deploy as web-based diagnostic tool  
- Integrate clinical metadata  

---

## 🎓 Learning Outcomes

By completing this project, you gain:

- Deep understanding of transfer learning  
- Handling real-world medical datasets  
- Model evaluation beyond accuracy  
- Explainable AI techniques  
- Debugging complex TensorFlow issues  

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com 

---

## 📌 Final Note

This project demonstrates how deep learning can be applied to **high-stakes domains like healthcare**.

It combines:

- Performance  
- Robust engineering  
- Explainability  

— making it closer to real-world AI systems than typical academic projects.
