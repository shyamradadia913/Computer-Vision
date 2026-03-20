<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg" />
<img src="https://img.shields.io/badge/Model-FaceNet%20(Simplified)-blueviolet.svg" />
<img src="https://img.shields.io/badge/Loss-Triplet%20Loss-yellow.svg" />
<img src="https://img.shields.io/badge/Task-Facial%20Recognition-green.svg" />
<img src="https://img.shields.io/badge/Mode-Webcam%20Demo-success.svg" />

</p>

# Facial Recognition System using FaceNet & Triplet Loss

## 📌 Overview

This project implements a **facial recognition system** using a simplified version of **FaceNet** with **Triplet Loss**.

Instead of classifying faces directly, the model learns to convert each face into a **128-dimensional embedding (feature vector)**.

Recognition is then performed by comparing distances between embeddings.

---

## 🎯 Objective

The goal of this project is to:

- Learn how modern facial recognition systems work  
- Implement embedding-based recognition (not classification)  
- Train a model using Triplet Loss  
- Capture live images using webcam (Google Colab)  
- Perform real-time identity verification  

---

## 🧠 Core Concept

Instead of saying:

> "This is person A"

The model learns:

> "These two faces are similar / different"

Each face is converted into a **128-dimensional vector (embedding)**.

Recognition = comparing distances between embeddings.

---

## 🏗 System Pipeline

### 1️⃣ Face Detection (MTCNN)

- Detect faces in input image  
- Crop and align face region  
- Resize to 160×160  
- Normalize pixel values  

---

### 2️⃣ Face Encoding (FaceNet)

- CNN-based encoder  
- Converts face → 128D embedding  
- Uses L2 normalization  

---

### 3️⃣ Training with Triplet Loss

Triplet consists of:

- Anchor (A) → same person  
- Positive (P) → same person  
- Negative (N) → different person  

Loss ensures:

```
Distance(A, P) < Distance(A, N)
```

This builds a meaningful embedding space.

---

### 4️⃣ Embedding Database

- Stores embeddings of known faces  
- Each person = 1 vector representation  

---

### 5️⃣ Recognition

- Capture new face from webcam  
- Generate embedding  
- Compare with stored embeddings using cosine distance  

Decision:

- If distance < threshold → Known  
- Else → Unknown  

---

## 🎥 Live Webcam Demo (Colab)

The system uses:

- JavaScript webcam bridge  
- Captures real-time images  
- Registers user face  
- Tests recognition  

---

## 📊 Output

- Face bounding box  
- Predicted identity  
- Distance score  
- Real-time recognition result  

---

## 🧠 Key Concepts Demonstrated

- Face detection (MTCNN)  
- Deep CNN embeddings  
- Metric learning  
- Triplet Loss  
- Cosine similarity  
- Real-time recognition pipeline  

---

## 🛠 Technologies Used

- Python 3  
- TensorFlow / Keras  
- OpenCV  
- MTCNN  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 📂 Project Structure

```
Face_Recognition_FaceNet/
│
├── facenet_triplet.ipynb
└── README.md
```

---

## ▶️ How to Run (Colab)

### 1️⃣ Install dependencies

```bash
pip install tensorflow opencv-python mtcnn numpy matplotlib scikit-learn
```

---

### 2️⃣ Run notebook

- Execute all cells  
- Enter your name  
- Capture face  
- Test recognition  

---

## ⚙️ Key Parameters

| Parameter | Description |
|----------|------------|
| `embedding_size` | Size of face vector (128) |
| `margin` | Triplet loss margin |
| `threshold` | Recognition threshold (e.g., 0.6) |

---

## 📈 Expected Output

- Registered face stored as embedding  
- Recognition result with label  
- Distance score displayed  
- Visual bounding box  

---

## ⚖ Classification vs FaceNet Approach

| Classification | FaceNet |
|---------------|--------|
| Fixed classes | Scalable |
| Retrain for new person | Just add embedding |
| Softmax output | Distance-based |
| Less flexible | More flexible |

This project uses **FaceNet-style embedding learning**.

---

## 🚀 Possible Improvements

- Use real dataset (LFW, VGGFace2)  
- Improve FaceNet architecture  
- Add multiple samples per identity  
- Deploy as real-time webcam app  
- Replace MTCNN with RetinaFace  

---

## 🎓 Learning Outcomes

By completing this project, you understand:

- How FaceNet works  
- Why Triplet Loss is used  
- Embedding-based recognition systems  
- Similarity-based classification  
- Real-world biometric systems  

---

## 👤 Author

**Shyam**  
AI & Data Science  

---

## 📌 Final Note

This project demonstrates how modern facial recognition systems work internally.

It moves beyond classification into:

- Metric learning  
- Embedding space modeling  
- Identity verification  

A key concept used in real-world systems like:

- Face unlock  
- Surveillance systems  
- Biometric authentication  