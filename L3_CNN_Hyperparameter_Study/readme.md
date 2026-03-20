<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" />
<img src="https://img.shields.io/badge/Model-CNN-blueviolet.svg" />
<img src="https://img.shields.io/badge/Focus-Hyperparameter%20Study-yellow.svg" />
<img src="https://img.shields.io/badge/Dataset-MNIST-lightgrey.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />

</p>

# CNN Hyperparameter Study — Padding, Stride & Pooling

## 📌 Overview

This project performs a **controlled experimental study** on how key CNN hyperparameters affect model performance.

The study systematically evaluates the impact of:

- Padding (`valid` vs `same`)
- Stride (`1` vs `2`)
- Pooling (with vs without MaxPooling)

All configurations are tested on the **MNIST dataset** using a consistent CNN architecture to ensure fair comparison.

---

## 🎯 Objective

The goal of this project is to:

- Understand how padding affects spatial dimensions and feature retention  
- Analyze how stride impacts resolution and computation  
- Compare pooling vs no-pooling scenarios  
- Measure trade-offs between accuracy, speed, and model size  
- Visualize how feature maps change across configurations  

This project focuses on **learning behavior of CNNs**, not just achieving accuracy.

---

## 🏗 Experimental Design

A total of **6 configurations** are tested:

| # | Configuration | Padding | Stride | Pooling |
|---|--------------|---------|--------|---------|
| 1 | Valid \| S1 \| Pool   | valid | 1 | Yes |
| 2 | Same  \| S1 \| Pool   | same  | 1 | Yes |
| 3 | Valid \| S2 \| Pool   | valid | 2 | Yes |
| 4 | Same  \| S2 \| Pool   | same  | 2 | Yes |
| 5 | Same  \| S1 \| NoPool | same  | 1 | No  |
| 6 | Same  \| S2 \| NoPool | same  | 2 | No  |

Configs 5 and 6 isolate the effect of **stride without pooling**.

---

## 🧠 Model Architecture

Each configuration uses the same base architecture:

```
Input (28×28×1)
    └── Conv2D (32 filters, 3×3, ReLU)
           └── [MaxPooling2D (optional)]
                  └── Flatten
                         └── Dense (10, Softmax)
```

Only padding, stride, and pooling vary.

---

## 📊 Metrics Collected

Each configuration is evaluated using:

- Test Accuracy  
- Validation Accuracy  
- Training Time per Epoch  
- Number of Parameters  
- Feature Map Size  

This allows comparison across:

- Performance
- Efficiency
- Model complexity

---

## 📈 Visualizations

The project generates:

- 📊 Test Accuracy vs Configuration  
- ⚡ Training Speed vs Configuration  
- 📦 Parameter Count vs Configuration  
- 🧠 Feature Map Visualizations  

Feature maps show how spatial information changes with stride and padding.

---

## 🔍 Key Observations

- **Padding = "same"** preserves spatial information better  
- **Stride = 2** reduces computation but may lose detail  
- **Pooling + Stride** can aggressively shrink feature maps  
- Removing pooling isolates stride impact clearly  
- Trade-off exists between **accuracy vs speed vs resolution**

---

## 🛠 Technologies Used

- Python 3  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Pandas  

---

## 📂 Project Structure

```
CNN-Hyperparameter-Study/
│
├── L3-1(Strides-padding->CNN).py
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install tensorflow numpy matplotlib pandas
```

### 2️⃣ Run script

```bash
python L3-1(Strides-padding->CNN).py
```

---

## 📊 Expected Output

- Comparison table of all configurations  
- Bar charts for accuracy, speed, and parameters  
- Visual feature maps for each configuration  
- Clear understanding of hyperparameter effects  

---

## 🧠 Key Concepts Demonstrated

- Convolutional spatial transformations  
- Padding strategies (`valid` vs `same`)  
- Stride-based downsampling  
- Pooling effects on feature maps  
- Trade-off analysis in CNN design  
- Experimental benchmarking  

---

## 🚀 Possible Improvements

- Add deeper CNN architectures  
- Include CIFAR-10 dataset for complexity  
- Compare with Batch Normalization  
- Add dropout for regularization  
- Automate hyperparameter search  

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- How CNN hyperparameters affect learning  
- Spatial dimension control in convolutions  
- Trade-offs in model design  
- Feature map interpretation  
- Experimental evaluation of architectures  

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India 

---

## 📌 Final Note

This project focuses on **understanding CNN behavior**, not just performance.

It highlights how small design choices like padding and stride can significantly impact:

- Model accuracy  
- Computational efficiency  
- Feature representation  

A critical step toward mastering deep learning architecture design.