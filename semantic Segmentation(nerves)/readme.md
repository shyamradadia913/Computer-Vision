<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" />
<img src="https://img.shields.io/badge/Model-U--Net-blueviolet.svg" />
<img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-yellow.svg" />
<img src="https://img.shields.io/badge/Domain-Medical%20Imaging-lightgrey.svg" />
<img src="https://img.shields.io/badge/Output-Tumor%20Mask-success.svg" />

</p>

# Medical Image Semantic Segmentation using U-Net

## 📌 Overview

This project implements a **U-Net based semantic segmentation model** for tumor detection in medical images.

The pipeline:

- Loads TIFF medical images
- Matches images with corresponding mask files
- Preprocesses and normalizes data
- Trains a full U-Net architecture
- Evaluates segmentation performance
- Generates tumor masks
- Saves the trained model

This implementation focuses on building a complete end-to-end segmentation workflow using TensorFlow.

---

## 🎯 Objective

The goal of this project is to:

- Segment tumor regions from medical images
- Train a U-Net architecture from scratch
- Handle TIFF images with LZW compression
- Automatically pair images with matching mask files
- Evaluate segmentation using Dice Score
- Save a deployable segmentation model

---

## 🏗 Model Architecture: U-Net

The project implements a full U-Net architecture consisting of:

### 🔹 Encoder
- Convolution blocks
- Batch Normalization
- MaxPooling layers
- Progressive feature depth increase (64 → 1024 filters)

### 🔹 Bottleneck
- Deep feature extraction
- High-level representation learning

### 🔹 Decoder
- Transposed Convolutions (Upsampling)
- Skip Connections
- Feature concatenation
- Progressive resolution recovery

### 🔹 Output Layer
- 1-channel sigmoid activation
- Binary mask prediction

U-Net is specifically designed for pixel-level segmentation tasks, especially in medical imaging.

---

## 📂 Dataset Handling

The data loader:

- Scans `.tif` and `.tiff` files
- Only loads images that have matching `_mask.tif` files
- Automatically handles LZW compression
- Converts images to grayscale
- Resizes to 128×128
- Normalizes pixel values to [0, 1]
- Applies thresholding for tumor mask generation

This ensures clean and valid image-mask pairs for training.

---

## 🧠 Training Pipeline

### Train/Validation Split
- 80% Training
- 20% Validation

### Loss Function
- Binary Crossentropy

### Metrics
- Accuracy
- Precision
- Recall
- Dice Score (custom calculation)

### Callbacks Used
- ModelCheckpoint (save best model)
- ReduceLROnPlateau (adaptive learning rate)
- EarlyStopping (prevent overfitting)

Training runs for up to 50 epochs with batch size 8.

---

## 📊 Evaluation

Model performance is evaluated using:

- Validation loss
- Precision
- Recall
- Dice Score

Dice Score formula:

```
Dice = 2 * (Prediction ∩ GroundTruth) / (Prediction + GroundTruth)
```

Dice Score is critical in medical segmentation tasks.

---

## 🖼 Visualization

After training:

- Input images are displayed
- Ground truth masks are shown
- Predicted masks are overlayed
- Comparison between actual and predicted tumor regions is visualized

---

## 🛠 Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- PIL
- tifffile
- Matplotlib
- imagecodecs (for LZW TIFF support)

---

## 📂 Project Structure

```
Tumor-Segmentation-U-Net/
│
├── sementic_segmentation.py
├── data/
│   ├── image1.tif
│   ├── image1_mask.tif
│   └── ...
├── best_unet.h5
├── tumor_segmenter.h5
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install tensorflow numpy pillow tifffile matplotlib imagecodecs
```

### 2️⃣ Set dataset folder path

Update:

```python
DATA_FOLDER = "/content/data"
```

### 3️⃣ Run training

Execute the script or notebook:

```bash
python sementic_segmentation.py
```

---

## 📦 Model Output

- `best_unet.h5` → Best validation model
- `tumor_segmenter.h5` → Final trained tumor segmentation model

Model size: ~118MB

---

## 🔬 Why U-Net?

U-Net is widely used in:

- Brain tumor segmentation
- Organ segmentation
- Cell detection
- Biomedical image analysis

Advantages:

- Works well with small datasets
- Preserves spatial resolution
- Strong performance in medical imaging

---

## 🚀 Possible Improvements

- Replace BCE with Dice Loss
- Add Data Augmentation
- Use larger input resolution
- Apply focal loss for class imbalance
- Add IoU metric
- Deploy via web interface
- Convert to TensorRT for faster inference

---

## 🎓 Learning Outcomes

By completing this project, you gain understanding of:

- Semantic segmentation fundamentals
- U-Net architecture design
- Medical image preprocessing
- TIFF handling and LZW compression
- Model training and validation strategies
- Dice score evaluation
- Deep learning deployment workflow

---

## 👤 Author

**Shyam A. Radadia**  
🎓 AI & Data Science - ADANI INSTITUTE OF DIGITAL TECHNOLOGY MANAGEMENT
📍 Gandhinagar, India
📧 shyamradadia99@gmail.com

---

## 📌 Final Note

This project demonstrates a complete medical image segmentation pipeline using U-Net.

It transitions from classical computer vision projects to advanced deep learning-based pixel-wise prediction systems.

This is a production-level segmentation architecture adapted for custom medical datasets.
