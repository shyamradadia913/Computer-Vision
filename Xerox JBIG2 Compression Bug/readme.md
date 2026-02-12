<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/Topic-JBIG2%20Compression-orange.svg" />
<img src="https://img.shields.io/badge/Domain-Computer%20Vision-blueviolet.svg" />
<img src="https://img.shields.io/badge/Focus-Semantic%20Integrity-yellow.svg" />
<img src="https://img.shields.io/badge/Type-Case%20Study-lightgrey.svg" />
<img src="https://img.shields.io/badge/Risk-Silent%20Corruption-red.svg" />

</p>

# Xerox JBIG2 Compression Bug  
## When Compression Breaks Meaning

---

## 📌 Overview

This project analyzes one of the most critical failures in document processing history:

**The Xerox JBIG2 Compression Bug**

In the early 2010s, users discovered that certain Xerox multifunction printers silently altered scanned documents.  
Digits such as “6” were replaced with visually similar digits like “8.”

The document appeared visually correct —  
but the meaning was corrupted.

This repository contains:

- A detailed technical case study report
- Experimental simulations (Tasks 1–5)
- Structural analysis of compression failure
- Implications for AI and computer vision systems

---

## 🚨 Why This Matters

Compression is normally assumed to degrade *quality*.

This case proves something far more dangerous:

> Compression can alter semantics while preserving visual similarity.

That is catastrophic for:

- Legal documents
- Financial records
- Identity verification systems
- OCR pipelines
- AI training datasets

---

## 📂 Project Files

- `Xerox JBIG2 Compression Bug.ipynb`
- `Case Study Report.pdf`
- Sample test images (including driver’s license)

---

## 🧠 Background: JBIG2 Compression

JBIG2 is a lossy compression algorithm designed for black-and-white scanned documents.

### How JBIG2 Works

1. Detect similar glyphs (characters)
2. Store one prototype
3. Replace all similar glyphs with that prototype

This achieves extremely high compression ratios.

---

## ❌ Where It Failed

Xerox’s implementation incorrectly grouped:

- “6” with “8”
- Similar numeric glyphs
- Visually similar but semantically different characters

During decompression, incorrect glyph substitution occurred.

The result:

Visually acceptable document  
Semantically corrupted data  

---

## 🧪 Experimental Implementation (Tasks 1–5)

The notebook reproduces and analyzes this failure through five structured tasks.

---

### 🔹 Task 1: Pattern Substitution Risk

Simulated JBIG2-style grouping:

- Extract connected components
- Compute shape similarity
- Group by threshold
- Replace with prototype glyph

**Observation:**
- Low similarity threshold → safe grouping
- Higher threshold → digit merging
- Corruption increases gradually

Critical insight:
Compression errors escalate silently.

---

### 🔹 Task 2: Human-Visible vs Machine-Relevant Differences

Process:

- Compress image at multiple JPEG qualities
- Compute PSNR and SSIM
- Apply edge detection

Findings:

- PSNR decreases gradually
- SSIM remains high
- Edge structure degrades rapidly

Conclusion:

Perceptual similarity does NOT guarantee machine reliability.

---

### 🔹 Task 3: Silent Data Corruption Detection

Approach:

- Compare lossless vs lossy scans
- Extract contours
- Compute structural differences

Key finding:

Images can look identical to humans  
but contain measurable structural inconsistencies.

Detection must be algorithmic, not visual.

---

### 🔹 Task 4: Compression Breaking Recognition

Test:

- Rule-based digit recognizer
- Evaluate original vs compressed images

Result:

- Accuracy drops significantly under heavy compression
- Similar-shaped digits fail first (6/8, 0/9, 1/7)
- Compression introduces structured bias

If AI models are trained on corrupted data:
They learn the wrong mapping as ground truth.

---

### 🔹 Task 5: Designing Safe Compression Rules

Heuristic based on:

- Edge density
- Connected component count
- Entropy

Decision logic:

| Image Type | Recommended Compression |
|------------|------------------------|
| Dense text | Lossless |
| Forms | Controlled lossy |
| Photos | Lossy |
| Legal documents | No lossy compression |

---

## 🆔 Real-World Risk Example

The included driver's license sample demonstrates a high-stakes scenario.

If compression silently alters:

- Date of birth
- License number
- Expiry date
- Address

The result is legal or financial disaster.

Semantic integrity must be preserved in identity documents.

---

## ⚖ Human vs Machine Perception

| Human Perception | Machine Interpretation |
|------------------|-----------------------|
| Tolerates small visual changes | Requires exact pixel-level precision |
| Auto-corrects context mentally | Relies on glyph accuracy |
| Focuses on readability | Requires structural integrity |

Mismatch between these systems caused the failure.

---

## 🔬 Risk to AI Systems

If OCR or vision models are trained on corrupted scans:

Expected failures:

- Systematic digit confusion
- Pattern bias
- Reduced generalization
- Model overfitting to corrupted glyph prototypes

This is catastrophic in:

- Banking
- Legal systems
- Healthcare records
- Government identity systems

---

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Jupyter Notebook

---

## ▶️ How to Run

1️⃣ Install dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-image
```

2️⃣ Launch notebook:

```bash
jupyter notebook
```

3️⃣ Run all tasks sequentially.

---

## 📌 Key Lessons

1. Lossy compression is not harmless.
2. Visual similarity ≠ semantic equivalence.
3. Perceptual metrics (PSNR/SSIM) are insufficient.
4. High-stakes documents require structural validation.
5. Silent corruption is more dangerous than visible artifacts.

---

## 🧠 Core Insight

Optimization for human perception can destroy machine-relevant information.

Compression is not just about storage efficiency.

It is about trust.

---

## 👤 Author

**Shyam**  
AI & Data Science  

---

## 📌 Final Thought

The Xerox JBIG2 incident is not just a printer bug.

It is a foundational lesson in AI system design:

> Silent corruption is worse than visible failure.
