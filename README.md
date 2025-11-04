# 🧠 Multimodal Tumor Classification using Deep Learning

This project focuses on building an **AI-powered multimodal tumor classification system** that integrates **radiology (medical imaging)** and **pathology (microscopic tissue images)** to improve cancer diagnosis accuracy.  
By combining information from both data modalities, the model leverages the strengths of image-based and microscopic analyses to make reliable predictions.

---

## 🚀 Project Overview

Traditional cancer diagnosis relies on either radiology or pathology separately — but combining them can significantly improve clinical insights.  
This project explores **fusion-based deep learning models** using CNN architectures for both modalities and a joint fusion network for final prediction.

---

## 🧩 Architecture

The overall workflow includes:

1. **Data Preprocessing**
   - Image normalization, resizing, and augmentation.
   - Dataset split into training, validation, and test sets.

2. **Radiology Model**
   - CNN-based feature extraction from medical scans.

3. **Pathology Model**
   - CNN-based feature extraction from histopathological images.

4. **Fusion Network**
   - Concatenates embeddings from both models.
   - Fully connected layers for joint classification.

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score.
   - Confusion matrix visualization.

---

## 🧠 Model Workflow

```mermaid
flowchart LR
A[Load Radiology + Pathology Datasets] --> B[Preprocessing & Augmentation]
B --> C[Train Radiology CNN]
B --> D[Train Pathology CNN]
C --> E[Feature Fusion Layer]
D --> E
E --> F[Classification Layer]
F --> G[Model Evaluation & Results]


