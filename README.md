# 🌸 CerviScan — Early Cervical Pre-Cancer Detection Using Color Moments and Texture Features

Welcome to **CerviScan**, an undergraduate thesis project focused on improving the early detection of cervical pre-cancer by analyzing colposcopy images treated with acetic acid (VIA/IVA).
This project evaluates the effectiveness of **color moment features** across different color spaces and their combination with **texture features** to enhance classification performance using traditional machine learning models.

---

## 📌 Overview

CerviScan explores how **color space selection** and **feature fusion** impact the accuracy of cervical pre-cancer detection. While many previous studies rely heavily on RGB-based color features or texture-only descriptors, this research investigates a more comprehensive representation using:

* **Color moments** (mean, standard deviation, skewness)
* **Three different color spaces:** **RGB**, **YUV**, and **LAB**
* **Texture descriptors:** **LBP**, **GLRLM**, and **Tamura**

The goal is to determine which feature combinations yield the most robust classification performance on post-VIA colposcopy images.

---

## 🎯 Dataset

The dataset consists of **162 post-VIA colposcopy images** obtained from the **IARC Colposcopy Image Bank**, including:

* **75 abnormal** images
* **87 normal** images

The dataset is **not distributed** in this repository due to licensing restrictions. Users must request access directly from IARC.

---

## ⚙️ Methodology

The CerviScan pipeline is divided into several stages:

### **1. Preprocessing**

* **Cropping** to isolate the cervical region (Region of Interest / ROI)
* **Grayscale conversion** for texture-based processing

### **2. Image Segmentation**

Segmentation is performed using **Multi-Otsu thresholding**, allowing the separation of the cervix from background artifacts based on multi-level intensity thresholds.

### **3. Feature Extraction**

#### **Color Moment Features**

Extracted from 3 color spaces:

* **RGB**
* **YUV**
* **LAB**

Each color channel is described using:

* Mean
* Standard deviation
* Skewness

#### **Texture Features**

Extracted from segmented grayscale ROIs:

* **LBP** (Local Binary Pattern)
* **GLRLM** (Gray Level Run Length Matrix) – extracted across four directions (0°, 45°, 90°, 135°)
* **Tamura** features: coarseness, contrast, directionality, roughness

### **4. Classification Models**

Two traditional machine learning classifiers are used:

* **XGBoost**
* **AdaBoost**

Feature selection and ranking were analyzed using **RFECV** (Recursive Feature Elimination with Cross-Validation).

### **5. Evaluation Metrics**

Model performance is evaluated using:

* Accuracy
* Precision
* Specificity
* Recall
* F1-score

---

## 📈 Key Findings

### 🔍 **1. Most Effective Color Space**

The **YUV color space** delivered the most consistent and discriminative color information, outperforming RGB and LAB—particularly because YUV cleanly separates luminance from chrominance, improving lesion visibility.

### 🔍 **2. Importance of Texture Features**

Texture descriptors (LBP, GLRLM, Tamura) played a **dominant role** in detecting lesion patterns, especially features such as:

* **std_LBP**
* **Tamura contrast**
* Specific GLRLM metrics (e.g., SRLGLE, LRHGLE)

### 🔍 **3. Best Overall Performance**

The **best classification result** was achieved using:

### **XGBoost + YUV Color Moments + Texture Features**

**Performance:**

<img width="800" height="550" alt="rfecv_visualization" src="https://github.com/user-attachments/assets/89ef93ff-4eaa-4386-a2d9-9b5b7d210260" />
<img width="800" height="550" alt="confusion_matrix_and_performance_metrics" src="https://github.com/user-attachments/assets/12691087-0c07-4c8e-83b8-b4c61f8140e2" />

### 🔍 **4. Feature Fusion Improves Accuracy**

Combining **color** and **texture** features consistently outperformed using either feature group alone.

---

## 🧬 Objectives

This project aims to:

* Evaluate how different **color spaces** affect the performance of color moment features
* Examine the contribution of **texture features** toward cervical lesion classification
* Demonstrate that **feature fusion** (color + texture) yields significantly stronger classification performance
* Provide an interpretable and reproducible pipeline for medical image-based pre-cancer detection

---

## 🛠️ System Pipeline

```
Raw Post-VIA Images
        ↓
Cropping & Grayscaling
        ↓
Multi-Otsu Segmentation
        ↓
Color & Texture Feature Extraction
        ↓
Feature Selection (RFECV)
        ↓
Classification (XGBoost / AdaBoost)
        ↓
Evaluation & Feature Ranking
```

---

## 📣 Notes

* This repository is part of an **undergraduate thesis project** at the
  *Department of Electrical Engineering, Universitas Jenderal Soedirman (UNSOED)*.
* The purpose of this project is **academic research**, not clinical usage.
* The colposcopy dataset from IARC cannot be redistributed here and must be obtained directly from the original source.

---

## 🙌 Acknowledgements

This project is supervised by:

* **Prof. Dr. Eng. Retno Supriyanti, S.T., M.T.**
* **Ir. Muhammad Syaiful Aliim, S.T., M.T.**

With research carried out at the **Biomedical Electronics Laboratory, UNSOED.**

---

Thank you for your interest in **CerviScan**!
For questions, discussions, or collaboration opportunities—feel free to reach out.
