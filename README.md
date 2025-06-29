# Cerviscan - Cervical Pre-Cancer Detection

Welcome to **Cerviscan**, an undergraduate thesis project focused on the detection of cervical pre-cancer lesions using colposcopy images treated with acetic acid.

## 📌 Overview

**Cerviscan** aims to support early detection and diagnosis of cervical pre-cancer by applying traditional machine learning techniques to medical images. The project uses **colposcopy images** that have been subjected to **Visual Inspection with Acetic Acid (VIA)** to highlight suspicious regions on the cervix.

## 🎯 Dataset

The dataset used in this project is sourced from the **International Agency for Research on Cancer (IARC)**. The images contain cervical tissue captured under colposcopy after applying acetic acid, which helps reveal areas with potential lesions.

---

## ⚙️ Methodology

The detection pipeline in **Cerviscan** includes:

- **Image Segmentation:**  
  Cervical region segmentation is performed using the **Multi-Otsu Thresholding** method to separate relevant tissue areas from the background.

- **Feature Extraction:**  
  From the segmented regions, the following features are extracted:
  - **Color Features:**  
    - YUV Color Moments  
    - RGB Color Moments  
    - LAB Color Moments
  - **Texture Features:**  
    - Gray Level Run Length Matrix (**GLRLM**)  
    - Gray Level Co-occurrence Matrix (**GLCM**)  
    - Local Binary Pattern (**LBP**)  
    - **Tamura** texture features (coarseness, contrast, directionality, roughness)

- **Classification Models:**  
  For classification, traditional machine learning algorithms are used:
  - **XGBoost**
  - **AdaBoost**

---

## 📈 Objectives

By combining robust feature extraction and well-established machine learning classifiers, **Cerviscan** aims to:
- Assist healthcare professionals in early detection of cervical pre-cancerous lesions
- Provide an interpretable pipeline for research and academic purposes
- Contribute to the development of computer-aided diagnosis (CAD) tools for cervical cancer prevention

---

## 📣 Notes

This repository is part of an **undergraduate thesis project** and is intended for **academic and research purposes only**. The dataset from IARC is not redistributed here and must be obtained directly from the source according to its usage license.

---

**Thank you for your interest in Cerviscan!** 🧬✨  
For any questions or collaboration opportunities, feel free to reach out.
