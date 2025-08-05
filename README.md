# Semantic Segmentation of Necrosis vs Non-Necrosis in H&E Medical Images

This repository provides an end-to-end deep learning pipeline for semantic segmentation of **necrosis vs non-necrosis regions** in **Hematoxylin and Eosin (H&E)** stained histopathological images using **DeepLabV3 with ResNet50 backbone**. 
The model supports **image augmentation**, **automatic hyperparameter tuning**, and **multi-GPU training**.
---

## 🧠 Problem Statement

Accurate detection of necrotic regions in histopathological slides is essential for disease grading and treatment decisions. Manual annotation is time-consuming and error-prone. This project uses semantic segmentation with deep learning to automate the classification of necrotic vs non-necrotic tissue areas.

---

Class 0: Non-necrotic tissue

Class 1: Necrotic tissue

🛠 Features
✅ DeepLabV3 with ResNet-50 backbone (PyTorch)

✅ Binary class mask generation

✅ Data augmentation using torchvision.transforms

✅ Automatic hyperparameter tuning (e.g., via Optuna)

✅ Multi-GPU support (via torch.nn.DataParallel or torch.distributed)

✅ Evaluation metrics: R² score and global pixel accuracy

## 📂 Dataset

The dataset used in this project was obtained from **The Cancer Genome Atlas (TCGA)**. The **Hematoxylin and Eosin (H&E) stained whole-slide images** were annotated and labeled by researchers from the **Stony Brook University Biomedical Informatics Department**.

### 🏷️ Labels

The masks and region annotations used in training and evaluation were manually curated and verified by the expert team at Stony Brook.

### Related Publication

Le Hou et al. (2020) provide the full dataset and detailed annotations in their paper:
Dataset of segmented nuclei in hematoxylin and eosin stained histopathology images of ten cancer types, Scientific Data, 7:185.
https://www.nature.com/articles/s41597-020-0528-1

Note: This repository only utilizes a subset of the full dataset. 

### 📄 License & Usage

Please ensure you follow the data usage guidelines as defined by TCGA and the labeling institution. 


