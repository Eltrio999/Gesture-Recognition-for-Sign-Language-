# ✋ Gesture Recognition for Sign Language

This project focuses on developing a hand gesture recognition system for sign language using a hybrid model that combines Convolutional Neural Networks (CNNs) for feature extraction and Support Vector Machines (SVMs) for classification.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)

---

## 🧠 Overview

The system aims to bridge the communication gap for the hearing impaired by recognizing sign language digits. It processes hand gesture images and identifies the corresponding sign language symbols using a machine learning pipeline.

---

## 🚀 Features

- 📸 Hand gesture recognition using image data
- 🧠 CNN-based feature extraction (ResNet-18)
- 🎯 SVM classifier for high-accuracy recognition
- 🔄 Data augmentation for robust performance
- 🛠️ Implemented in Python (Google Colab + PyTorch)

---

## 🛠️ Tech Stack

- Python
- PyTorch
- scikit-learn
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Dataset

We use the American Sign Language (ASL) Digits Dataset from Kaggle:

- 10 classes (0-9)
- RGB images of hands showing digits
- Preprocessed to 224x224 resolution
- Augmented using flipping, rotation, and scaling

🔗 Dataset: [ASL Digits Dataset on Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)

---

## ⚙️ Methodology

1. 📦 Preprocessing
   - Resize images to 224x224
   - Normalize pixel values
   - Apply data augmentation

2. 🧠 Feature Extraction
   - Use pre-trained ResNet-18 model
   - Remove final FC layer to extract features

3. 🧪 Classification
   - Train a Support Vector Machine (SVM) with a linear kernel
   - Input: CNN-extracted feature vectors

4. 📊 Evaluation
   - Accuracy
   - Precision / Recall / F1-Score

---

## 📈 Results

| Metric     | Value      |
|------------|------------|
| Accuracy   | ~95%       |
| Precision  | High       |
| Recall     | High       |
| F1 Score   | High       |

The hybrid CNN + SVM model showed strong performance under different lighting and gesture angles compared to traditional methods.

---

## 🧪 Installation

To run the project in a local or cloud-based Python environment:

```bash
# Clone the repository
git clone https://github.com/your-username/sign-language-gesture-recognition.git
cd sign-language-gesture-recognition

# Install dependencies
pip install -r requirements.txt
