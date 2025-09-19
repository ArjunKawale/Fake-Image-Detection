# 🕵️‍♂️ Fake Image Detection (CNN – VGG16 & Beyond)

This project implements a **deep learning model** for detecting whether an image of a face is **real or fake (deepfake/manipulated)**. The current implementation uses a **VGG16-based Convolutional Neural Network (CNN)**, and we aim to expand the repository by **comparing VGG16 against other state-of-the-art models** such as **Xception, EfficientNetB0, and more**.

---

## 🎯 Objective

With the rise of **AI-generated fake content (deepfakes)**, there is an urgent need for reliable detection methods. This project aims to:

* Build a **baseline CNN classifier using VGG16**.
* Classify input images into **Real** vs. **Fake**.
* Compare VGG16’s performance against advanced models (**Xception, EfficientNetB0, etc.**) to benchmark results.
* Provide a reproducible workflow for training, evaluation, and future extensions.

---

## 🧠 Current Model

* **Architecture:** VGG16 (transfer learning)
* **Framework:** PyTorch
* **Task:** Binary image classification (Real vs. Fake faces)

---

## 📊 Planned Extensions

* Add benchmarks with **Xception, EfficientNetB0, ResNet, MobileNet**.
* Report metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
* Improve preprocessing with **face alignment and augmentation**.

---

## 🛠 Tech Stack

* **Python**
* **PyTorch** (Deep Learning Framework)
* **scikit-learn** (Evaluation metrics, preprocessing)
* **OpenCV** (Image preprocessing)
* **Matplotlib/Seaborn** (Visualization)


---

## 📈 Roadmap

* [x] Implement baseline VGG16 model
* [ ] Add comparisons with Xception, EfficientNetB0, ResNet, MobileNet

