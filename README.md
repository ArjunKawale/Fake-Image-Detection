

# 🕵️‍♂️ Fake Image Detection

This project implements a **deep learning model** for detecting whether an image of a face is **real or fake (deepfake/manipulated)**.
We benchmark and compare different CNN and modern architectures such as **VGG16, Xception, EfficientNetB0, and more**.

---

## 🎯 Objective

The goal of this project is to:

* Compare the performance of multiple deep learning models (**VGG16, Xception, EfficientNetB0, ResNet, MobileNet, etc.**) for **real vs. fake face classification**.
* Evaluate models on standard metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
* Provide a reproducible workflow for training, evaluation, and extension with additional models.

---

## 🧠 Models & Results

We trained and compared multiple architectures on the **RVF10K dataset** (linked below).

### Dataset

* [RVF10K Dataset (Kaggle)](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k)
* **Train set:** 7,000 images (3,500 real, 3,500 fake)
* **Validation set:** 3,000 images (1,500 real, 1,500 fake)
* **Sources:**

  * Real: [NVIDIA Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset)
  * Fake: [StyleGAN-based dataset by Bojan Tunguz](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces)

---

### Benchmark Results

| Model              | Accuracy | Precision | Recall | F1-score | Notes                                                                                                  |
| ------------------ | -------- | --------- | ------ | -------- | ------------------------------------------------------------------------------------------------------ |
| **VGG16**          | 0.8647   | 0.8377    | 0.9047 | 0.8699   | Started **overfitting** early → training stopped at epoch **7** (all layers frozen except classifier). |
| **EfficientNetB0** | 0.9887   | 0.9913    | 0.9860 | 0.9886   | All layers unfrozen, trained for **17 epochs**.                                                        |
| **Xception**       | 0.9787   | 0.9699    | 0.9880 | 0.9789   | All layers unfrozen, trained for **14 epochs**.                                                        |

**Confusion Matrices:**

* **VGG16**

  ```
  [[1237  263]
   [ 143 1357]]
  ```
* **EfficientNetB0**

  ```
  [[1487   13]
   [  21 1479]]
  ```
* **Xception**

  ```
  [[1454   46]
   [  18 1482]]
  ```

---

## 🛠 Tech Stack

* **Python**
* **PyTorch** (Deep Learning Framework)
* **scikit-learn** (Evaluation metrics, preprocessing)
* **OpenCV** (Image preprocessing)
* **Matplotlib/Seaborn** (Visualization)

---

