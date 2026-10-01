# 🕵️‍♂️ Real/Fake Sentinel: CNN-Based Detection of Fake / Generated Faces

This repository contains the implementation, benchmark evaluation notebooks, and full-stack deployment for a **deep learning–based fake face image detection system**, designed to classify whether a face image is **authentic (real)** or **manipulated (fake / deepfake)**.

Multiple deep transfer learning architectures (**EfficientNet-B0, XceptionNet, and VGG16**) are evaluated and compared on facial authenticity classification, and the optimal model is deployed via a decoupled **FastAPI REST backend** paired with an interactive **cyber-forensic web frontend**.

---

## 🎯 Objectives & Key Features

- **Empirical CNN Benchmark:** Comparative analysis of **VGG16**, **Xception**, and **EfficientNet-B0** architectures to evaluate classification accuracy, latency, and parameter footprint.
- **High Detection Performance:** Achieves **98.87% test accuracy** using fine-tuned EfficientNet-B0 transfer learning.
- **Full-Stack Deployment:** Asynchronous REST API serving PyTorch inference via FastAPI connected to an interactive, responsive web interface.
- **Real-Time Telemetry:** Live inference latency calculation, dynamic scanning indicators, and softmax probability distributions (`P(Real)` vs. `P(Fake)`).
- **Edge-Ready Footprint:** EfficientNet-B0 deployment weights occupy only **~18 MB**, enabling sub-100 ms inference times on standard CPU/GPU systems.

---

## 🧪 Evaluation Notebooks

The `notebooks/` directory contains evaluation workflows and metrics across the benchmarked architectures.

| Notebook | Description |
| :--- | :--- |
| `FakeVsReal_EfficientNetB0_Model_Evaluation.ipynb` | Evaluates the EfficientNet-B0 model with complete performance metrics and confusion matrices. |
| `FakeVsReal_VGG16_Model_Evaluation.ipynb` | Evaluation notebook for transfer learning using VGG16. |
| `FakeVsReal_Xception_Model_Evaluation.ipynb` | Evaluation notebook using the Xception architecture. |

---

## 🗂 Dataset – RVF10K Real vs Fake Face Dataset

The benchmark models were evaluated using the publicly available **RVF10K dataset**, consisting of high-quality real portraits and GAN-synthesized faces.

- **Total Images:** 10,000
- **Training Set:** 7,000 images (3,500 real + 3,500 fake)
- **Validation Set:** 3,000 images (1,500 real + 1,500 fake)
- **Real Data Source:** NVIDIA **FFHQ Dataset** (Flickr-Faces-HQ)
- **Fake Data Source:** **StyleGAN-generated synthetic faces** (sampled from Bojan Tunguz 1 Million Faces)
- **Dataset Link:** [Kaggle - RVF10K Real vs Fake Face Dataset](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k)

---

## 📊 Benchmark Model Results

| Architecture | Accuracy | Precision | Recall | F1-Score | Weight Size | Training Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **VGG16** | 86.47% | 83.77% | 90.47% | 0.8699 | 153 MB | Early overfitting, training stopped at epoch 7 |
| **Xception** | 97.87% | 96.99% | 98.80% | 0.9789 | 85 MB | Strong results with 14-epoch training |
| **EfficientNet-B0 (Best)** | **98.87%** | **99.13%** | **98.60%** | **0.9886** | **18 MB** | **Best performing model**, converged at epoch 17 |

### Benchmark Confusion Matrices

* **VGG16**

  ```
  [[1237  263]
  [ 143 1357]]
  ```
1,237 True Negatives, 263 False Positives | 143 False Negatives, 1,357 True Positives

* **EfficientNetB0 (Selected for Deployment)**

  ```
  [[1487   13]
  [  21 1479]]
  ```
1,487 True Negatives, 13 False Positives | 21 False Negatives, 1,479 True Positives

* **Xception**

  ```
  [[1454   46]
  [  18 1482]]
  ```
1,454 True Negatives, 46 False Positives | 18 False Negatives, 1,482 True Positives

---

## 🏗️ System Workflow
```
[ User Browser / Client ]
            │
            │  multipart/form-data (Target Face Image)
            ▼
[ FastAPI Server (:8000) ]
            │
            │  Preprocessing Pipeline:
            │  - Resize to (224, 224)
            │  - ToTensor()
            │  - Normalize(mean=[0.485, 0.456, 0.406],
            │             std=[0.229, 0.224, 0.225])
            ▼
[ EfficientNet-B0 Engine (PyTorch) ]
            │
            │  Softmax Probability Extraction
            ▼
[ JSON Telemetry Response ]
            │
            │  {
            │    "prediction": "Real" | "Fake",
            │    "confidence": 99.8
            │  }
            ▼
[ Cyber-Forensic Web UI (:3000) ]
```

---

## ⚙️ Tech Stack

- **Deep Learning Framework**: PyTorch, Torchvision
- **API & Backend**: FastAPI, Uvicorn, Python-Multipart
- **Image Processing & Mathematics**: Pillow (PIL), NumPy, OpenCV
- **Evaluation & Metrics**: scikit-learn, SciPy, Matplotlib, Seaborn
- **Frontend Interface**: HTML5, Modern JavaScript (Fetch API), Tailwind CSS

---

## 📦 Folder Structure
```
Fake-Image-Detection/
│
├── backend/
│   ├── best_efficientnet.pth          # Active PyTorch model weights (18 MB)
│   ├── main.py                        # FastAPI application & inference endpoint
│   └── requirements.txt               # Backend dependencies
│
├── frontend/
│   ├── index.html                     # UI dashboard (Tailwind CSS)
│   └── script.js                      # Async API fetch & telemetry logic
│
├── notebooks/
│   ├── FakeVsReal_EfficientNetB0_Model_Evaluation.ipynb
│   ├── FakeVsReal_VGG16_Model_Evaluation.ipynb
│   └── FakeVsReal_Xception_Model_Evaluation.ipynb
│
├── .gitignore                         # Excludes large binaries & cache
└── README.md                          # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Backend Service Setup

- Navigate to backend directory
```
cd backend
```

- Install dependencies
```
pip install -r requirements.txt
```

- Start FastAPI server
```
python -m uvicorn main:app --reload --port 8000
```

- Backend API: http://127.0.0.1:8000
- Interactive API Documentation (Swagger): http://127.0.0.1:8000/docs

### 2. Frontend Interface Setup

- Open a second terminal window:

- Navigate to frontend directory
```
cd frontend
```

- Launch lightweight local server
```
python -m http.server 3000
```

- Open your browser and navigate to: http://localhost:3000

The frontend can then be used to upload face images and run real-time fake/real classification through the FastAPI backend.

---

## 📌 Project Status

This project is developed for academic and portfolio demonstration purposes.
The repository provides evaluation notebooks and a full-stack deployment pipeline using the highest-performing transfer learning model, best_efficientnet.pth.

---

## 🔒 License & Usage

This project may be used for research reference, learning, or academic demonstration.
Feel free to fork, clone, and extend the project for educational purposes.
