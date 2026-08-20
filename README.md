# 🎓 Student Placement Prediction — MLOps Pipeline

> An end-to-end MLOps project that automates the full machine learning lifecycle for predicting student placement outcomes — from data versioning and model training to containerized deployment and experiment tracking.

---

## 📌 Project Overview

This project implements a production-grade **MLOps pipeline** for predicting whether a student will be placed based on academic and skill-based features. It covers the complete ML lifecycle using industry-standard tools including MLflow, DVC, FastAPI, Docker, GitHub Actions, Jenkins, and cloud platforms (Azure ML & GCP Vertex AI).

The project is structured around **8 progressive experiments**, each targeting a key area of MLOps: reproducibility, automation, deployment, tracking, and versioning.

---

## ❓ Problem Statement

Campus placement is a critical outcome for students and institutions alike. Manually assessing placement likelihood is time-consuming and inconsistent. This project builds a **machine learning model** that predicts whether a student will be placed, and wraps it in a **fully automated, versioned, and deployable MLOps pipeline** — making the model reproducible, monitorable, and production-ready.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Simulated campus placement records |
| **Size** | 10,000 student records |
| **Target** | `PlacementStatus` (Placed / Not Placed) |

### Features

| Feature | Type | Description |
|---|---|---|
| `CGPA` | Float | Cumulative Grade Point Average |
| `Internships` | Integer | Number of internships completed |
| `Projects` | Integer | Number of projects done |
| `AptitudeTestScore` | Integer | Score in aptitude assessment |
| `SoftSkillsRating` | Float | Rating of communication & soft skills |
| `SSC_Marks` | Integer | 10th standard marks |
| `HSC_Marks` | Integer | 12th standard marks |

---

## 🛠️ Technologies

### Machine Learning
- `scikit-learn` — Random Forest Classifier (primary model)
- `pandas`, `numpy` — Data processing
- `joblib` — Model serialization

### MLOps & Tracking
- `MLflow` — Experiment tracking, model logging, model registry
- `DVC` — Data and model versioning with cloud remote

### API & Deployment
- `FastAPI` + `Uvicorn` — REST inference API
- `Docker` — Containerization
- `Kubernetes (AKS / GKE)` — Scalable cloud deployment

### CI/CD & Automation
- `GitHub Actions` — Automated training pipeline on push
- `Jenkins` — Job-based training trigger (GCP Vertex AI)

### Cloud Platforms
- `Azure ML` — Workspace, environments, MLflow integration
- `GCP Vertex AI` — Cloud training jobs
- `Azure Container Registry / GCP Artifact Registry` — Docker image hosting

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                           │
│                                                                 │
│  [Data] ──DVC──► [Preprocess] ──► [Train] ──► [Evaluate]       │
│                                      │             │            │
│                                   MLflow        MLflow          │
│                                   (log)        (register)       │
│                                      │                          │
│                                   [Model]                       │
│                                      │                          │
│                                  [FastAPI]                      │
│                                      │                          │
│                                  [Docker]                       │
│                                      │                          │
│                           [Kubernetes Cluster]                  │
│                                      │                          │
│                            /predict  endpoint                   │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment Breakdown

| # | Experiment | Tools | Outcome |
|---|---|---|---|
| 1 | Project Setup | Git, Conda, Azure ML | Reproducible repo & environment |
| 2 | CI/CD Pipeline | GitHub Actions | Auto-train on every push to `main` |
| 3 | Jenkins Training | Jenkins, GCP Vertex AI | Cloud job-triggered training |
| 4 | Inference API | FastAPI, Uvicorn | `/predict` REST endpoint |
| 5 | Containerization | Docker, AKS, GKE | Scalable cloud deployment |
| 6 | Experiment Tracking | MLflow | Logged params, metrics, artifacts |
| 7 | Data Versioning | DVC | Versioned data & models in cloud |
| 8 | Model Registry | MLflow Registry | Staged model promotion (Prod) |

---

## 📈 Results

| Metric | Value |
|---|---|
| **Model** | Random Forest Classifier |
| **Accuracy** | ~87% (on test split) |
| **Features Used** | 7 (CGPA, Internships, Projects, Aptitude, SoftSkills, SSC, HSC) |
| **Train / Test Split** | 80% / 20% |
| **Tracking** | MLflow (logged per run) |
| **Registry Stage** | Promoted to `Production` via MLflow Model Registry |

> 📝 *Results vary per run. All experiments are tracked via MLflow for comparison.*

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Conda
- Docker Desktop
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-placement-mlops.git
cd student-placement-mlops
```

### 2. Set Up Environment

```bash
conda create -n placement-env python=3.10
conda activate placement-env
pip install -r requirements.txt
```

### 3. Pull Data with DVC

```bash
dvc pull
```

### 4. Train the Model

```bash
python src/train.py
```

### 5. View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

### 6. Run the FastAPI Server

```bash
uvicorn api.app:app --reload
# Open http://localhost:8000/docs
```

### 7. Make a Prediction (Sample Request)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"CGPA": 8.5, "Internships": 2, "Projects": 3, "AptitudeTestScore": 78, "SoftSkillsRating": 4.2, "SSC_Marks": 85, "HSC_Marks": 80}'
```

**Response:**
```json
{
  "placed": true,
  "probability": 0.874
}
```

### 8. Build & Run with Docker

```bash
docker build -t placement-api .
docker run -p 8000:8000 placement-api
```

---

## 📁 Project Structure

```
student-placement-mlops/
├── data/
│   └── placementdata.csv        # Raw dataset (tracked by DVC)
├── src/
│   ├── preprocess.py            # Data cleaning & feature engineering
│   ├── train.py                 # Model training + MLflow logging
│   └── predict.py               # Inference logic
├── api/
│   └── app.py                   # FastAPI application
├── models/
│   └── placement_model.pkl      # Trained model artifact (tracked by DVC)
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis
├── .github/
│   └── workflows/
│       └── train.yml            # GitHub Actions CI/CD
├── Jenkinsfile                  # Jenkins pipeline definition
├── Dockerfile                   # Container definition
├── dvc.yaml                     # DVC pipeline stages
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots / Output

### MLflow Experiment Dashboard
> Tracks every training run with parameters (`n_estimators`, `max_depth`) and metrics (`accuracy`, `f1_score`).

```
Experiment: student-placement
Run ID: a3f8c1...     Accuracy: 0.872    n_estimators: 100
Run ID: b7d2e4...     Accuracy: 0.861    n_estimators: 50
Run ID: c9a1f6...     Accuracy: 0.879    n_estimators: 200  ← Production
```

### FastAPI Swagger UI
> Auto-generated interactive API docs available at `http://localhost:8000/docs`

```
POST /predict
  Request Body: { CGPA, Internships, Projects, AptitudeTestScore,
                  SoftSkillsRating, SSC_Marks, HSC_Marks }
  Response:     { "placed": true, "probability": 0.874 }
```

### GitHub Actions Pipeline
```
✅ Checkout code
✅ Set up Python 3.10
✅ Install dependencies
✅ Run training script
✅ Log model to MLflow
```

### DVC Pipeline
```
$ dvc repro
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date!
```

### MLflow Model Registry
```
Model: StudentPlacementModel
  Version 1 → Staging
  Version 2 → Staging
  Version 3 → Production ✅
```

---

## 👤 Author

**[Your Name]**
MLOps Project | [Your College / Institution Name]
Subject: MLOps Lab | Academic Year: 2025–26

---

## 📄 License

This project is created for academic purposes as part of the MLOps lab curriculum.
