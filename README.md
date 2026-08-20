# Student Placement Prediction

A machine learning project built for the MLOps lab that predicts whether a student will get placed based on their academic profile. The goal was to go beyond just training a model — the whole point was to build the infrastructure around it: versioning, automation, deployment, and monitoring.

---

## Why this project

Placement season is stressful, and most of the time nobody really knows which factors matter most until after the results. This project tries to answer that question using historical data. More importantly, it gave a reason to wire together the entire MLOps stack — Git, DVC, MLflow, Docker, GitHub Actions, Jenkins, FastAPI — into something that actually runs end to end.

---

## The dataset

10,000 student records with the following columns:

| Column | What it means |
|--------|--------------|
| CGPA | Overall GPA on a 10-point scale |
| Internships | How many internships done |
| Projects | Number of projects completed |
| AptitudeTestScore | Score on aptitude test (out of 100) |
| SoftSkillsRating | Communication and interpersonal rating (1–5) |
| SSC_Marks | 10th board exam percentage |
| HSC_Marks | 12th board exam percentage |
| PlacementStatus | **Target** — Placed or Not Placed |

---

## Stack

- **Model:** scikit-learn (Random Forest), pandas, numpy, joblib
- **Experiment tracking:** MLflow
- **Data versioning:** DVC
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker → pushed to Azure Container Registry / GCP Artifact Registry
- **Orchestration:** Kubernetes (AKS or GKE)
- **CI/CD:** GitHub Actions, Jenkins
- **Cloud:** Azure ML, GCP Vertex AI

---

## How the pipeline works

Training the model is just one step. Here's the full flow:

```
placementdata.csv
      │
     DVC (version the data)
      │
  preprocess.py
      │
   train.py ──────── MLflow (log params + metrics)
      │
 placement_model.pkl
      │
     DVC (version the model)
      │
   app.py (FastAPI)
      │
   Dockerfile
      │
   Docker image ──── pushed to registry
      │
  Kubernetes pod
      │
  /predict endpoint
```

Whenever code is pushed to `main`, GitHub Actions re-runs training automatically. Jenkins handles triggering cloud training jobs on GCP Vertex AI separately.

### The 8 experiments

| # | What was done |
|---|---------------|
| 1 | Set up Git repo + conda env + connected to Azure ML workspace |
| 2 | GitHub Actions workflow to retrain on every push |
| 3 | Jenkins job that triggers training on GCP Vertex AI |
| 4 | FastAPI app with a `/predict` endpoint, tested locally with Uvicorn |
| 5 | Dockerized the API and deployed to Kubernetes (AKS) |
| 6 | MLflow tracking — params, metrics, and model artifacts logged per run |
| 7 | DVC set up for the dataset and model file, remote on Azure Blob |
| 8 | MLflow Model Registry — versioned the model and promoted to Production |

---

## Results

Random Forest with 200 estimators came out on top across runs. Tracked everything through MLflow so the comparison was easy.

```
Run 1   n_estimators=50    accuracy=0.861
Run 2   n_estimators=100   accuracy=0.872
Run 3   n_estimators=200   accuracy=0.879   ← registered as Production
```

CGPA, AptitudeTestScore, and Internships turned out to be the most important features. SSC and HSC marks mattered less than expected.

---

## Running it locally

**Requirements:** Python 3.10, Conda, Docker, Git

```bash
# Clone
git clone https://github.com/your-username/student-placement-mlops.git
cd student-placement-mlops

# Environment
conda create -n placement-env python=3.10
conda activate placement-env
pip install -r requirements.txt

# Pull data
dvc pull

# Train
python src/train.py

# Check runs in MLflow
mlflow ui
# → http://localhost:5000

# Start the API
uvicorn api.app:app --reload
# → http://localhost:8000/docs
```

**Test the API:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CGPA": 8.5,
    "Internships": 2,
    "Projects": 3,
    "AptitudeTestScore": 78,
    "SoftSkillsRating": 4.2,
    "SSC_Marks": 85,
    "HSC_Marks": 80
  }'
```

```json
{ "placed": true, "probability": 0.874 }
```

**Docker:**

```bash
docker build -t placement-api .
docker run -p 8000:8000 placement-api
```

---

## Folder structure

```
student-placement-mlops/
├── data/
│   └── placementdata.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── api/
│   └── app.py
├── models/
│   └── placement_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── .github/
│   └── workflows/
│       └── train.yml
├── Jenkinsfile
├── Dockerfile
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## Sample outputs

**MLflow registry after 3 runs:**
```
StudentPlacementModel
  v1  →  Staging
  v2  →  Staging
  v3  →  Production
```

**GitHub Actions on push:**
```
✓  Set up Python
✓  Install dependencies
✓  Run train.py
✓  Model logged to MLflow
```

**DVC after data changes:**
```
$ dvc status
data/placementdata.csv: modified
$ dvc add data/placementdata.csv && dvc push
```

---

## About

Built as part of the MLOps Lab — [Your College Name]  
[Your Name] | 2025–26
