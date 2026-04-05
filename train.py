# train.py — Experiment 6 & 8
# Trains two models, tracks metrics with MLflow on Azure ML
# Run: python train.py

from dotenv import load_dotenv
load_dotenv()

import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
import mlflow
import joblib

# ── MLflow setup ───────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "placement-prediction"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/placementdata.csv")

FEATURES = [
    "CGPA", "Internships", "Projects",
    "AptitudeTestScore", "SoftSkillsRating",
    "SSC_Marks", "HSC_Marks"
]

X = df[FEATURES]
y = (df["PlacementStatus"] == "Placed").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train both models, pick best ───────────────────────────────────────────
candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=500, random_state=42
    ),
}

best_f1    = 0
best_model = None
best_name  = None
best_meta  = {}
RUN_ID     = None

for name, clf in candidates.items():
    with mlflow.start_run(run_name=f"{name}-run") as run:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, preds),  4),
            "f1":        round(f1_score(y_test, preds),        4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall":    round(recall_score(y_test, preds),    4),
            "roc_auc":   round(roc_auc_score(y_test, proba),   4),
        }

        # Log only params and metrics to MLflow — no model logging
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_estimators", 150 if name == "RandomForest" else "N/A")
        mlflow.log_param("max_depth", 8 if name == "RandomForest" else "N/A")
        mlflow.log_metrics(metrics)

        print(f"  [{name}] {metrics}")

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_model = clf
            best_name  = name
            RUN_ID     = run.info.run_id
            best_meta  = {
                "run_id":     RUN_ID,
                "model_type": best_name,
                "features":   FEATURES,
                "metrics":    metrics
            }

# ── Save best model locally ────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
with open("models/metadata.json", "w") as f:
    json.dump(best_meta, f, indent=2)

print(f"\n{'='*50}")
print(f"  Best Model : {best_name}")
print(f"  F1 Score   : {best_f1}")
print(f"  Run ID     : {RUN_ID}")
print(f"  Saved      : models/model.pkl + models/metadata.json")
print(f"{'='*50}")
print(f"\n  MLflow UI  : mlflow ui -> http://127.0.0.1:5000")
print(f"  Azure ML   : https://ml.azure.com\n")
