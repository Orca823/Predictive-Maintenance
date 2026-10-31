
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from github import Github, GithubException

from pyngrok import ngrok
import subprocess
import mlflow

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

RANDOM_STATE = 42
TARGET = "Engine Condition"

# Start MLflow UI on port 5000
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Create public tunnel
public_url = ngrok.connect(5000).public_url
print("MLflow UI is available at:", public_url)

# Set the tracking URL for MLflow
mlflow.set_tracking_uri(public_url)

# Set the name for the experiment
mlflow.set_experiment("Predictive Maintanance")

models = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 3, 5],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    ),
}


def evaluate(model, X, y):
    """Return test-set metrics for a fitted classifier."""
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        if len(np.unique(proba)) > 1:  # ROC AUC is undefined for constant scores
            metrics["roc_auc"] = roc_auc_score(y, proba)
    return metrics


def log_metrics_to_mlflow(metrics, cv_f1=None):
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items() if not np.isnan(v)})
    if cv_f1 is not None:
        mlflow.log_metric("cv_f1", float(cv_f1))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
searches = {}
results = []

# Baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_metrics = evaluate(baseline, X_test, y_test)
results.append({"model": "Dummy baseline", "best_cv_f1": np.nan, **baseline_metrics})

with mlflow.start_run(run_name="Dummy baseline"):
    mlflow.log_params({"model_name": "Dummy baseline", "strategy": "most_frequent"})
    log_metrics_to_mlflow(baseline_metrics)

# Grid search for each model
for name, (estimator, param_grid) in models.items():
    print(f"Training/searching: {name}")
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    searches[name] = search

    metrics = evaluate(search, X_test, y_test)
    results.append({"model": name, "best_cv_f1": search.best_score_, **metrics})
    print("  Best parameters:", search.best_params_)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model_name": name, "random_state": RANDOM_STATE})
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        log_metrics_to_mlflow(metrics, cv_f1=search.best_score_)
        mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",  # use artifact_path="model" on MLflow < 3.0
            signature=infer_signature(X_train, search.best_estimator_.predict(X_train)),
            input_example=X_train[:5],
        )

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
display(results_df.round(4))
