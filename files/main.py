from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import io

app = FastAPI(title="K-Means Study Hours API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- In-memory state ---
model_state = {
    "kmeans": None,
    "scaler": None,
    "cluster_labels": {},   # maps cluster index -> "pass"/"fail"
    "training_data": [],    # list of {study_hours, cluster, label}
    "centroids": [],
    "inertia": None,
}

# --- Sample data (cleaned) ---
SAMPLE_DATA = [
    {"study_hours": 1, "result": "fail"},
    {"study_hours": 2, "result": "fail"},
    {"study_hours": 3, "result": "fail"},
    {"study_hours": 4, "result": "fail"},
    {"study_hours": 5, "result": "pass"},
    {"study_hours": 6, "result": "pass"},
    {"study_hours": 7, "result": "pass"},
    {"study_hours": 8, "result": "pass"},
]


def parse_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Clean merged cells like "7.pass"
    if "study_hours" in df.columns:
        def clean_hours(val):
            val = str(val).strip()
            # handle "7.pass" → 7
            if "." in val:
                parts = val.split(".")
                try:
                    return float(parts[0])
                except ValueError:
                    pass
            try:
                return float(val)
            except ValueError:
                return np.nan

        df["study_hours"] = df["study_hours"].apply(clean_hours)

    if "result" in df.columns:
        df["result"] = df["result"].str.strip().str.lower()

    df = df.dropna(subset=["study_hours"])
    return df


def train_model(data: list[dict]):
    """Train KMeans with k=2 and assign pass/fail labels to clusters."""
    df = pd.DataFrame(data)
    X = df["study_hours"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    df["cluster"] = kmeans.labels_

    # Determine which cluster is "pass" by higher mean study hours
    cluster_means = df.groupby("cluster")["study_hours"].mean()
    pass_cluster = int(cluster_means.idxmax())
    fail_cluster = 1 - pass_cluster

    cluster_labels = {pass_cluster: "pass", fail_cluster: "fail"}

    if "result" in df.columns:
        df["predicted_label"] = df["cluster"].map(cluster_labels)

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    training_data = []
    for _, row in df.iterrows():
        entry = {
            "study_hours": float(row["study_hours"]),
            "cluster": int(row["cluster"]),
            "label": cluster_labels[int(row["cluster"])],
        }
        if "result" in df.columns:
            entry["actual"] = str(row["result"]) if not pd.isna(row.get("result")) else None
        training_data.append(entry)

    # Persist
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(cluster_labels, f"{MODEL_DIR}/cluster_labels.pkl")

    model_state["kmeans"] = kmeans
    model_state["scaler"] = scaler
    model_state["cluster_labels"] = cluster_labels
    model_state["training_data"] = training_data
    model_state["centroids"] = [
        {"cluster": i, "centroid": float(centroids_original[i][0]), "label": cluster_labels[i]}
        for i in range(2)
    ]
    model_state["inertia"] = float(kmeans.inertia_)

    return {
        "message": "Model trained successfully",
        "n_samples": len(training_data),
        "centroids": model_state["centroids"],
        "inertia": model_state["inertia"],
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "training_data": training_data,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": model_state["kmeans"] is not None}


@app.get("/sample-data")
def get_sample_data():
    return {"data": SAMPLE_DATA}


@app.post("/train")
def train_on_sample():
    """Train using built-in sample data."""
    result = train_model(SAMPLE_DATA)
    return result


@app.post("/train/upload")
async def train_on_upload(file: UploadFile = File(...)):
    """Train using uploaded CSV."""
    content = await file.read()
    try:
        df = parse_csv(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if "study_hours" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'study_hours' column.")

    data = df.to_dict(orient="records")
    result = train_model(data)
    return result


class PredictRequest(BaseModel):
    study_hours: float


@app.post("/predict")
def predict(req: PredictRequest):
    if model_state["kmeans"] is None:
        # Auto-load from disk if available
        try:
            model_state["kmeans"] = joblib.load(f"{MODEL_DIR}/kmeans.pkl")
            model_state["scaler"] = joblib.load(f"{MODEL_DIR}/scaler.pkl")
            model_state["cluster_labels"] = joblib.load(f"{MODEL_DIR}/cluster_labels.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model not trained yet. POST /train first.")

    X = np.array([[req.study_hours]])
    X_scaled = model_state["scaler"].transform(X)
    cluster = int(model_state["kmeans"].predict(X_scaled)[0])
    label = model_state["cluster_labels"][cluster]

    return {
        "study_hours": req.study_hours,
        "cluster": cluster,
        "prediction": label,
    }


@app.get("/clusters")
def get_clusters():
    if not model_state["training_data"]:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    return {
        "centroids": model_state["centroids"],
        "inertia": model_state["inertia"],
        "training_data": model_state["training_data"],
        "cluster_labels": {str(k): v for k, v in model_state["cluster_labels"].items()},
    }
