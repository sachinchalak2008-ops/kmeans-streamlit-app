import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# ── KMeans ───────────────────────────────────────────
class KMeansNumpy:
    def __init__(self, k=2, max_iter=300, tol=1e-4, random_state=42):
        self.k, self.max_iter, self.tol = k, max_iter, tol
        self.random_state = random_state
        self.centroids_ = self.labels_ = self.inertia_ = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        centroids = X[rng.choice(len(X), self.k, replace=False)].copy()

        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_c = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.k)
            ])

            if np.linalg.norm(new_c - centroids) < self.tol:
                break

            centroids = new_c

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum((X - centroids[labels])**2))
        return self

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids_[None, :], axis=2), axis=1)


class ScalerNumpy:
    def fit_transform(self, X):
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_[self.std_ == 0] = 1
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_

# ── PAGE ─────────────────────────────────────────────
st.set_page_config(page_title="StudyCluster", layout="wide")

# ── STATE ────────────────────────────────────────────
for k in ["model", "scaler", "training_data", "columns", "inertia", "prediction"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── HELPERS ──────────────────────────────────────────
def add_log(msg):
    st.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def parse_csv(content):
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if numeric_df.shape[1] < 1:
        st.error("CSV must have at least one numeric column")
        return None, None

    return numeric_df, df

def run_train(df_numeric, df_original):
    X = df_numeric.values.astype(float)

    scaler = ScalerNumpy()
    X_scaled = scaler.fit_transform(X)

    model = KMeansNumpy(k=2).fit(X_scaled)

    df_original["cluster"] = model.labels_

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.training_data = df_original
    st.session_state.columns = df_numeric.columns.tolist()
    st.session_state.inertia = model.inertia_

# ── SIDEBAR ──────────────────────────────────────────
st.sidebar.title("Train Model")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("Train"):
    if uploaded:
        df_numeric, df_original = parse_csv(uploaded.read())

        if df_numeric is not None:
            run_train(df_numeric, df_original)
            st.success("Model trained successfully")
    else:
        st.warning("Upload a CSV first")

# ── PREDICTION ───────────────────────────────────────
st.sidebar.title("Prediction")

if st.session_state.model:
    inputs = []

    for col in st.session_state.columns:
        val = st.sidebar.number_input(f"{col}", value=0.0)
        inputs.append(val)

    if st.sidebar.button("Predict"):
        X = np.array([inputs])
        X_scaled = st.session_state.scaler.transform(X)

        cluster = int(st.session_state.model.predict(X_scaled)[0])

        st.session_state.prediction = cluster

# ── MAIN ─────────────────────────────────────────────
st.title("📊 StudyCluster")

# Metrics
if st.session_state.training_data is not None:
    st.subheader("Model Metrics")
    st.write("Inertia:", st.session_state.inertia)

# Visualization
if st.session_state.training_data is not None:
    st.subheader("Cluster Visualization")

    df = st.session_state.training_data
    cols = st.session_state.columns

    if len(cols) >= 2:
        import plotly.express as px

        fig = px.scatter(
            df,
            x=cols[0],
            y=cols[1],
            color=df["cluster"].astype(str),
            title="Clusters"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for visualization")

# Prediction Output
if st.session_state.prediction is not None:
    st.subheader("Prediction Result")
    st.write(f"Assigned to Cluster: {st.session_state.prediction}")

# Training Data
if st.session_state.training_data is not None:
    st.subheader("Training Data")
    st.dataframe(st.session_state.training_data)