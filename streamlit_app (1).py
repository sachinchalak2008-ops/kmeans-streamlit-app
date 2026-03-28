import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# ── Pure-NumPy K-Means (zero external ML deps) ───────
class KMeansNumpy:
    def __init__(self, k=2, max_iter=300, tol=1e-4, random_state=42):
        self.k, self.max_iter, self.tol = k, max_iter, tol
        self.random_state = random_state
        self.centroids_ = self.labels_ = self.inertia_ = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        centroids = X[rng.choice(len(X), self.k, replace=False)].copy().astype(float)
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
        self.centroids_, self.labels_ = centroids, labels
        self.inertia_ = float(sum(
            np.sum((X[labels == i] - centroids[i]) ** 2)
            for i in range(self.k) if np.any(labels == i)
        ))
        return self

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids_[None, :], axis=2), axis=1)


class ScalerNumpy:
    def fit_transform(self, X):
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_[self.std_ == 0] = 1
        return (X - self.mean_) / self.std_

    def transform(self, X):       return (X - self.mean_) / self.std_
    def inverse_transform(self, X): return X * self.std_ + self.mean_


# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(page_title="StudyCluster", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box}
.stApp{background:#0a0a0f!important;color:#e8e8f0!important;font-family:'Space Mono',monospace!important}
.stApp::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(124,92,252,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(124,92,252,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
[data-testid="stSidebar"]{background:#111118!important;border-right:1px solid #2a2a3a!important}
[data-testid="stSidebar"]>div{padding-top:24px!important}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stToolbar"]{display:none}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:#e8e8f0!important}
.sc-card{background:#16161f;border:1px solid #2a2a3a;border-radius:16px;padding:20px 22px;margin-bottom:16px}
.sc-card-title{font-family:'Syne',sans-serif;font-weight:700;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#6b6b80;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.sc-card-title::before{content:'';display:inline-block;width:3px;height:13px;background:#7c5cfc;border-radius:2px}
.sc-header{display:flex;align-items:flex-end;gap:18px;padding-bottom:24px;border-bottom:1px solid #2a2a3a;margin-bottom:28px}
.sc-logo{width:52px;height:52px;background:#7c5cfc;border-radius:12px;display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:800;font-size:20px;color:#fff;box-shadow:0 0 24px rgba(124,92,252,.4);flex-shrink:0}
.sc-title{font-family:'Syne',sans-serif;font-weight:800;font-size:30px;line-height:1;letter-spacing:-.5px}
.sc-title span{color:#7c5cfc}
.sc-subtitle{color:#6b6b80;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-top:5px;font-family:'Space Mono',monospace}
.sc-status{margin-left:auto;display:flex;align-items:center;gap:8px;padding:7px 16px;background:#16161f;border:1px solid #2a2a3a;border-radius:100px;font-size:11px;color:#6b6b80;font-family:'Space Mono',monospace}
.sc-dot-on{width:7px;height:7px;border-radius:50%;background:#00e5a0;box-shadow:0 0 8px #00e5a0;display:inline-block}
.sc-dot-off{width:7px;height:7px;border-radius:50%;background:#6b6b80;display:inline-block}
.sc-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.sc-metric{background:#0f0f16;border:1px solid #2a2a3a;border-radius:12px;padding:16px;text-align:center}
.sc-metric-val{font-family:'Syne',sans-serif;font-weight:800;font-size:26px;line-height:1;margin-bottom:5px}
.sc-metric-lbl{color:#6b6b80;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-family:'Space Mono',monospace}
.badge-pass{display:inline-block;padding:3px 10px;border-radius:5px;background:rgba(0,229,160,.12);color:#00e5a0;border:1px solid rgba(0,229,160,.3);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-family:'Space Mono',monospace}
.badge-fail{display:inline-block;padding:3px 10px;border-radius:5px;background:rgba(255,77,109,.12);color:#ff4d6d;border:1px solid rgba(255,77,109,.3);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-family:'Space Mono',monospace}
.sc-predict{display:flex;align-items:center;gap:16px;padding:18px;border-radius:12px;border:1px solid #2a2a3a;background:#0f0f16;margin-top:12px}
.sc-predict-badge{width:56px;height:56px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:800;font-size:11px;text-transform:uppercase;letter-spacing:1px;flex-shrink:0}
.sc-predict-badge.pass{background:rgba(0,229,160,.12);color:#00e5a0;border:1px solid rgba(0,229,160,.3)}
.sc-predict-badge.fail{background:rgba(255,77,109,.12);color:#ff4d6d;border:1px solid rgba(255,77,109,.3)}
.sc-predict-info .hours{color:#6b6b80;font-size:11px;margin-bottom:4px;font-family:'Space Mono',monospace}
.sc-predict-info .verdict{font-family:'Syne',sans-serif;font-weight:700;font-size:17px}
.sc-predict-info .verdict.pass{color:#00e5a0}
.sc-predict-info .verdict.fail{color:#ff4d6d}
.sc-centroids{display:flex;gap:12px}
.sc-centroid{flex:1;padding:14px;border-radius:10px;text-align:center;border:1px solid #2a2a3a}
.sc-centroid.pass{background:rgba(0,229,160,.05);border-color:rgba(0,229,160,.2)}
.sc-centroid.fail{background:rgba(255,77,109,.05);border-color:rgba(255,77,109,.2)}
.sc-centroid .cval{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;margin-bottom:3px}
.sc-centroid.pass .cval{color:#00e5a0}
.sc-centroid.fail .cval{color:#ff4d6d}
.sc-centroid .clbl{color:#6b6b80;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-family:'Space Mono',monospace}
.sc-log{font-size:11px;line-height:2;font-family:'Space Mono',monospace;max-height:120px;overflow-y:auto}
.sc-log .ts{color:#7c5cfc}
.sc-log .ok{color:#00e5a0}
.sc-log .err{color:#ff4d6d}
.sc-table{width:100%;border-collapse:collapse;font-size:12px;font-family:'Space Mono',monospace}
.sc-table th{text-align:left;padding:9px 14px;color:#6b6b80;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #2a2a3a;font-family:'Syne',sans-serif}
.sc-table td{padding:10px 14px;border-bottom:1px solid rgba(42,42,58,.5);color:#e8e8f0}
.sc-table tr:last-child td{border-bottom:none}
.sc-table tr:hover td{background:rgba(124,92,252,.04)}
.stButton>button{width:100%;background:#7c5cfc!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:'Space Mono',monospace!important;font-size:12px!important;font-weight:700!important;letter-spacing:.5px!important;padding:10px 16px!important;box-shadow:0 4px 20px rgba(124,92,252,.3)!important;transition:all .2s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 24px rgba(124,92,252,.45)!important;background:#8f72fd!important}
.stNumberInput input{background:#0f0f16!important;border:1px solid #2a2a3a!important;border-radius:8px!important;color:#e8e8f0!important;font-family:'Space Mono',monospace!important;font-size:13px!important}
.stNumberInput input:focus{border-color:#7c5cfc!important;box-shadow:none!important}
.stNumberInput label,.stFileUploader label{color:#6b6b80!important;font-family:'Space Mono',monospace!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:1px!important}
[data-testid="stFileUploader"]{background:#0f0f16!important;border:1px dashed #2a2a3a!important;border-radius:8px!important}
.sc-empty{text-align:center;padding:32px 20px;color:#6b6b80;font-size:12px;line-height:2;font-family:'Space Mono',monospace}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────
for k, v in {"model": None, "scaler": None, "cluster_labels": {}, "training_data": [],
              "centroids": [], "inertia": None, "log": [], "prediction": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

SAMPLE_DATA = [
    {"study_hours": 1, "result": "fail"}, {"study_hours": 2, "result": "fail"},
    {"study_hours": 3, "result": "fail"}, {"study_hours": 4, "result": "fail"},
    {"study_hours": 5, "result": "pass"}, {"study_hours": 6, "result": "pass"},
    {"study_hours": 7, "result": "pass"}, {"study_hours": 8, "result": "pass"},
]

# ── HELPERS ──────────────────────────────────────────
def add_log(msg, kind="ok"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log = [(ts, msg, kind)] + st.session_state.log[:19]

def clean_hours(val):
    s = str(val).strip()
    parts = s.split(".")
    try:
        return float(parts[0]) if len(parts) > 1 and not s.replace(".", "").lstrip("-").isdigit() else float(s)
    except ValueError:
        return np.nan

def parse_csv(content):
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "study_hours" in df.columns:
        df["study_hours"] = df["study_hours"].apply(clean_hours)
    if "result" in df.columns:
        df["result"] = df["result"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["study_hours"])

def run_train(data):
    df = pd.DataFrame(data)
    X = df["study_hours"].values.reshape(-1, 1).astype(float)
    scaler = ScalerNumpy()
    X_sc = scaler.fit_transform(X)
    km = KMeansNumpy(k=2, random_state=42).fit(X_sc)

    df["cluster"] = km.labels_
    pass_cluster = int(df.groupby("cluster")["study_hours"].mean().idxmax())
    cluster_labels = {pass_cluster: "pass", 1 - pass_cluster: "fail"}

    centroids_orig = scaler.inverse_transform(km.centroids_)
    training_data = []
    for _, row in df.iterrows():
        e = {"study_hours": float(row["study_hours"]), "cluster": int(row["cluster"]),
             "label": cluster_labels[int(row["cluster"])]}
        if "result" in df.columns and str(row.get("result", "")) not in ("nan", ""):
            e["actual"] = str(row["result"])
        training_data.append(e)

    st.session_state.update({
        "model": km, "scaler": scaler, "cluster_labels": cluster_labels,
        "training_data": training_data, "inertia": km.inertia_,
        "centroids": [{"cluster": i, "centroid": float(centroids_orig[i][0]),
                       "label": cluster_labels[i]} for i in range(2)],
    })
    add_log(f"Trained on {len(training_data)} samples · inertia {km.inertia_:.2f}")

# ── HEADER ───────────────────────────────────────────
trained = st.session_state.model is not None
st.markdown(f"""
<div class="sc-header">
  <div class="sc-logo">SC</div>
  <div>
    <div class="sc-title">Study<span>Cluster</span></div>
    <div class="sc-subtitle">K-Means · Pass/Fail Predictor</div>
  </div>
  <div class="sc-status">
    <span class="{'sc-dot-on' if trained else 'sc-dot-off'}"></span>
    {'Model ready' if trained else 'Not trained'}
  </div>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sc-card"><div class="sc-card-title">Train Model</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Sample"):
            run_train(SAMPLE_DATA); st.rerun()
    with c2:
        if st.button("Upload & Train"):
            if uploaded:
                run_train(parse_csv(uploaded.read()).to_dict(orient="records")); st.rerun()
            else:
                add_log("No CSV selected", "err")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-card"><div class="sc-card-title">Predict</div>', unsafe_allow_html=True)
    hours_input = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5, value=4.0)
    if st.button("Run Prediction"):
        if not trained:
            add_log("Train the model first!", "err")
        else:
            X_in = np.array([[hours_input]])
            cluster = int(st.session_state.model.predict(st.session_state.scaler.transform(X_in))[0])
            label = st.session_state.cluster_labels[cluster]
            st.session_state.prediction = {"hours": hours_input, "cluster": cluster, "label": label}
            add_log(f"Predict({hours_input}h) → {label.upper()}"); st.rerun()

    if st.session_state.prediction:
        p = st.session_state.prediction
        verdict = "✓ Predicted to PASS" if p["label"] == "pass" else "✗ Predicted to FAIL"
        st.markdown(f"""
        <div class="sc-predict">
          <div class="sc-predict-badge {p['label']}">{p['label'].upper()}</div>
          <div class="sc-predict-info">
            <div class="hours">{p['hours']}h → Cluster {p['cluster']}</div>
            <div class="verdict {p['label']}">{verdict}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-card"><div class="sc-card-title">Cluster Centroids</div>', unsafe_allow_html=True)
    if st.session_state.centroids:
        cards = "".join([f'<div class="sc-centroid {c["label"]}"><div class="cval">{c["centroid"]:.1f}h</div>'
                         f'<div class="clbl">Cluster {c["cluster"]} · {c["label"].upper()}</div></div>'
                         for c in st.session_state.centroids])
        st.markdown(f'<div class="sc-centroids">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sc-empty">Train the model<br>to see centroids</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-card"><div class="sc-card-title">Activity Log</div>', unsafe_allow_html=True)
    if st.session_state.log:
        entries = "".join([f'<div class="entry"><span class="ts">[{ts}]</span> <span class="{k}">{m}</span></div>'
                           for ts, m, k in st.session_state.log])
        st.markdown(f'<div class="sc-log">{entries}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sc-empty">No activity yet</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────
td = st.session_state.training_data
pass_count = sum(1 for d in td if d["label"] == "pass")
fail_count = sum(1 for d in td if d["label"] == "fail")
inertia_val = f'{st.session_state.inertia:.2f}' if st.session_state.inertia else "—"

st.markdown(f"""
<div class="sc-card">
  <div class="sc-card-title">Model Metrics</div>
  <div class="sc-metrics">
    <div class="sc-metric"><div class="sc-metric-val" style="color:#00e5a0">{pass_count if td else '—'}</div><div class="sc-metric-lbl">Pass Students</div></div>
    <div class="sc-metric"><div class="sc-metric-val" style="color:#ff4d6d">{fail_count if td else '—'}</div><div class="sc-metric-lbl">Fail Students</div></div>
    <div class="sc-metric"><div class="sc-metric-val" style="color:#7c5cfc">{inertia_val}</div><div class="sc-metric-lbl">Inertia</div></div>
  </div>
</div>""", unsafe_allow_html=True)

import plotly.graph_objects as go

st.markdown('<div class="sc-card"><div class="sc-card-title">Cluster Visualization</div>', unsafe_allow_html=True)
if td:
    fig = go.Figure()
    fail_pts = [(d["study_hours"], 0) for d in td if d["label"] == "fail"]
    pass_pts  = [(d["study_hours"], 1) for d in td if d["label"] == "pass"]
    if fail_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in fail_pts], y=[p[1] for p in fail_pts],
            mode="markers", name="Fail",
            marker=dict(color="rgba(255,77,109,0.8)", size=14, line=dict(color="#ff4d6d", width=1.5)),
            hovertemplate="<b>FAIL</b><br>%{x} study hours<extra></extra>"))
    if pass_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in pass_pts], y=[p[1] for p in pass_pts],
            mode="markers", name="Pass",
            marker=dict(color="rgba(0,229,160,0.8)", size=14, line=dict(color="#00e5a0", width=1.5)),
            hovertemplate="<b>PASS</b><br>%{x} study hours<extra></extra>"))
    if st.session_state.prediction:
        p = st.session_state.prediction
        color = "#00e5a0" if p["label"] == "pass" else "#ff4d6d"
        fig.add_trace(go.Scatter(x=[p["hours"]], y=[0.5], mode="markers",
            name=f"⬟ Prediction ({p['hours']}h)",
            marker=dict(color=color, size=16, symbol="diamond", line=dict(color="#fff", width=2)),
            hovertemplate=f"<b>PREDICTED: {p['label'].upper()}</b><br>{p['hours']} hours<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,22,1)",
        font=dict(family="Space Mono, monospace", color="#6b6b80", size=11),
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2a3a", borderwidth=1, font=dict(size=11)),
        xaxis=dict(title="Study Hours", gridcolor="#1a1a26", zerolinecolor="#2a2a3a", tickfont=dict(size=10)),
        yaxis=dict(title="Cluster", gridcolor="#1a1a26", zerolinecolor="#2a2a3a", tickfont=dict(size=10),
                   tickvals=[0, 1], ticktext=["Fail", "Pass"], range=[-0.4, 1.4]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown('<div class="sc-empty">Train the model to see the cluster visualization</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="sc-card"><div class="sc-card-title">Training Data</div>', unsafe_allow_html=True)
if td:
    rows = "".join([
        f"<tr><td style='color:#6b6b80'>{i}</td><td>{d['study_hours']}h</td>"
        f"<td style='color:#6b6b80'>{d['cluster']}</td>"
        f"<td><span class='badge-{d['label']}'>{d['label']}</span></td>"
        f"<td>{'<span class=\"badge-' + d.get('actual','') + '\">' + d.get('actual','') + '</span>' if d.get('actual') else '<span style=\"color:#6b6b80\">—</span>'}</td></tr>"
        for i, d in enumerate(td, 1)
    ])
    st.markdown(f"""
    <table class="sc-table">
      <thead><tr><th>#</th><th>Study Hours</th><th>Cluster</th><th>Predicted</th><th>Actual</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="sc-empty">No training data yet.<br>Train the model to populate this table.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
