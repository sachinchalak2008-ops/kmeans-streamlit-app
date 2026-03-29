
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# ── Pure-NumPy K-Means ────────────────────────────────
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

    def transform(self, X):        return (X - self.mean_) / self.std_
    def inverse_transform(self, X): return X * self.std_ + self.mean_


# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(page_title="StudyCluster", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box}
.stApp{background:#0a0a0f!important;color:#e8e8f0!important;font-family:'Space Mono',monospace!important}
.stApp::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(124,92,252,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(124,92,252,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
[data-testid="stSidebar"]{background:#111118!important;border-right:1px solid #2a2a3a!important}
[data-testid="stSidebar"]>div{padding-top:24px!important}
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
.badge-c0{display:inline-block;padding:3px 10px;border-radius:5px;background:rgba(124,92,252,.12);color:#7c5cfc;border:1px solid rgba(124,92,252,.3);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-family:'Space Mono',monospace}
.badge-c1{display:inline-block;padding:3px 10px;border-radius:5px;background:rgba(255,190,50,.12);color:#ffbe32;border:1px solid rgba(255,190,50,.3);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-family:'Space Mono',monospace}
.sc-predict{display:flex;align-items:center;gap:16px;padding:18px;border-radius:12px;border:1px solid #2a2a3a;background:#0f0f16;margin-top:12px}
.sc-predict-badge{width:56px;height:56px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:800;font-size:11px;text-transform:uppercase;letter-spacing:1px;flex-shrink:0}
.sc-predict-badge.pass{background:rgba(0,229,160,.12);color:#00e5a0;border:1px solid rgba(0,229,160,.3)}
.sc-predict-badge.fail{background:rgba(255,77,109,.12);color:#ff4d6d;border:1px solid rgba(255,77,109,.3)}
.sc-predict-badge.cluster0{background:rgba(124,92,252,.12);color:#7c5cfc;border:1px solid rgba(124,92,252,.3)}
.sc-predict-badge.cluster1{background:rgba(255,190,50,.12);color:#ffbe32;border:1px solid rgba(255,190,50,.3)}
.sc-predict-info .hours{color:#6b6b80;font-size:11px;margin-bottom:4px;font-family:'Space Mono',monospace}
.sc-predict-info .verdict{font-family:'Syne',sans-serif;font-weight:700;font-size:17px}
.sc-predict-info .verdict.pass{color:#00e5a0}
.sc-predict-info .verdict.fail{color:#ff4d6d}
.sc-predict-info .verdict.cluster0{color:#7c5cfc}
.sc-predict-info .verdict.cluster1{color:#ffbe32}
.sc-centroids{display:flex;gap:12px}
.sc-centroid{flex:1;padding:14px;border-radius:10px;text-align:center;border:1px solid #2a2a3a}
.sc-centroid.pass{background:rgba(0,229,160,.05);border-color:rgba(0,229,160,.2)}
.sc-centroid.fail{background:rgba(255,77,109,.05);border-color:rgba(255,77,109,.2)}
.sc-centroid.cluster0{background:rgba(124,92,252,.05);border-color:rgba(124,92,252,.2)}
.sc-centroid.cluster1{background:rgba(255,190,50,.05);border-color:rgba(255,190,50,.2)}
.sc-centroid .cval{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;margin-bottom:3px}
.sc-centroid.pass .cval{color:#00e5a0}
.sc-centroid.fail .cval{color:#ff4d6d}
.sc-centroid.cluster0 .cval{color:#7c5cfc}
.sc-centroid.cluster1 .cval{color:#ffbe32}
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
.sc-sample-table{width:100%;border-collapse:collapse;font-size:11px;font-family:'Space Mono',monospace}
.sc-sample-table th{text-align:left;padding:7px 12px;color:#6b6b80;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #2a2a3a;font-family:'Syne',sans-serif}
.sc-sample-table td{padding:7px 12px;border-bottom:1px solid rgba(42,42,58,.4);color:#b0b0c0}
.sc-sample-table tr:last-child td{border-bottom:none}
.stButton>button{width:100%;background:#7c5cfc!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:'Space Mono',monospace!important;font-size:12px!important;font-weight:700!important;letter-spacing:.5px!important;padding:10px 16px!important;box-shadow:0 4px 20px rgba(124,92,252,.3)!important;transition:all .2s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 24px rgba(124,92,252,.45)!important;background:#8f72fd!important}
.stNumberInput input,.stTextInput input{background:#0f0f16!important;border:1px solid #2a2a3a!important;border-radius:8px!important;color:#e8e8f0!important;font-family:'Space Mono',monospace!important;font-size:13px!important}
.stNumberInput input:focus,.stTextInput input:focus{border-color:#7c5cfc!important;box-shadow:none!important}
.stNumberInput label,.stTextInput label,.stFileUploader label,.stSelectbox label{color:#6b6b80!important;font-family:'Space Mono',monospace!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:1px!important}
[data-testid="stFileUploader"]{background:#0f0f16!important;border:1px dashed #2a2a3a!important;border-radius:8px!important}
.sc-empty{text-align:center;padding:32px 20px;color:#6b6b80;font-size:12px;line-height:2;font-family:'Space Mono',monospace}
.sc-info{background:rgba(124,92,252,.06);border:1px solid rgba(124,92,252,.2);border-radius:10px;padding:12px 16px;font-size:11px;color:#9a8fd0;font-family:'Space Mono',monospace;line-height:1.8;margin-bottom:12px}
.sc-col-pill{display:inline-block;padding:2px 9px;margin:2px;border-radius:5px;background:rgba(124,92,252,.12);color:#7c5cfc;border:1px solid rgba(124,92,252,.25);font-size:10px;font-family:'Space Mono',monospace}
.sc-label-col{display:inline-block;padding:2px 9px;margin:2px;border-radius:5px;background:rgba(0,229,160,.10);color:#00e5a0;border:1px solid rgba(0,229,160,.25);font-size:10px;font-family:'Space Mono',monospace}
</style>
""", unsafe_allow_html=True)

# ── SAMPLE CSV (embedded, not used for training) ──────
SAMPLE_CSV_CONTENT = """Study_hours,result
1,fail
2,fail
3,fail
4,fail
5,pass
6,pass
7,pass
8,pass
"""

# ── SESSION STATE ─────────────────────────────────────
for k, v in {
    "model": None, "scaler": None, "cluster_labels": {},
    "training_data": [], "centroids": [], "inertia": None,
    "log": [], "prediction": None,
    "feature_col": None, "label_col": None, "has_label": False,
    "feature_unit": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── HELPERS ───────────────────────────────────────────
def add_log(msg, kind="ok"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log = [(ts, msg, kind)] + st.session_state.log[:19]

def clean_numeric(val):
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

def detect_columns(df):
    """Detect numeric feature column and optional label column."""
    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    # Try to pick a sensible feature col (first numeric col)
    feature_col = numeric_cols[0] if numeric_cols else None

    # Label col: text col OR numeric col with only 2 unique values
    label_col = None
    for c in text_cols:
        label_col = c
        break
    if label_col is None:
        for c in numeric_cols:
            if c != feature_col and df[c].nunique() <= 2:
                label_col = c
                break

    return feature_col, label_col

def parse_csv_dynamic(content):
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip() for c in df.columns]  # keep original casing for display
    # Normalize column names internally for detection
    df_norm = df.copy()
    df_norm.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    feature_col, label_col = detect_columns(df_norm)

    if feature_col is None:
        return None, None, None, "No numeric column found in CSV."

    df_norm[feature_col] = df_norm[feature_col].apply(clean_numeric)
    df_norm = df_norm.dropna(subset=[feature_col])

    return df_norm, feature_col, label_col, None

def determine_pass_fail(df, label_col, cluster_labels_raw, pass_cluster):
    """Returns True if label column has pass/fail style strings."""
    if label_col and label_col in df.columns:
        unique_vals = df[label_col].dropna().astype(str).str.strip().str.lower().unique()
        pass_words = {"pass", "yes", "1", "true", "good", "positive", "success", "high"}
        fail_words = {"fail", "no", "0", "false", "bad", "negative", "failure", "low"}
        has_pass = any(v in pass_words for v in unique_vals)
        has_fail = any(v in fail_words for v in unique_vals)
        return has_pass or has_fail
    return False

def run_train(df, feature_col, label_col):
    X = df[feature_col].values.reshape(-1, 1).astype(float)
    scaler = ScalerNumpy()
    X_sc = scaler.fit_transform(X)
    km = KMeansNumpy(k=2, random_state=42).fit(X_sc)

    df = df.copy()
    df["__cluster__"] = km.labels_

    pass_cluster = int(df.groupby("__cluster__")[feature_col].mean().idxmax())
    has_label = label_col is not None and label_col in df.columns

    # Determine label semantics
    use_pass_fail = False
    cluster_labels = {}
    if has_label:
        unique_vals = df[label_col].dropna().astype(str).str.strip().str.lower().unique()
        pass_words = {"pass", "yes", "1", "true", "good", "positive", "success", "high"}
        fail_words = {"fail", "no", "0", "false", "bad", "negative", "failure", "low"}
        has_pos = any(v in pass_words for v in unique_vals)
        has_neg = any(v in fail_words for v in unique_vals)
        use_pass_fail = has_pos or has_neg

    if use_pass_fail:
        cluster_labels = {pass_cluster: "pass", 1 - pass_cluster: "fail"}
    else:
        cluster_labels = {pass_cluster: "cluster1", 1 - pass_cluster: "cluster0"}

    centroids_orig = scaler.inverse_transform(km.centroids_)
    training_data = []
    for _, row in df.iterrows():
        cid = int(row["__cluster__"])
        e = {
            "feature": float(row[feature_col]),
            "cluster": cid,
            "label": cluster_labels[cid],
        }
        if has_label:
            actual_raw = str(row.get(label_col, "")).strip()
            if actual_raw.lower() not in ("nan", ""):
                e["actual"] = actual_raw
        training_data.append(e)

    st.session_state.update({
        "model": km, "scaler": scaler, "cluster_labels": cluster_labels,
        "training_data": training_data, "inertia": km.inertia_,
        "feature_col": feature_col, "label_col": label_col,
        "has_label": has_label,
        "centroids": [
            {"cluster": i, "centroid": float(centroids_orig[i][0]), "label": cluster_labels[i]}
            for i in range(2)
        ],
        "prediction": None,
    })
    add_log(f"Trained on {len(training_data)} rows · feature='{feature_col}' · inertia {km.inertia_:.2f}")


# ── HEADER ────────────────────────────────────────────
trained = st.session_state.model is not None
feature_col = st.session_state.feature_col or "feature"
label_col   = st.session_state.label_col or ""
has_label   = st.session_state.has_label

st.markdown(f"""
<div class="sc-header">
  <div class="sc-logo">SC</div>
  <div>
    <div class="sc-title">Study<span>Cluster</span></div>
    <div class="sc-subtitle">K-Means · Dynamic CSV Predictor</div>
  </div>
  <div class="sc-status">
    <span class="{'sc-dot-on' if trained else 'sc-dot-off'}"></span>
    {'Model ready · <b>' + feature_col + '</b>' if trained else 'Not trained'}
  </div>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:

    # ── Sample CSV Preview ────────────────────────────
    st.markdown('<div class="sc-card"><div class="sc-card-title">📄 Sample CSV</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sc-info">
    No CSV? Use the sample below.<br>
    Download it or click <b>Train with Sample</b>.
    </div>""", unsafe_allow_html=True)

    sample_df = pd.read_csv(io.StringIO(SAMPLE_CSV_CONTENT))
    sample_rows = "".join(
        f"<tr>{''.join(f'<td>{v}</td>' for v in row)}</tr>"
        for row in sample_df.values
    )
    sample_headers = "".join(f"<th>{c}</th>" for c in sample_df.columns)
    st.markdown(f"""
    <div style="max-height:180px;overflow-y:auto;margin-bottom:10px">
    <table class="sc-sample-table">
      <thead><tr>{sample_headers}</tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    st.download_button(
        label="⬇ Download Sample CSV",
        data=SAMPLE_CSV_CONTENT,
        file_name="study_hours_result.csv",
        mime="text/csv",
    )
    if st.button("Train with Sample"):
        sample_bytes = SAMPLE_CSV_CONTENT.encode()
        df_s, fc, lc, err = parse_csv_dynamic(sample_bytes)
        if err:
            add_log(err, "err")
        else:
            run_train(df_s, fc, lc)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload & Train ────────────────────────────────
    st.markdown('<div class="sc-card"><div class="sc-card-title">Train Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sc-info">
    Upload any CSV. The first numeric column is used as the feature.
    A text/binary column is treated as the label (optional).
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        # Preview detected columns before training
        try:
            preview_bytes = uploaded.read()
            uploaded.seek(0)
            df_prev, fc_prev, lc_prev, err_prev = parse_csv_dynamic(preview_bytes)
            if err_prev:
                st.markdown(f'<div class="sc-info" style="color:#ff4d6d">{err_prev}</div>', unsafe_allow_html=True)
            else:
                pills = "".join(f'<span class="sc-col-pill">{c}</span>' for c in df_prev.columns if c != lc_prev)
                label_pill = f'<span class="sc-label-col">{lc_prev} (label)</span>' if lc_prev else ""
                st.markdown(f"""
                <div class="sc-info">
                <b>Detected columns:</b><br>{pills}{label_pill}<br>
                <b>Feature:</b> <span class="sc-col-pill">{fc_prev}</span>
                </div>""", unsafe_allow_html=True)
        except Exception:
            pass

    if st.button("Upload & Train"):
        if uploaded:
            uploaded.seek(0)
            raw = uploaded.read()
            df_u, fc_u, lc_u, err = parse_csv_dynamic(raw)
            if err:
                add_log(err, "err")
            else:
                run_train(df_u, fc_u, lc_u)
            st.rerun()
        else:
            add_log("No CSV uploaded", "err")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict ───────────────────────────────────────
    st.markdown('<div class="sc-card"><div class="sc-card-title">Predict</div>', unsafe_allow_html=True)
    feature_label = feature_col.replace("_", " ").title() if trained else "Feature Value"
    input_val = st.number_input(feature_label, min_value=0.0, step=0.5, value=4.0)

    if st.button("Run Prediction"):
        if not trained:
            add_log("Train the model first!", "err")
        else:
            X_in = np.array([[input_val]])
            cluster = int(st.session_state.model.predict(st.session_state.scaler.transform(X_in))[0])
            label = st.session_state.cluster_labels[cluster]
            st.session_state.prediction = {"value": input_val, "cluster": cluster, "label": label}
            add_log(f"Predict({feature_col}={input_val}) → {label.upper()}"); st.rerun()

    if st.session_state.prediction:
        p = st.session_state.prediction
        lbl = p["label"]
        is_pf = lbl in ("pass", "fail")
        badge_cls = lbl if is_pf else ("cluster0" if lbl == "cluster0" else "cluster1")
        if is_pf:
            verdict_txt = f"✓ Predicted to PASS" if lbl == "pass" else "✗ Predicted to FAIL"
        else:
            verdict_txt = f"→ Assigned to {lbl.replace('cluster','Cluster ')}"
        st.markdown(f"""
        <div class="sc-predict">
          <div class="sc-predict-badge {badge_cls}">{lbl.upper()}</div>
          <div class="sc-predict-info">
            <div class="hours">{feature_label}: {p['value']} → Cluster {p['cluster']}</div>
            <div class="verdict {badge_cls}">{verdict_txt}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cluster Centroids ─────────────────────────────
    st.markdown('<div class="sc-card"><div class="sc-card-title">Cluster Centroids</div>', unsafe_allow_html=True)
    if st.session_state.centroids:
        cards = "".join([
            f'<div class="sc-centroid {c["label"]}">'
            f'<div class="cval">{c["centroid"]:.2f}</div>'
            f'<div class="clbl">Cluster {c["cluster"]} · {c["label"].upper()}</div></div>'
            for c in st.session_state.centroids
        ])
        st.markdown(f'<div class="sc-centroids">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sc-empty">Train the model<br>to see centroids</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Activity Log ──────────────────────────────────
    st.markdown('<div class="sc-card"><div class="sc-card-title">Activity Log</div>', unsafe_allow_html=True)
    if st.session_state.log:
        entries = "".join([
            f'<div class="entry"><span class="ts">[{ts}]</span> <span class="{k}">{m}</span></div>'
            for ts, m, k in st.session_state.log
        ])
        st.markdown(f'<div class="sc-log">{entries}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sc-empty">No activity yet</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────
td = st.session_state.training_data
inertia_val = f'{st.session_state.inertia:.2f}' if st.session_state.inertia else "—"

# Dynamic metric labels
if trained:
    is_pf_mode = all(d["label"] in ("pass", "fail") for d in td)
    if is_pf_mode:
        pos_count = sum(1 for d in td if d["label"] == "pass")
        neg_count = sum(1 for d in td if d["label"] == "fail")
        m1_lbl, m2_lbl = "Pass Records", "Fail Records"
        m1_color, m2_color = "#00e5a0", "#ff4d6d"
    else:
        pos_count = sum(1 for d in td if d["label"] == "cluster1")
        neg_count = sum(1 for d in td if d["label"] == "cluster0")
        m1_lbl = f"High {feature_col.replace('_',' ').title()}"
        m2_lbl = f"Low {feature_col.replace('_',' ').title()}"
        m1_color, m2_color = "#00e5a0", "#ff4d6d"
else:
    pos_count = neg_count = 0
    m1_lbl, m2_lbl = "Group A", "Group B"
    m1_color, m2_color = "#00e5a0", "#ff4d6d"
    is_pf_mode = False

st.markdown(f"""
<div class="sc-card">
  <div class="sc-card-title">Model Metrics {('· ' + feature_col.replace('_',' ').title()) if trained else ''}</div>
  <div class="sc-metrics">
    <div class="sc-metric"><div class="sc-metric-val" style="color:{m1_color}">{pos_count if td else '—'}</div><div class="sc-metric-lbl">{m1_lbl}</div></div>
    <div class="sc-metric"><div class="sc-metric-val" style="color:{m2_color}">{neg_count if td else '—'}</div><div class="sc-metric-lbl">{m2_lbl}</div></div>
    <div class="sc-metric"><div class="sc-metric-val" style="color:#7c5cfc">{inertia_val}</div><div class="sc-metric-lbl">Inertia</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Cluster Visualization ─────────────────────────────
import plotly.graph_objects as go

x_axis_label = feature_col.replace("_", " ").title() if trained else "Feature"
st.markdown('<div class="sc-card"><div class="sc-card-title">Cluster Visualization</div>', unsafe_allow_html=True)
if td:
    fig = go.Figure()
    COLORS = {
        "pass":     ("rgba(0,229,160,0.8)",   "#00e5a0",  1),
        "fail":     ("rgba(255,77,109,0.8)",   "#ff4d6d",  0),
        "cluster1": ("rgba(0,229,160,0.8)",    "#00e5a0",  1),
        "cluster0": ("rgba(124,92,252,0.8)",   "#7c5cfc",  0),
    }
    unique_labels = list(dict.fromkeys(d["label"] for d in td))
    for lbl in unique_labels:
        pts = [(d["feature"], COLORS.get(lbl, ("rgba(200,200,200,0.8)", "#aaa", 0))[2]) for d in td if d["label"] == lbl]
        color, line_color, y_val = COLORS.get(lbl, ("rgba(200,200,200,0.8)", "#aaa", unique_labels.index(lbl)))
        hover_name = lbl.upper()
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[y_val] * len(pts),
            mode="markers", name=hover_name,
            marker=dict(color=color, size=14, line=dict(color=line_color, width=1.5)),
            hovertemplate=f"<b>{hover_name}</b><br>{x_axis_label}: %{{x}}<extra></extra>"
        ))

    if st.session_state.prediction:
        p = st.session_state.prediction
        lbl = p["label"]
        color, _, _ = COLORS.get(lbl, ("rgba(200,200,200,0.8)", "#aaa", 0))
        fig.add_trace(go.Scatter(
            x=[p["value"]], y=[0.5], mode="markers",
            name=f"⬟ Prediction ({p['value']})",
            marker=dict(color=color, size=16, symbol="diamond", line=dict(color="#fff", width=2)),
            hovertemplate=f"<b>PREDICTED: {lbl.upper()}</b><br>{x_axis_label}: {p['value']}<extra></extra>"
        ))

    y_tick_vals = list(set(COLORS.get(d["label"], (None, None, unique_labels.index(d["label"])))[2] for d in td))
    y_tick_text = []
    seen = {}
    for d in td:
        lbl = d["label"]
        yv = COLORS.get(lbl, (None, None, unique_labels.index(lbl)))[2]
        if yv not in seen:
            seen[yv] = lbl.upper()
    y_tick_text = [seen.get(v, str(v)) for v in sorted(seen.keys())]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,22,1)",
        font=dict(family="Space Mono, monospace", color="#6b6b80", size=11),
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2a3a", borderwidth=1, font=dict(size=11)),
        xaxis=dict(title=x_axis_label, gridcolor="#1a1a26", zerolinecolor="#2a2a3a", tickfont=dict(size=10)),
        yaxis=dict(title="Cluster", gridcolor="#1a1a26", zerolinecolor="#2a2a3a", tickfont=dict(size=10),
                   tickvals=sorted(seen.keys()), ticktext=y_tick_text, range=[-0.5, 1.6])
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown('<div class="sc-empty">Train the model to see the cluster visualization</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Training Data Table ───────────────────────────────
feat_label_display = feature_col.replace("_", " ").title() if trained else "Feature"
actual_label_display = label_col.replace("_", " ").title() if label_col else "Actual"

st.markdown('<div class="sc-card"><div class="sc-card-title">Training Data</div>', unsafe_allow_html=True)
if td:
    actual_header = f"<th>{actual_label_display}</th>" if has_label else ""
    rows = ""
    for i, d in enumerate(td, 1):
        lbl = d["label"]
        is_pf = lbl in ("pass", "fail")
        badge_cls = lbl if is_pf else ("badge-c0" if lbl == "cluster0" else "badge-c1")
        badge_html = f"<span class='badge-{lbl if is_pf else (\"c0\" if lbl==\"cluster0\" else \"c1\")}'>{lbl.upper()}</span>"
        actual_html = ""
        if has_label:
            act = d.get("actual", "")
            if act:
                act_lower = act.lower()
                if act_lower in ("pass", "yes", "1", "true", "good"):
                    actual_html = f"<td><span class='badge-pass'>{act}</span></td>"
                elif act_lower in ("fail", "no", "0", "false", "bad"):
                    actual_html = f"<td><span class='badge-fail'>{act}</span></td>"
                else:
                    actual_html = f"<td>{act}</td>"
            else:
                actual_html = "<td><span style='color:#6b6b80'>—</span></td>"
        rows += (
            f"<tr><td style='color:#6b6b80'>{i}</td>"
            f"<td>{d['feature']}</td>"
            f"<td style='color:#6b6b80'>{d['cluster']}</td>"
            f"<td>{badge_html}</td>"
            f"{actual_html}</tr>"
        )
    actual_th = f"<th>{actual_label_display}</th>" if has_label else ""
    st.markdown(f"""
    <div style="max-height:360px;overflow-y:auto">
    <table class="sc-table">
      <thead><tr><th>#</th><th>{feat_label_display}</th><th>Cluster</th><th>Predicted</th>{actual_th}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="sc-empty">No training data yet.<br>Upload a CSV or use the sample to train the model.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
