# StudyCluster — K-Means Pass/Fail Predictor

A full-stack ML deployment using K-Means clustering to predict student pass/fail from study hours.

---

## Project Structure

```
Project/
├── backend/
│   ├── main.py            # FastAPI app
│   └── requirements.txt
├── frontend/
│   └── index.html         # Single-file dashboard
└── README.md
```

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Frontend

Just open `frontend/index.html` in your browser. No build step needed.

> **Note:** If you get CORS errors, serve via a local server:
> ```bash
> cd frontend
> python -m http.server 3000
> ```
> Then open http://localhost:3000

---

## Data Note

Your uploaded CSV had a formatting issue in row 6 (`7.pass` merged in one cell). The backend automatically cleans this — it splits on `.` and takes the numeric part.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + model status |
| `GET` | `/sample-data` | Returns built-in sample dataset |
| `POST` | `/train` | Train on built-in sample data |
| `POST` | `/train/upload` | Train on uploaded CSV file |
| `POST` | `/predict` | Predict pass/fail for given study hours |
| `GET` | `/clusters` | Get centroids, assignments, and metrics |

### Example: Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 4.5}'
```

Response:
```json
{
  "study_hours": 4.5,
  "cluster": 1,
  "prediction": "fail"
}
```

---

## How It Works

- K-Means runs with **k=2** (one cluster per outcome)
- The cluster with the **higher mean study hours** is automatically labeled **"pass"**
- The model persists to `backend/saved_models/` via joblib so it survives restarts

---

## CSV Format

Your CSV should have at minimum a `study_hours` column. A `result` column is optional (used to show actual vs predicted in the table).

```
Study_hours,result
1,fail
2,fail
5,pass
8,pass
```
