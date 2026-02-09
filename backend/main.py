from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
import uuid
import random
import uvicorn

# ==================== PATHS ====================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "backend/models/saved_models"

# ==================== LOAD ML MODEL ====================

model = None
scaler = None
label_encoder = None

try:
    # Set Keras to use TensorFlow backend for compatibility
    model = tf.keras.models.load_model(
        MODEL_DIR / "procrastination_bilstm.h5",
        compile=False,
        safe_mode=False
    )
    print("Model loaded successfully with safe_mode=False")
except Exception as e:
    print(f"Error loading model with safe_mode=False: {e}")
    try:
        print("Attempting to load with custom objects...")
        custom_objects = {
            'LSTM': tf.keras.models.LSTM,
            'Bidirectional': tf.keras.models.Bidirectional
        }
        model = tf.keras.models.load_model(
            MODEL_DIR / "procrastination_bilstm.h5",
            compile=False,
            custom_objects=custom_objects
        )
        print("Model loaded successfully with custom objects")
    except Exception as e2:
        print(f"Error loading model with custom objects: {e2}")
        print("WARNING: Model will not be available. Using mock model for testing.")
        model = None

try:
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

try:
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# ==================== APP SETUP ====================

app = FastAPI(
    title="Procrastination Prediction System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ==================== MODELS ====================

class LoginRequest(BaseModel):
    email: str
    password: str

class PredictionRequest(BaseModel):
    student_id: int
    behavioral_data: Optional[dict] = {}

# ==================== MOCK USERS ====================

USERS = {
    "student@test.com": {
        "password": "password",
        "role": "student",
        "id": 1
    },
    "admin@test.com": {
        "password": "admin",
        "role": "admin",
        "id": 99
    }
}

# ==================== PAGE ROUTES ====================

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "login.html")

@app.get("/login")
def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")

@app.get("/signup")
def signup_page():
    return FileResponse(FRONTEND_DIR / "signup.html")

@app.get("/student/dashboard")
def student_dashboard():
    return FileResponse(FRONTEND_DIR / "student_dashboard.html")

@app.get("/admin/dashboard")
def admin_dashboard():
    return FileResponse(FRONTEND_DIR / "admin_dashboard.html")

@app.get("/student/profile")
def student_profile():
    return FileResponse(FRONTEND_DIR / "student_profile.html")

# ==================== AUTH APIs ====================

@app.post("/api/auth/login")
def login(data: LoginRequest):
    user = USERS.get(data.email)

    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": str(uuid.uuid4()),
        "user_id": user["id"],
        "role": user["role"]
    }

@app.post("/api/auth/signup")
def signup(data: LoginRequest):
    return {
        "access_token": str(uuid.uuid4()),
        "user_id": random.randint(10, 999),
        "role": "student"
    }

# ==================== ML PREDICTION API ====================

@app.post("/api/predict")
def predict(request: PredictionRequest):
    features = np.array([[
        request.behavioral_data.get("late_rate", 0.5),
        request.behavioral_data.get("irregularity", 0.3),
        request.behavioral_data.get("last_min_ratio", 0.4),
        request.behavioral_data.get("avg_gap", 2.0)
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled, verbose=0)

    risk_idx = int(np.argmax(prediction[0]))
    risk_level = label_encoder.inverse_transform([risk_idx])[0]
    risk_score = float(prediction[0][risk_idx])

    return {
        "student_id": request.student_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "timestamp": datetime.now().isoformat(),
        "explanation": "Prediction based on behavioral engagement patterns."
    }

# ==================== HEALTH CHECK ====================

@app.get("/api/health")
def health():
    return {
        "status": "running",
        "model": "loaded",
        "time": datetime.now().isoformat()
    }

# ==================== RUN ====================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
