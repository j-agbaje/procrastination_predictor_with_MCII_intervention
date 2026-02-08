"""
FastAPI Backend for Procrastination Prediction Platform
Author: Jeremiah Agbaje
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Procrastination Prediction API",
    description="AI-powered procrastination risk prediction with MCII interventions",
    version="1.0.0"
)

# CORS middleware (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================

class StudentLogin(BaseModel):
    email: str
    password: str

class PredictionRequest(BaseModel):
    student_id: int
    behavioral_data: Optional[dict] = None

class PredictionResponse(BaseModel):
    student_id: int
    risk_level: str  # "Low", "Medium", "High"
    risk_score: float
    timestamp: str
    explanation: str

class MCIIRequest(BaseModel):
    student_id: int
    message: str

class TaskCreate(BaseModel):
    title: str
    deadline: str
    description: Optional[str] = None

# ==================== MOCK DATA (Replace with DB later) ====================

MOCK_STUDENTS = {
    1: {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "risk_level": "High",
        "risk_score": 0.85
    },
    2: {
        "id": 2,
        "name": "Jane Smith",
        "email": "jane@example.com",
        "risk_level": "Low",
        "risk_score": 0.25
    }
}

# ==================== HELPER FUNCTIONS ====================

def load_ml_model():
    """Load the trained Bi-LSTM model (placeholder)"""
    # TODO: Implement actual model loading
    # model = tf.keras.models.load_model('models/saved_models/procrastination_bilstm_model.h5')
    return None

def predict_procrastination(student_id: int, features: dict) -> dict:
    """
    Make procrastination prediction
    TODO: Replace with actual model inference
    """
    # Mock prediction logic
    risk_score = np.random.uniform(0, 1)
    
    if risk_score < 0.33:
        risk_level = "Low"
    elif risk_score < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        "risk_level": risk_level,
        "risk_score": float(risk_score),
        "timestamp": datetime.now().isoformat()
    }

def generate_mcii_response(student_message: str, risk_level: str) -> str:
    """
    Generate MCII intervention using GPT API
    TODO: Implement OpenAI API call
    """
    # Mock MCII response
    if risk_level == "High":
        return """Let's use Mental Contrasting to address this. 

**Desired Future:** What academic goal would you most like to achieve this week?

**Present Obstacle:** What's the main thing preventing you from making progress right now?

Once you share these, I'll help you create an 'If-Then' plan to overcome the obstacle."""
    else:
        return "Great work maintaining consistent study habits! Keep it up. 💪"

# ==================== API ROUTES ====================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Procrastination Prediction API",
        "version": "1.0.0"
    }

@app.get("/api/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",  # TODO: Check actual DB
            "ml_model": "loaded",      # TODO: Check actual model
            "openai_api": "ready"      # TODO: Check API key
        }
    }

@app.post("/api/auth/login")
def login(credentials: StudentLogin):
    """
    Student/Admin login
    TODO: Implement proper authentication with JWT
    """
    # Mock authentication
    if credentials.email == "student@test.com" and credentials.password == "password":
        return {
            "access_token": "mock_token_12345",
            "token_type": "bearer",
            "user_id": 1,
            "role": "student"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    """
    Get procrastination risk prediction for a student
    """
    # Check if student exists (mock check)
    if request.student_id not in MOCK_STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Make prediction
    prediction = predict_procrastination(request.student_id, request.behavioral_data or {})
    
    # Generate explanation
    explanations = {
        "Low": "Your study patterns show consistency and early engagement with tasks. Keep up the great work!",
        "Medium": "Some irregularity detected in your study patterns. Consider establishing a more consistent schedule.",
        "High": "Late submission patterns and irregular study habits detected. Let's work on building better habits together."
    }
    
    return PredictionResponse(
        student_id=request.student_id,
        risk_level=prediction["risk_level"],
        risk_score=prediction["risk_score"],
        timestamp=prediction["timestamp"],
        explanation=explanations[prediction["risk_level"]]
    )

@app.post("/api/mcii/chat")
def mcii_chat(request: MCIIRequest):
    """
    MCII intervention chatbot endpoint
    """
    # Get student's current risk level
    student = MOCK_STUDENTS.get(request.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Generate MCII response
    response = generate_mcii_response(request.message, student["risk_level"])
    
    return {
        "student_id": request.student_id,
        "bot_response": response,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/students/{student_id}")
def get_student(student_id: int):
    """Get student profile"""
    if student_id not in MOCK_STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return MOCK_STUDENTS[student_id]

@app.post("/api/tasks")
def create_task(task: TaskCreate):
    """Create a new task for student"""
    return {
        "task_id": np.random.randint(1000, 9999),
        "title": task.title,
        "deadline": task.deadline,
        "description": task.description,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }

@app.get("/api/admin/students")
def list_students():
    """Admin: Get all students with risk levels"""
    return {
        "students": list(MOCK_STUDENTS.values()),
        "total": len(MOCK_STUDENTS)
    }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)