from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum, Date, JSON, Text, Boolean, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import os
import secrets
import hashlib
import jwt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
import uvicorn

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "backend/models/saved_models"

# JWT configuration for secure authentication
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Database connection setup
DATABASE_URL = "mysql+pymysql://root:astroball197310@localhost/procrastination_db"
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
    echo=False           # Set to True for SQL debugging
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# Database Models

class Student(Base):
    """Student model representing registered students in the system"""
    __tablename__ = "Students"
    student_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    enrollment_date = Column(Date, nullable=False)
    current_risk_level = Column(Enum('low', 'medium', 'high'), default='low')
    created_at = Column(DateTime, default=datetime.now)

class Admin(Base):
    """Admin model for platform administrators"""
    __tablename__ = "Admins"
    admin_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    department = Column(String(100))
    access_level = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)

class Prediction(Base):
    """Model for storing AI-generated procrastination risk predictions"""
    __tablename__ = "Predictions"
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, nullable=False)
    prediction_date = Column(Date, nullable=False)
    risk_level = Column(Enum('low', 'medium', 'high'), nullable=False)
    confidence_score = Column(DECIMAL(3,2), nullable=False)
    attention_weights_json = Column(JSON)

class Task(Base):
    """Model for student tasks and assignments"""
    __tablename__ = "Tasks"
    task_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    due_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)
    status = Column(Enum('pending', 'in_progress', 'completed', 'overdue'), default='pending')

class Survey(Base):
    """Model for student survey responses"""
    __tablename__ = "Surveys"
    survey_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, unique=True, nullable=False)
    responses_json = Column(JSON, nullable=False)
    completion_date = Column(DateTime, default=datetime.now)

class BehavioralLog(Base):
    """Model for tracking student behavioral patterns and session data"""
    __tablename__ = "BehavioralLogs"
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, nullable=False)
    login_time = Column(DateTime, nullable=False)
    logout_time = Column(DateTime)
    pages_visited = Column(Integer, default=0)
    session_duration = Column(Integer)  # in seconds

class MCIIIntervention(Base):
    """Model for Mental Contrasting and Implementation Intentions interventions"""
    __tablename__ = "MCIIInterventions"
    intervention_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=False)
    student_id = Column(Integer, nullable=False)
    prompt_text = Column(Text, nullable=False)
    delivery_time = Column(DateTime, default=datetime.now)
    user_response = Column(Text)
    was_helpful = Column(Boolean)

# Pydantic models for request/response validation

class LoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str = Field(..., min_length=6)

class SignupRequest(BaseModel):
    """Request model for new student registration"""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Ensure password meets minimum security requirements"""
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class TaskCreate(BaseModel):
    """Request model for creating a new task"""
    student_id: int
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: datetime
    status: str = Field(default='pending')

class TaskUpdate(BaseModel):
    """Request model for updating an existing task"""
    title: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    status: Optional[str] = None
    completed_at: Optional[datetime] = None

class PredictionRequest(BaseModel):
    """Request model for generating a procrastination risk prediction"""
    student_id: int
    behavioral_data: Dict[str, float] = Field(default_factory=dict)

class TokenData(BaseModel):
    """Model for JWT token payload data"""
    user_id: int
    email: str
    role: str

# Utility Functions

def get_db():
    """Database session dependency that ensures proper session cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT access token for authenticated sessions"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token, returning the payload if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Dependency to get the current authenticated user from the request"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    return payload

# Load ML model and preprocessing tools
# try:
#     model = tf.keras.models.load_model(MODEL_DIR / "procrastination_bilstm.h5", compile=False)
#     with open(MODEL_DIR / "scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
#         label_encoder = pickle.load(f)
#     print("✓ ML models loaded successfully")
# except Exception as e:
#     print(f"⚠ Warning: Could not load ML models: {e}")
#     model = None
#     scaler = None
#     label_encoder = None


# Load ML model
try:
    # 1. Define the fix for the version mismatch
    custom_objects = {'Orthogonal': tf.keras.initializers.Orthogonal}
    
    # 2. Load the model using the fix
    model = tf.keras.models.load_model(
        MODEL_DIR / "procrastination_bilstm.h5", 
        custom_objects=custom_objects,
        compile=False
    )
    
    # 3. Load the scalers
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    print("✓ ML models loaded successfully (with version patch)")

except Exception as e:
    print(f"⚠ Warning: Could not load ML models: {e}")
    model = None

# FastAPI application initialization
app = FastAPI(
    title="ProActive - Procrastination Prediction Platform",
    description="AI-powered platform for predicting and preventing student procrastination",
    version="1.0.0"
)

# Mount the frontend directory to serve static assets like JS and CSS
# This allows requests to /js/auth.js to find the file in frontend/js/auth.js
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
# app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
# app.mount("/images", StaticFiles(directory=FRONTEND_DIR / "images"), name="images")

# CORS middleware configuration to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML Page Routes

@app.get("/")
def serve_login_page():
    """Serve the login page as the application entry point"""
    return FileResponse(FRONTEND_DIR / "login.html")

@app.get("/signup")
def serve_signup_page():
    """Serve the student registration page"""
    return FileResponse(FRONTEND_DIR / "signup.html")

@app.get("/student/dashboard")
def serve_student_dashboard():
    """Serve the main student dashboard"""
    return FileResponse(FRONTEND_DIR / "student_dashboard.html")

@app.get("/student/tasks")
def serve_student_tasks():
    """Serve the student tasks management page"""
    return FileResponse(FRONTEND_DIR / "tasks.html")

@app.get("/student/profile")
def serve_student_profile():
    """Serve the student profile page"""
    return FileResponse(FRONTEND_DIR / "student_profile.html")

@app.get("/student/mcii")
def serve_mcii_chat():
    """Serve the MCII intervention chat interface"""
    return FileResponse(FRONTEND_DIR / "mcii_chat.html")

@app.get("/admin/dashboard")
def serve_admin_dashboard():
    """Serve the administrator dashboard"""
    return FileResponse(FRONTEND_DIR / "admin_dashboard.html")

# Authentication API Routes

@app.post("/api/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a user (student or admin) and return a JWT token
    
    Returns:
        - access_token: JWT for subsequent authenticated requests
        - user_id: Unique identifier for the user
        - role: 'student' or 'admin'
        - email: User's email address
    """
    try:
        password_hash = hash_password(data.password)
        
        # Check if user is a student
        student = db.query(Student).filter(Student.email == data.email).first()
        if student and student.password_hash == password_hash:
            token_data = {
                "user_id": student.student_id,
                "email": student.email,
                "role": "student"
            }
            access_token = create_access_token(token_data)
            
            # Create behavioral log entry for this login session
            behavior_log = BehavioralLog(
                student_id=student.student_id,
                login_time=datetime.now()
            )
            db.add(behavior_log)
            db.commit()
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user_id": student.student_id,
                "role": "student",
                "email": student.email
            }
        
        # Check if user is an admin
        admin = db.query(Admin).filter(Admin.email == data.email).first()
        if admin and admin.password_hash == password_hash:
            token_data = {
                "user_id": admin.admin_id,
                "email": admin.email,
                "role": "admin"
            }
            access_token = create_access_token(token_data)
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user_id": admin.admin_id,
                "role": "admin",
                "email": admin.email
            }
        
        # Invalid credentials
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/api/auth/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    """
    Register a new student account
    
    Returns:
        - access_token: JWT for immediate login
        - user_id: New student's unique identifier
        - role: Always 'student'
        - email: Student's email address
    """
    try:
        # Check if email already exists
        existing_student = db.query(Student).filter(Student.email == data.email).first()
        if existing_student:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists"
            )
        
        # Create new student account
        student = Student(
            email=data.email,
            password_hash=hash_password(data.password),
            enrollment_date=date.today(),
            current_risk_level='low'
        )
        db.add(student)
        db.commit()
        db.refresh(student)
        
        # Generate access token for immediate login
        token_data = {
            "user_id": student.student_id,
            "email": student.email,
            "role": "student"
        }
        access_token = create_access_token(token_data)
        
        # Create initial behavioral log
        behavior_log = BehavioralLog(
            student_id=student.student_id,
            login_time=datetime.now()
        )
        db.add(behavior_log)
        db.commit()
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": student.student_id,
            "role": "student",
            "email": student.email,
            "message": "Account created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )

@app.post("/api/auth/logout")
def logout(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Log out the current user and update their session data
    """
    try:
        if current_user['role'] == 'student':
            # Find the most recent unclosed behavioral log
            recent_log = db.query(BehavioralLog).filter(
                BehavioralLog.student_id == current_user['user_id'],
                BehavioralLog.logout_time.is_(None)
            ).order_by(BehavioralLog.login_time.desc()).first()
            
            if recent_log:
                logout_time = datetime.now()
                recent_log.logout_time = logout_time
                
                # Calculate session duration in seconds
                session_duration = (logout_time - recent_log.login_time).total_seconds()
                recent_log.session_duration = int(session_duration)
                
                db.commit()
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

# Task Management API Routes

@app.get("/api/tasks/{student_id}")
def get_tasks(
    student_id: int,
    status_filter: Optional[str] = None,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve all tasks for a specific student, with optional status filtering
    
    Query Parameters:
        - status_filter: Optional filter for task status (pending, in_progress, completed, overdue)
    """
    try:
        # Ensure user can only access their own tasks (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own tasks"
            )
        
        query = db.query(Task).filter(Task.student_id == student_id)
        
        if status_filter:
            query = query.filter(Task.status == status_filter)
        
        tasks = query.order_by(Task.due_date.asc()).all()
        
        # Convert to dictionaries for JSON response
        tasks_data = []
        for task in tasks:
            tasks_data.append({
                "task_id": task.task_id,
                "student_id": task.student_id,
                "title": task.title,
                "description": task.description,
                "due_date": task.due_date.isoformat(),
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "status": task.status
            })
        
        return {"tasks": tasks_data, "count": len(tasks_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tasks: {str(e)}"
        )

@app.post("/api/tasks")
def create_task(
    task_data: TaskCreate,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new task for a student
    """
    try:
        # Ensure user can only create tasks for themselves (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != task_data.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only create tasks for yourself"
            )
        
        task = Task(
            student_id=task_data.student_id,
            title=task_data.title,
            description=task_data.description,
            due_date=task_data.due_date,
            status=task_data.status
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        return {
            "success": True,
            "message": "Task created successfully",
            "task_id": task.task_id,
            "task": {
                "task_id": task.task_id,
                "title": task.title,
                "due_date": task.due_date.isoformat(),
                "status": task.status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )

@app.put("/api/tasks/{task_id}")
def update_task(
    task_id: int,
    task_data: TaskUpdate,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing task
    """
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        # Ensure user can only update their own tasks (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != task.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update your own tasks"
            )
        
        # Update only provided fields
        if task_data.title is not None:
            task.title = task_data.title
        if task_data.description is not None:
            task.description = task_data.description
        if task_data.due_date is not None:
            task.due_date = task_data.due_date
        if task_data.status is not None:
            task.status = task_data.status
            # If status is completed, set completion timestamp
            if task_data.status == 'completed' and not task.completed_at:
                task.completed_at = datetime.now()
        if task_data.completed_at is not None:
            task.completed_at = task_data.completed_at
        
        db.commit()
        db.refresh(task)
        
        return {
            "success": True,
            "message": "Task updated successfully",
            "task": {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update task: {str(e)}"
        )

@app.delete("/api/tasks/{task_id}")
def delete_task(
    task_id: int,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a task
    """
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        # Ensure user can only delete their own tasks (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != task.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete your own tasks"
            )
        
        db.delete(task)
        db.commit()
        
        return {
            "success": True,
            "message": "Task deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete task: {str(e)}"
        )

# Student Profile API Routes

@app.get("/api/students/{student_id}")
def get_student_profile(
    student_id: int,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve student profile information
    """
    try:
        # Ensure user can only access their own profile (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own profile"
            )
        
        student = db.query(Student).filter(Student.student_id == student_id).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )
        
        # Get latest prediction
        latest_prediction = db.query(Prediction).filter(
            Prediction.student_id == student_id
        ).order_by(Prediction.prediction_date.desc()).first()
        
        # Get task statistics
        total_tasks = db.query(Task).filter(Task.student_id == student_id).count()
        completed_tasks = db.query(Task).filter(
            Task.student_id == student_id,
            Task.status == 'completed'
        ).count()
        
        return {
            "student_id": student.student_id,
            "email": student.email,
            "enrollment_date": student.enrollment_date.isoformat(),
            "current_risk_level": student.current_risk_level,
            "created_at": student.created_at.isoformat(),
            "latest_prediction": {
                "risk_level": latest_prediction.risk_level,
                "confidence_score": float(latest_prediction.confidence_score),
                "prediction_date": latest_prediction.prediction_date.isoformat()
            } if latest_prediction else None,
            "task_stats": {
                "total": total_tasks,
                "completed": completed_tasks,
                "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )

# Prediction API Routes

@app.post("/api/predict")
def generate_prediction(
    req: PredictionRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a procrastination risk prediction for a student using the ML model
    
    Behavioral data expected:
        - late_rate: Proportion of tasks completed late
        - irregularity: Variability in work patterns
        - last_min_ratio: Proportion of work done at last minute
        - avg_gap: Average time gap between task creation and completion
    """
    try:
        # Ensure user can only generate predictions for themselves (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != req.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only generate predictions for yourself"
            )
        
        if not model or not scaler or not label_encoder:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model is not available"
            )
        
        # Extract and prepare features
        features = np.array([[
            req.behavioral_data.get("late_rate", 0.5),
            req.behavioral_data.get("irregularity", 0.3),
            req.behavioral_data.get("last_min_ratio", 0.4),
            req.behavioral_data.get("avg_gap", 2.0)
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Generate prediction
        prediction = model.predict(features_scaled, verbose=0)
        risk_idx = int(np.argmax(prediction[0]))
        risk_level = label_encoder.inverse_transform([risk_idx])[0].lower()
        risk_score = float(prediction[0][risk_idx])
        
        # Save prediction to database
        pred = Prediction(
            student_id=req.student_id,
            prediction_date=date.today(),
            risk_level=risk_level,
            confidence_score=round(risk_score, 2)
        )
        db.add(pred)
        
        # Update student's current risk level
        student = db.query(Student).filter(Student.student_id == req.student_id).first()
        if student:
            student.current_risk_level = risk_level
        
        db.commit()
        db.refresh(pred)
        
        return {
            "prediction_id": pred.prediction_id,
            "student_id": req.student_id,
            "risk_level": risk_level,
            "confidence_score": risk_score,
            "prediction_date": pred.prediction_date.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/api/predictions/{student_id}")
def get_predictions(
    student_id: int,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve prediction history for a student
    """
    try:
        # Ensure user can only access their own predictions (unless admin)
        if current_user['role'] == 'student' and current_user['user_id'] != student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own predictions"
            )
        
        predictions = db.query(Prediction).filter(
            Prediction.student_id == student_id
        ).order_by(Prediction.prediction_date.desc()).limit(limit).all()
        
        predictions_data = []
        for pred in predictions:
            predictions_data.append({
                "prediction_id": pred.prediction_id,
                "prediction_date": pred.prediction_date.isoformat(),
                "risk_level": pred.risk_level,
                "confidence_score": float(pred.confidence_score)
            })
        
        return {"predictions": predictions_data, "count": len(predictions_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )

# Admin API Routes

@app.get("/api/admin/students")
def get_all_students(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve all students (admin only)
    """
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        students = db.query(Student).all()
        
        students_data = []
        for student in students:
            # Get latest prediction for each student
            latest_pred = db.query(Prediction).filter(
                Prediction.student_id == student.student_id
            ).order_by(Prediction.prediction_date.desc()).first()
            
            students_data.append({
                "student_id": student.student_id,
                "email": student.email,
                "enrollment_date": student.enrollment_date.isoformat(),
                "current_risk_level": student.current_risk_level,
                "latest_prediction": {
                    "risk_level": latest_pred.risk_level,
                    "confidence_score": float(latest_pred.confidence_score),
                    "date": latest_pred.prediction_date.isoformat()
                } if latest_pred else None
            })
        
        return {"students": students_data, "total": len(students_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve students: {str(e)}"
        )

@app.get("/api/admin/dashboard/stats")
def get_dashboard_stats(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve platform statistics for admin dashboard
    """
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Total students
        total_students = db.query(Student).count()
        
        # High risk students
        high_risk_count = db.query(Student).filter(
            Student.current_risk_level == 'high'
        ).count()
        
        # MCII engagement rate (students with interventions / total students)
        students_with_interventions = db.query(MCIIIntervention.student_id).distinct().count()
        mcii_engagement = (students_with_interventions / total_students * 100) if total_students > 0 else 0
        
        # Average completion rate
        all_tasks = db.query(Task).count()
        completed_tasks = db.query(Task).filter(Task.status == 'completed').count()
        avg_completion = (completed_tasks / all_tasks * 100) if all_tasks > 0 else 0
        
        return {
            "total_students": total_students,
            "high_risk_alerts": high_risk_count,
            "mcii_engagement": round(mcii_engagement, 1),
            "avg_progress": round(avg_completion, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard stats: {str(e)}"
        )

# Health Check Route

@app.get("/api/health")
def health_check():
    """
    Health check endpoint to verify API and ML model status
    """
    model_status = "loaded" if model is not None else "not_loaded"
    db_status = "connected"
    
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
    except:
        db_status = "disconnected"
    
    return {
        "status": "ok",
        "model": model_status,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )