"""
ProActive - FastAPI Application
Server-side rendering with Jinja2 + session-based auth.
"""

from fastapi import FastAPI, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum, Date, JSON, Text, Boolean, DECIMAL, TIMESTAMP
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import os
import secrets
import hashlib
import json
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import uvicorn


# ── Directory configuration ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR  = BASE_DIR / "static"
MODEL_DIR   = BASE_DIR / "models" / "saved_models"

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = "mysql+pymysql://root:astroball197310@localhost/ProActive_db"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────
# Kept in sync with schema.sql — if you change the schema, update these too.

class Student(Base):
    __tablename__ = "Students"
    student_id         = Column(Integer, primary_key=True, autoincrement=True)
    email              = Column(String(255), unique=True, nullable=False)
    password_hash      = Column(String(255), nullable=False)
    enrollment_date    = Column(Date, nullable=False)
    current_risk_level = Column(Enum('low', 'medium', 'high'), default='low')
    prior_profile      = Column(Enum('early', 'mixed', 'lastminute'), default='mixed')
    # prior_profile drives cold-start synthetic bundle rows
    days_active        = Column(Integer, default=0)
    # days_active determines which model to use: <7 closed bundles → 3window, 7+ → 7window
    created_at         = Column(TIMESTAMP, default=datetime.now)

class Admin(Base):
    __tablename__ = "Admins"
    admin_id     = Column(Integer, primary_key=True, autoincrement=True)
    email        = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    department   = Column(String(100))
    access_level = Column(Integer, default=1)
    created_at   = Column(TIMESTAMP, default=datetime.now)

class WeeklyBundle(Base):
    __tablename__ = "WeeklyBundles"
    bundle_id       = Column(Integer, primary_key=True, autoincrement=True)
    student_id      = Column(Integer, nullable=False)
    week_number     = Column(Integer, nullable=False)
    start_date      = Column(Date, nullable=False)       # Monday
    end_date        = Column(Date, nullable=False)       # Sunday (deadline)
    tasks_total     = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    tasks_late      = Column(Integer, default=0)
    completion_rate = Column(DECIMAL(4, 3), default=0.000)
    submitted_late  = Column(TINYINT, default=0)        # 1 if completion_rate < 1.0 at close
    is_closed       = Column(TINYINT, default=0)        # 1 after Sunday snapshot
    closed_at       = Column(TIMESTAMP, nullable=True)
    created_at      = Column(TIMESTAMP, default=datetime.now)

class Task(Base):
    __tablename__ = "Tasks"
    task_id      = Column(Integer, primary_key=True, autoincrement=True)
    student_id   = Column(Integer, nullable=False)
    bundle_id    = Column(Integer, nullable=True)       # NULL until assigned to a bundle
    title        = Column(String(200), nullable=False)
    description  = Column(Text)
    due_date     = Column(DateTime, nullable=False)
    created_at   = Column(TIMESTAMP, default=datetime.now)
    completed_at = Column(TIMESTAMP, nullable=True)
    status       = Column(Enum('pending', 'in_progress', 'completed', 'overdue'), default='pending')

class Prediction(Base):
    __tablename__ = "Predictions"
    prediction_id          = Column(Integer, primary_key=True, autoincrement=True)
    student_id             = Column(Integer, nullable=False)
    bundle_id              = Column(Integer, nullable=True)
    prediction_date        = Column(Date, nullable=False)
    model_used             = Column(Enum('3window', '7window'), nullable=False)
    risk_level             = Column(Enum('low', 'medium', 'high'), nullable=False)
    confidence_score       = Column(DECIMAL(3, 2), nullable=False)
    attention_weights_json = Column(JSON, nullable=True)
    features_json          = Column(JSON, nullable=True)

class Survey(Base):
    __tablename__ = "Surveys"
    survey_id       = Column(Integer, primary_key=True, autoincrement=True)
    student_id      = Column(Integer, unique=True, nullable=False)
    responses_json  = Column(JSON, nullable=False)
    completion_date = Column(TIMESTAMP, default=datetime.now)

class BehavioralLog(Base):
    __tablename__ = "BehavioralLogs"
    log_id           = Column(Integer, primary_key=True, autoincrement=True)
    student_id       = Column(Integer, nullable=False)
    login_time       = Column(TIMESTAMP, nullable=False)
    logout_time      = Column(TIMESTAMP, nullable=True)
    pages_visited    = Column(Integer, default=0)
    session_duration = Column(Integer, nullable=True)

class MCIIIntervention(Base):
    __tablename__ = "MCIIInterventions"
    intervention_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id   = Column(Integer, nullable=False)
    student_id      = Column(Integer, nullable=False)
    prompt_text     = Column(Text, nullable=False)
    delivery_time   = Column(TIMESTAMP, default=datetime.now)
    user_response   = Column(Text, nullable=True)
    was_helpful     = Column(Boolean, nullable=True)


# ── ML Model Loading ──────────────────────────────────────────────────────────
# Both models are loaded at startup. Which one runs depends on closed bundle count.
# No label_encoder — risk thresholds are applied directly to the sigmoid output.

model_3window = None
model_7window = None
scaler_3window = None
scaler_7window = None
prior_profiles: Dict[str, Any] = {}
feature_config: Dict[str, Any] = {}

try:
    custom_objects = {'Orthogonal': tf.keras.initializers.Orthogonal}

    model_3window = tf.keras.models.load_model(
        MODEL_DIR / "bilstm_3window.h5",
        custom_objects=custom_objects,
        compile=False
    )
    model_7window = tf.keras.models.load_model(
        MODEL_DIR / "bilstm_7window.h5",
        custom_objects=custom_objects,
        compile=False
    )
    with open(MODEL_DIR / "scaler_3window.pkl", "rb") as f:
        scaler_3window = pickle.load(f)
    with open(MODEL_DIR / "scaler_7window.pkl", "rb") as f:
        scaler_7window = pickle.load(f)
    with open(MODEL_DIR / "prior_profiles.json", "r") as f:
        prior_profiles = json.load(f)
    with open(MODEL_DIR / "feature_config.json", "r") as f:
        feature_config = json.load(f)

    print("✓ All ML artifacts loaded")

except Exception as e:
    print(f"Could not load ML artifacts: {e}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def require_login(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return user

def require_student(request: Request) -> dict:
    user = require_login(request)
    if user["role"] != "student":
        raise HTTPException(status_code=403, detail="Student access only")
    return user

def require_admin(request: Request) -> dict:
    user = require_login(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access only")
    return user


# ── Inference helper ──────────────────────────────────────────────────────────

def _bundle_to_features(bundle: WeeklyBundle, today: date) -> list[float]:
    """
    Convert a closed WeeklyBundle row into the 5-feature vector.
    For a closed bundle: days_until_deadline is negative if tasks ran late.
    """
    days_until_deadline  = (bundle.end_date - today).days
    # For closed bundles we use the end_date vs today; negative = overdue
    return [
        float(days_until_deadline),
        0.0,   # days_since_last_sub — filled in by caller with proper gap
        1.0,   # submitted_today — 1 for closed bundles (snapshot day counts)
        float(bundle.completion_rate),
        0.0,   # overdue_count — filled in by caller as running total
    ]

def compute_prediction(student: Student, db: Session) -> Optional[dict]:
    """
    Full inference pipeline as described in the pipeline doc.
    Returns a dict ready to store in Predictions, or None if models not loaded.
    """
    if not model_3window:
        return None

    today = date.today()

    # ── 1. Count closed bundles to decide which model to use ──
    closed_bundles = (
        db.query(WeeklyBundle)
        .filter(WeeklyBundle.student_id == student.student_id, WeeklyBundle.is_closed == 1)
        .order_by(WeeklyBundle.week_number.asc())
        .all()
    )
    num_closed = len(closed_bundles)

    if num_closed >= 7 and model_7window:
        model      = model_7window
        scaler     = scaler_7window
        window     = 7
        model_name = "7window"
    else:
        model      = model_3window
        scaler     = scaler_3window
        window     = 3
        model_name = "3window"

    # ── 2. Get the current (live) bundle ──
    current_bundle = (
        db.query(WeeklyBundle)
        .filter(WeeklyBundle.student_id == student.student_id, WeeklyBundle.is_closed == 0)
        .order_by(WeeklyBundle.week_number.desc())
        .first()
    )

    if not current_bundle:
        return None  # no active bundle yet — student hasn't set up their week

    # ── 3. Compute live features for current bundle (Row 7 / Row 3) ──
    tasks_in_bundle = (
        db.query(Task)
        .filter(Task.bundle_id == current_bundle.bundle_id)
        .all()
    )
    tasks_total     = len(tasks_in_bundle)
    tasks_completed = sum(1 for t in tasks_in_bundle if t.status == 'completed')
    completion_rate = (tasks_completed / tasks_total) if tasks_total > 0 else 0.0

    overdue_count = sum(1 for b in closed_bundles if b.submitted_late == 1)

    submitted_today = 1 if any(
        t.completed_at and t.completed_at.date() == today
        for t in tasks_in_bundle
    ) else 0

    days_until_deadline = (current_bundle.end_date - today).days

    # days_since_last_sub: gap from previous bundle's last completed task
    if closed_bundles:
        prev_bundle = closed_bundles[-1]
        prev_tasks = (
            db.query(Task)
            .filter(Task.bundle_id == prev_bundle.bundle_id, Task.status == 'completed')
            .order_by(Task.completed_at.desc())
            .first()
        )
        if prev_tasks and prev_tasks.completed_at:
            days_since_last_sub = max(0, (today - prev_tasks.completed_at.date()).days)
        else:
            days_since_last_sub = 7  # default: assume last week
    else:
        days_since_last_sub = max(0, (today - student.enrollment_date).days)

    live_features = [
        float(days_until_deadline),
        float(days_since_last_sub),
        float(submitted_today),
        float(completion_rate),
        float(overdue_count),
    ]

    # ── 4. Build feature sequence, prepending priors for gaps ──
    # We need (window - 1) historical rows + 1 live row = window total
    real_history_needed = window - 1
    real_rows = []

    running_overdue = 0
    for i, bundle in enumerate(closed_bundles[-(real_history_needed):]):
        if bundle.submitted_late:
            running_overdue += 1
        prev_last_completed = None
        if i > 0:
            pb = closed_bundles[-(real_history_needed) + i - 1]
            prev_t = (
                db.query(Task)
                .filter(Task.bundle_id == pb.bundle_id, Task.status == 'completed')
                .order_by(Task.completed_at.desc())
                .first()
            )
            if prev_t and prev_t.completed_at:
                prev_last_completed = prev_t.completed_at.date()

        gap = (bundle.end_date - prev_last_completed).days if prev_last_completed else 7
        real_rows.append([
            float((bundle.end_date - bundle.end_date).days),  # 0 — submitted on deadline
            float(max(0, gap)),
            1.0,
            float(bundle.completion_rate),
            float(running_overdue),
        ])

    # Fill remaining slots with prior profile synthetic rows
    rows_needed = real_history_needed - len(real_rows)
    prior_key   = student.prior_profile if student.prior_profile else 'mixed'
    prior_rows  = prior_profiles.get(prior_key, [])[:rows_needed] if rows_needed > 0 else []

    sequence = prior_rows + real_rows + [live_features]

    if len(sequence) != window:
        # Safety check — should not happen with correct prior_profiles.json
        return None

    # ── 5. Scale ──
    seq_array = np.array(sequence, dtype=np.float32)  # (window, 5)
    seq_scaled = scaler.transform(seq_array).reshape(1, window, 5)

    # ── 6. Inference ──
    output = model.predict(seq_scaled, verbose=0)
    risk_score = float(output[0][0])  # sigmoid output — probability of late submission

    # ── 7. Map score to risk level ──
    if risk_score < 0.40:
        risk_level = "low"
    elif risk_score < 0.65:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "risk_level":        risk_level,
        "confidence_score":  round(risk_score, 2),
        "model_used":        model_name,
        "bundle_id":         current_bundle.bundle_id,
        "features_json":     {
            "days_until_deadline":  days_until_deadline,
            "days_since_last_sub":  days_since_last_sub,
            "submitted_today":      submitted_today,
            "completion_rate":      round(completion_rate, 3),
            "overdue_count":        overdue_count,
        }
    }


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="ProActive")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SECRET_KEY", "dev-secret-change-in-production"),
    max_age=86400
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Page Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        role = request.session["user"]["role"]
        return RedirectResponse(url="/admin/dashboard" if role == "admin" else "/student/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/student/dashboard", response_class=HTMLResponse)
async def student_dashboard(
    request: Request,
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.student_id == current_user["user_id"]).first()

    today = date.today()
    tasks = (
        db.query(Task)
        .filter(Task.student_id == current_user["user_id"])
        .filter(Task.due_date >= datetime.combine(today, datetime.min.time()))
        .order_by(Task.due_date.asc())
        .all()
    )

    latest_prediction = (
        db.query(Prediction)
        .filter(Prediction.student_id == current_user["user_id"])
        .order_by(Prediction.prediction_date.desc())
        .first()
    )

    return templates.TemplateResponse("student_dashboard.html", {
        "request":          request,
        "current_user":     current_user,
        "student":          student,
        "tasks":            tasks,
        "prediction":       latest_prediction,
    })


@app.get("/student/tasks", response_class=HTMLResponse)
async def tasks_page(
    request: Request,
    filter_status: Optional[str] = None,
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db)
):
    query = db.query(Task).filter(Task.student_id == current_user["user_id"])
    if filter_status:
        query = query.filter(Task.status == filter_status)
    tasks = query.order_by(Task.due_date.asc()).all()

    return templates.TemplateResponse("tasks.html", {
        "request":       request,
        "current_user":  current_user,
        "tasks":         tasks,
        "filter_status": filter_status,
    })


@app.get("/student/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.student_id == current_user["user_id"]).first()
    completed_count = db.query(Task).filter(
        Task.student_id == current_user["user_id"],
        Task.status == "completed"
    ).count()
    latest_prediction = (
        db.query(Prediction)
        .filter(Prediction.student_id == current_user["user_id"])
        .order_by(Prediction.prediction_date.desc())
        .first()
    )

    return templates.TemplateResponse("student_profile.html", {
        "request":          request,
        "current_user":     current_user,
        "student":          student,
        "completed_count":  completed_count,
        "prediction":       latest_prediction,
    })


@app.get("/student/mcii", response_class=HTMLResponse)
async def mcii_page(
    request: Request,
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db)
):
    # Load recent MCII interventions for this student to pre-populate chat history
    interventions = (
        db.query(MCIIIntervention)
        .filter(MCIIIntervention.student_id == current_user["user_id"])
        .order_by(MCIIIntervention.delivery_time.desc())
        .limit(10)
        .all()
    )
    return templates.TemplateResponse("mcii_chat.html", {
        "request":       request,
        "current_user":  current_user,
        "interventions": interventions,
    })


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    total_students   = db.query(Student).count()
    high_risk_count  = db.query(Student).filter(Student.current_risk_level == "high").count()
    recent_students  = (
        db.query(Student)
        .order_by(Student.created_at.desc())
        .limit(10)
        .all()
    )
    # Attach latest prediction to each student for the table
    students_with_predictions = []
    for s in recent_students:
        pred = (
            db.query(Prediction)
            .filter(Prediction.student_id == s.student_id)
            .order_by(Prediction.prediction_date.desc())
            .first()
        )
        students_with_predictions.append({"student": s, "prediction": pred})

    return templates.TemplateResponse("admin_dashboard.html", {
        "request":      request,
        "current_user": current_user,
        "stats": {
            "total_students":    total_students,
            "high_risk_alerts":  high_risk_count,
        },
        "students": students_with_predictions,
    })


# ── Auth Form Handlers ────────────────────────────────────────────────────────

@app.post("/auth/login")
async def handle_login(
    request: Request,
    email:    str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    pw_hash = hash_password(password)

    student = db.query(Student).filter(Student.email == email).first()
    if student and student.password_hash == pw_hash:
        request.session["user"] = {
            "user_id": student.student_id,
            "email":   student.email,
            "role":    "student"
        }
        db.add(BehavioralLog(student_id=student.student_id, login_time=datetime.now()))
        db.commit()
        return RedirectResponse(url="/student/dashboard", status_code=303)

    admin = db.query(Admin).filter(Admin.email == email).first()
    if admin and admin.password_hash == pw_hash:
        request.session["user"] = {
            "user_id": admin.admin_id,
            "email":   admin.email,
            "role":    "admin"
        }
        return RedirectResponse(url="/admin/dashboard", status_code=303)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid email or password", "form_email": email},
        status_code=400
    )


@app.post("/auth/signup")
async def handle_signup(
    request:       Request,
    name:          str = Form(...),
    email:         str = Form(...),
    password:      str = Form(...),
    prior_profile: str = Form(default="mixed"),  # from study habits question on signup form
    db: Session = Depends(get_db)
):
    if db.query(Student).filter(Student.email == email).first():
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "An account with this email already exists"},
            status_code=400
        )

    student = Student(
        email=email,
        password_hash=hash_password(password),
        enrollment_date=date.today(),
        current_risk_level="low",
        prior_profile=prior_profile,
        days_active=0,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    request.session["user"] = {
        "user_id": student.student_id,
        "email":   student.email,
        "role":    "student"
    }
    return RedirectResponse(url="/student/dashboard", status_code=303)


@app.post("/auth/logout")
async def handle_logout(
    request: Request,
    db: Session = Depends(get_db)
):
    user = request.session.get("user")
    if user and user["role"] == "student":
        recent_log = (
            db.query(BehavioralLog)
            .filter(BehavioralLog.student_id == user["user_id"], BehavioralLog.logout_time.is_(None))
            .order_by(BehavioralLog.login_time.desc())
            .first()
        )
        if recent_log:
            recent_log.logout_time = datetime.now()
            recent_log.session_duration = int(
                (recent_log.logout_time - recent_log.login_time).total_seconds()
            )
            db.commit()

    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ── Task Form Handlers ────────────────────────────────────────────────────────

@app.post("/student/tasks/create")
async def create_task(
    request:     Request,
    title:       str      = Form(...),
    due_date:    datetime = Form(...),
    description: str      = Form(default=""),
    current_user: dict    = Depends(require_student),
    db: Session           = Depends(get_db)
):
    # Find the active (open) bundle for this student to assign the task to
    active_bundle = (
        db.query(WeeklyBundle)
        .filter(WeeklyBundle.student_id == current_user["user_id"], WeeklyBundle.is_closed == 0)
        .first()
    )
    task = Task(
        student_id  = current_user["user_id"],
        bundle_id   = active_bundle.bundle_id if active_bundle else None,
        title       = title,
        description = description,
        due_date    = due_date,
        status      = "pending"
    )
    db.add(task)
    db.commit()
    return RedirectResponse(url="/student/tasks", status_code=303)


@app.post("/student/tasks/{task_id}/toggle")
async def toggle_task(
    task_id:     int,
    request:     Request,
    current_user: dict  = Depends(require_student),
    db: Session         = Depends(get_db)
):
    task = db.query(Task).filter(
        Task.task_id   == task_id,
        Task.student_id == current_user["user_id"]
    ).first()

    if task:
        if task.status == "completed":
            task.status       = "pending"
            task.completed_at = None
        else:
            task.status       = "completed"
            task.completed_at = datetime.now()
        db.commit()

    referer = request.headers.get("referer", "/student/dashboard")
    return RedirectResponse(url=referer, status_code=303)


@app.post("/student/tasks/{task_id}/delete")
async def delete_task(
    task_id:     int,
    request:     Request,
    current_user: dict  = Depends(require_student),
    db: Session         = Depends(get_db)
):
    task = db.query(Task).filter(
        Task.task_id    == task_id,
        Task.student_id == current_user["user_id"]
    ).first()
    if task:
        db.delete(task)
        db.commit()
    return RedirectResponse(url="/student/tasks", status_code=303)


# ── Prediction API ────────────────────────────────────────────────────────────
# Kept as a JSON endpoint because it's compute-heavy and called on-demand.

@app.post("/api/predict/{student_id}")
async def generate_prediction(
    student_id:   int,
    request:      Request,
    current_user: dict    = Depends(require_student),
    db: Session           = Depends(get_db)
):
    """
    Runs the full inference pipeline for a student.
    Which model is used depends on how many closed bundles exist, not days_active directly.
    """
    if current_user["user_id"] != student_id:
        raise HTTPException(status_code=403, detail="You can only generate predictions for yourself")

    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    result = compute_prediction(student, db)
    if not result:
        raise HTTPException(status_code=503, detail="Model not available or no active bundle")

    pred = Prediction(
        student_id        = student_id,
        bundle_id         = result["bundle_id"],
        prediction_date   = date.today(),
        model_used        = result["model_used"],
        risk_level        = result["risk_level"],
        confidence_score  = result["confidence_score"],
        features_json     = result["features_json"],
    )
    db.add(pred)

    student.current_risk_level = result["risk_level"]
    db.commit()
    db.refresh(pred)

    return {
        "prediction_id":    pred.prediction_id,
        "risk_level":       result["risk_level"],
        "confidence_score": result["confidence_score"],
        "model_used":       result["model_used"],
        "features_used":    result["features_json"],
    }


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status":        "ok",
        "model_3window": "loaded" if model_3window else "not_loaded",
        "model_7window": "loaded" if model_7window else "not_loaded",
        "timestamp":     datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")