"""
ProActive - FastAPI Application
Server-side rendering with Jinja2 + session-based auth.
"""

from fastapi import FastAPI, Request, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.templating import Jinja2Templates # serve templates to the client
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware  
from starlette.exceptions import HTTPException as StarletteHTTPException

from sqlalchemy import create_engine, Column, Integer, String, or_, Float, DateTime, Enum, Date, JSON, Text, Boolean, DECIMAL, TIMESTAMP, func
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
import random
import uuid

from anthropic import Anthropic
from apscheduler.schedulers.background import BackgroundScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import uvicorn
from schemas import SignupRequest, LoginRequest, TaskCreate, TaskUpdate, ProfileUpdate, MCIIMessage, PredictionRequest
from dotenv import load_dotenv


load_dotenv()


# ── Directory configuration 
BASE_DIR    = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR  = BASE_DIR / "static"
MODEL_DIR   = BASE_DIR / "models" / "saved_models"

templates = Jinja2Templates(directory=TEMPLATE_DIR) # templates directory

# ── Database 
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

scheduler = BackgroundScheduler(timezone="UTC")


# ── ORM Models 
# Kept in sync with schema.sql — if you change the schema, update these too.

class Student(Base):
    __tablename__ = "Students"
    student_id         = Column(Integer, primary_key=True, autoincrement=True)
    email              = Column(String(255), unique=True, nullable=False)
    full_name         = Column(String(100), nullable=True)
    password_hash      = Column(String(255), nullable=False)
    enrollment_date    = Column(Date, nullable=False)
    current_risk_level = Column(Enum('low', 'medium', 'high'), default='low')
    prior_profile      = Column(Enum('early', 'mixed', 'lastminute'), default='mixed')
    # prior_profile drives cold-start synthetic bundle rows
    days_active        = Column(Integer, default=0)
    # days_active determines which model to use: <7 closed bundles → 3window, 7+ → 7window
    created_at         = Column(TIMESTAMP, default=datetime.now)
    phone              = Column(String(20), nullable=True)
    profile_pic        = Column(String(255), nullable=True)
    bio                = Column(Text, nullable=True)

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


# ── ML Model Loading 
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




# ── Utilities 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── Auth helpers 

def require_login(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=status.HTTP_302_FOUND, headers={"Location": "/login"})
    return user

def require_student(request: Request) -> dict:
    user = require_login(request)
    if user["role"] != "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student access only")
    return user

def require_admin(request: Request) -> dict:
    user = require_login(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access only")
    return user


# ── Bundle utilities and inference helper 


def create_initial_bundle(student_id: int, db: Session) -> Optional[WeeklyBundle]:
    """
    Create or return the current week's open WeeklyBundle row for a student.
    Ensures idempotency so repeated calls in the same week do not duplicate bundles.
    """
    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    week_number = today.isocalendar().week

    existing = (
        db.query(WeeklyBundle)
        .filter(
            WeeklyBundle.student_id == student_id,
            WeeklyBundle.week_number == week_number,
        )
        .first()
    )
    if existing:
        return existing

    bundle = WeeklyBundle(
        student_id=student_id,
        week_number=week_number,
        start_date=start_of_week,
        end_date=end_of_week,
        tasks_total=0,
        tasks_completed=0,
        tasks_late=0,
        completion_rate=0.0,
        submitted_late=0,
        is_closed=0,
    )

    try:
        db.add(bundle)
        db.commit()
        db.refresh(bundle)
        return bundle
    except Exception as exc:
        db.rollback()
        print(f"[bundles] Failed to create initial bundle for student {student_id}: {exc}")
        return None


def collate_weekly_bundles(db: Session) -> None:
    """
    Close the current open bundle for each student, compute snapshot metrics, and
    provision the next week's bundle. Intended for use by a Sunday night scheduler.
    """
    students = db.query(Student).all()

    for student in students:
        try:
            open_bundle = (
                db.query(WeeklyBundle)
                .filter(
                    WeeklyBundle.student_id == student.student_id,
                    WeeklyBundle.is_closed == 0,
                )
                .order_by(WeeklyBundle.week_number.desc())
                .first()
            )
            if not open_bundle:
                continue

            start_dt = datetime.combine(open_bundle.start_date, datetime.min.time())
            end_dt = datetime.combine(open_bundle.end_date, datetime.max.time())

            tasks = (
                db.query(Task)
                .filter(
                    Task.student_id == student.student_id,
                    Task.due_date >= start_dt,
                    Task.due_date <= end_dt,
                )
                .all()
            )

            tasks_total = len(tasks)
            tasks_completed = sum(1 for t in tasks if t.status == "completed")
            tasks_late = sum(
                1
                for t in tasks
                if t.status == "overdue"
                or (t.completed_at is not None and t.completed_at > t.due_date)
            )
            completion_rate = (tasks_completed / tasks_total) if tasks_total > 0 else 0.0
            submitted_late = 1 if tasks_late > 0 else 0

            open_bundle.tasks_total = tasks_total
            open_bundle.tasks_completed = tasks_completed
            open_bundle.tasks_late = tasks_late
            open_bundle.completion_rate = float(completion_rate)
            open_bundle.submitted_late = submitted_late
            open_bundle.is_closed = 1
            open_bundle.closed_at = datetime.now()

            next_monday = open_bundle.start_date + timedelta(days=7)
            next_sunday = next_monday + timedelta(days=6)
            next_week_number = next_monday.isocalendar().week

            existing_next = (
                db.query(WeeklyBundle)
                .filter(
                    WeeklyBundle.student_id == student.student_id,
                    WeeklyBundle.week_number == next_week_number,
                )
                .first()
            )
            if not existing_next:
                next_bundle = WeeklyBundle(
                    student_id=student.student_id,
                    week_number=next_week_number,
                    start_date=next_monday,
                    end_date=next_sunday,
                    tasks_total=0,
                    tasks_completed=0,
                    tasks_late=0,
                    completion_rate=0.0,
                    submitted_late=0,
                    is_closed=0,
                )
                db.add(next_bundle)

            db.commit()
            print(f"[bundles] Collated bundle {open_bundle.bundle_id} for student {student.student_id}")
        except Exception as exc:
            db.rollback()
            print(f"[bundles] Failed to collate bundles for student {student.student_id}: {exc}")


def assign_tasks_to_bundles(db: Session) -> None:
    """
    Attach tasks without a bundle_id to the correct WeeklyBundle based on due_date.
    Commits in small batches for efficiency and safety.
    """
    pending_tasks = (
        db.query(Task)
        .filter(Task.bundle_id.is_(None))
        .order_by(Task.due_date.asc())
        .all()
    )

    batch_size = 50
    counter = 0

    for task in pending_tasks:
        try:
            task_date = task.due_date.date()
            bundle = (
                db.query(WeeklyBundle)
                .filter(
                    WeeklyBundle.student_id == task.student_id,
                    WeeklyBundle.start_date <= task_date,
                    WeeklyBundle.end_date >= task_date,
                )
                .order_by(WeeklyBundle.week_number.desc())
                .first()
            )
            if bundle:
                task.bundle_id = bundle.bundle_id
                counter += 1

            if counter and counter % batch_size == 0:
                db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[bundles] Failed assigning task {task.task_id} to bundle: {exc}")

    if counter % batch_size != 0:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[bundles] Failed final commit while assigning tasks to bundles: {exc}")


# ── Inference helper 

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

    # ── 1. Count closed bundles to decide which model to use 
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

    # ── 2. Get the current (live) bundle 
    current_bundle = (
        db.query(WeeklyBundle)
        .filter(WeeklyBundle.student_id == student.student_id, WeeklyBundle.is_closed == 0)
        .order_by(WeeklyBundle.week_number.desc())
        .first()
    )

    if not current_bundle:
        return None  # no active bundle yet — student hasn't set up their week

    # ── 3. Compute live features for current bundle (Row 7 / Row 3) 
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

    # ── 4. Build feature sequence, prepending priors for gaps
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

    # ── 5. Scale 
    seq_array = np.array(sequence, dtype=np.float32)  # (window, 5)
    seq_scaled = scaler.transform(seq_array).reshape(1, window, 5)

    # ── 6. Inference 
    output = model.predict(seq_scaled, verbose=0)
    risk_score = float(output[0][0])  # sigmoid output — probability of late submission

    # ── 7. Map score to risk level 
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


def generate_mcii_tip(risk_level: str, confidence_score: float) -> str:
    """
    Generate a concise MCII (Mental Contrasting and Implementation Intentions) tip using Claude
    for a given procrastination risk level and confidence score.
    """
    normalized_risk = (risk_level or "").lower()
    safe_confidence = max(0.0, min(float(confidence_score or 0.0), 1.0))

    base_prompt = (
        f"Generate a concise MCII tip under 100 words for a student "
        f"with {normalized_risk or 'unknown'} procrastination risk "
        f"(confidence: {safe_confidence:.2f}). "
        "Use the Mental Contrasting and Implementation Intentions framework."
    )

    low_risk_variants = [
        base_prompt
        + " Focus on positive reinforcement, maintaining momentum, and acknowledging recent wins. "
        "Keep the tone warm and encouraging, under 80 words.",
        base_prompt
        + " Emphasize how staying consistent this week will protect their current low-risk status. "
        "Highlight a clear if-then plan for keeping their good habits, under 80 words.",
        base_prompt
        + " Celebrate that the upcoming weekly bundle deadline looks manageable and encourage one small "
        "implementation intention that locks in their current study rhythm, under 80 words.",
    ]

    medium_risk_variants = [
        base_prompt
        + " Format the response exactly as:\n"
        "Goal: [specific academic goal linked to this week's bundle deadline].\n"
        "Obstacle: [concrete obstacle based on typical procrastination patterns].\n"
        "Plan: If [specific trigger situation before the deadline], then I will [precise study action].\n"
        "Be specific and mention that the bundle deadline is approaching soon. Stay under 120 words.",
        base_prompt
        + " The response must follow this structure:\n"
        "Goal: [clear goal tied to tasks due by the end of the current week].\n"
        "Obstacle: [realistic internal or external barrier, like phone distraction or fatigue].\n"
        "Plan: If [time or context near the deadline], then I will [focused behavior that moves one task forward].\n"
        "Keep it concrete, deadline-aware, and under 120 words.",
        base_prompt
        + " Use MCII to help the student close the gap before this week's bundle deadline. "
        "Write:\nGoal: ...\nObstacle: ...\nPlan: If ... then I will ....\n"
        "Refer explicitly to the remaining days before the deadline and keep it under 120 words.",
        base_prompt
        + " Assume the student still has several tasks due by Sunday. "
        "Structure the tip as Goal / Obstacle / Plan, with the plan being an if-then action they can execute "
        "today or tomorrow before the bundle closes. Under 120 words.",
    ]

    high_risk_variants = [
        base_prompt
        + " Write in an urgent but supportive tone. "
        "Assume many tasks are still unfinished and the bundle deadline is very close. "
        "Include a clear if-then implementation intention that can be acted on immediately, and stay under 150 words.",
        base_prompt
        + " Emphasize that time is almost up for this week's bundle and several tasks remain. "
        "Direct the student to pick one high-impact task and create a sharp if-then plan to start within the next hour. "
        "Keep it structured, concrete, and under 150 words.",
        base_prompt
        + " Treat this as a high-urgency situation. "
        "Highlight the cost of not acting before the weekly deadline, then provide one specific if-then plan "
        "for tackling the most overdue or risky task. Under 150 words.",
        base_prompt
        + " Assume the student has been postponing work until the last minute. "
        "Be direct: mention the urgent deadline, the number of tasks likely remaining, and give a firm if-then rule "
        "they can follow tonight to reduce risk. Under 150 words.",
    ]

    if normalized_risk == "low":
        prompt = random.choice(low_risk_variants)
    elif normalized_risk == "medium":
        prompt = random.choice(medium_risk_variants)
    elif normalized_risk == "high":
        prompt = random.choice(high_risk_variants)
    else:
        prompt = base_prompt + " Provide a balanced, supportive MCII tip suitable for an unknown risk level."

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        if response and getattr(response, "content", None):
            first_block = response.content[0]
            text = getattr(first_block, "text", None) or getattr(first_block, "content", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
    except Exception:
        pass

    if normalized_risk == "low":
        return (
            "You are on a strong path this week. Picture how it will feel to submit everything on time, "
            "then commit: if I start to drift or scroll my phone during my planned study block, "
            "then I will pause, take a breath, and return to the next small step on my task list."
        )
    if normalized_risk == "medium":
        return (
            "Goal: Finish this week’s key tasks before the bundle deadline.\n"
            "Obstacle: I tend to delay starting when studying feels overwhelming.\n"
            "Plan: If it is the next available 30-minute window today, then I will open my planner, pick one task "
            "due soonest, and work on it without checking my phone until the timer ends."
        )
    return (
        "Goal: Submit as many remaining tasks as possible before this week’s deadline.\n"
        "Obstacle: I keep putting tasks off until it feels too late to start.\n"
        "Plan: If it is the next hour, then I will choose the single most urgent task, silence notifications, "
        "and work on it in a focused 25-minute block, followed by a 5-minute break."
    )


# ── FastAPI App 

app = FastAPI(title="ProActive")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SECRET_KEY", "dev-secret-change-in-production"),
    max_age=86400
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


    

# ── Page Routes 

@app.get("/", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request): # request: Request required by Jinja 2 template rendering
    # redirect to admin dashboard if user is admin else redirect to student dashboard
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


@app.post("/student/profile/update")
async def update_profile(
    request: Request,
    full_name: str = Form(...),
    bio: str = Form(""),
    profile_pic: Optional[UploadFile] = File(None),
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db),
):
    """
    Handle profile updates for the logged-in student, including name, bio, and
    optional profile picture upload with validation and safe database commit.
    """
    student = (
        db.query(Student)
        .filter(Student.student_id == current_user["user_id"])
        .first()
    )

    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student record not found for profile update",
        )

    name_clean = (full_name or "").strip()
    bio_clean = (bio or "").strip()
    error_message: Optional[str] = None

    if len(name_clean) < 2 or len(name_clean) > 100:
        error_message = "Full name must be between 2 and 100 characters."
    elif len(bio_clean) > 500:
        error_message = "Bio must be at most 500 characters."

    image_path: Optional[str] = None
    new_file_written = False

    if not error_message and profile_pic and profile_pic.filename:
        allowed_types = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        if profile_pic.content_type not in allowed_types:
            error_message = "Profile picture must be a JPG, PNG, or WEBP image."
        else:
            data = await profile_pic.read()
            if len(data) > 2 * 1024 * 1024:
                error_message = "Profile picture must be under 2MB."
            else:
                ext = allowed_types[profile_pic.content_type]
                upload_dir = STATIC_DIR / "uploads" / "profile_pics"
                upload_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{student.student_id}_{uuid.uuid4().hex[:8]}{ext}"
                file_path = upload_dir / filename
                with open(file_path, "wb") as f:
                    f.write(data)
                image_path = f"/static/uploads/profile_pics/{filename}"
                new_file_written = True

    if error_message:
        completed_count = db.query(Task).filter(
            Task.student_id == current_user["user_id"],
            Task.status == "completed",
        ).count()
        latest_prediction = (
            db.query(Prediction)
            .filter(Prediction.student_id == current_user["user_id"])
            .order_by(Prediction.prediction_date.desc())
            .first()
        )
        return templates.TemplateResponse(
            "student_profile.html",
            {
                "request": request,
                "current_user": current_user,
                "student": student,
                "completed_count": completed_count,
                "prediction": latest_prediction,
                "error": error_message,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    old_profile_pic = student.profile_pic
    student.full_name = name_clean
    student.bio = bio_clean
    if image_path:
        student.profile_pic = image_path

    try:
        db.commit()
    except Exception:
        db.rollback()
        completed_count = db.query(Task).filter(
            Task.student_id == current_user["user_id"],
            Task.status == "completed",
        ).count()
        latest_prediction = (
            db.query(Prediction)
            .filter(Prediction.student_id == current_user["user_id"])
            .order_by(Prediction.prediction_date.desc())
            .first()
        )
        return templates.TemplateResponse(
            "student_profile.html",
            {
                "request": request,
                "current_user": current_user,
                "student": student,
                "completed_count": completed_count,
                "prediction": latest_prediction,
                "error": "Could not update profile. Please try again.",
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if image_path and old_profile_pic:
        try:
            old_name = os.path.basename(old_profile_pic)
            old_path = STATIC_DIR / "uploads" / "profile_pics" / old_name
            if old_path.exists():
                old_path.unlink()
        except Exception:
            pass

    return RedirectResponse(
        url="/student/profile",
        status_code=status.HTTP_303_SEE_OTHER,
    )


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
    page: int = 1,
    risk_filter: Optional[str] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    per_page = 20

    # ── Stats (all dynamic from DB) ──
    total_students  = db.query(Student).count()
    high_risk_count = db.query(Student).filter(Student.current_risk_level == "high").count()

    students_with_interventions = db.query(MCIIIntervention.student_id).distinct().count()
    mcii_engagement = round((students_with_interventions / total_students * 100), 1) if total_students > 0 else 0

    total_tasks     = db.query(Task).count()
    completed_tasks = db.query(Task).filter(Task.status == "completed").count()
    avg_progress    = round((completed_tasks / total_tasks * 100), 1) if total_tasks > 0 else 0

    # ── Student query with filters ──
    query = db.query(Student)

    if risk_filter:
        query = query.filter(Student.current_risk_level == risk_filter)

    if search:
        query = query.filter(
    or_(
        Student.email.ilike(f"%{search}%"),
        Student.full_name.ilike(f"%{search}%")
    )
)

    total_filtered = query.count()
    total_pages    = max(1, (total_filtered + per_page - 1) // per_page)
    page           = max(1, min(page, total_pages))
    offset         = (page - 1) * per_page

    students = query.order_by(Student.created_at.desc()).offset(offset).limit(per_page).all()

    # ── Attach latest prediction to each student ──
    students_with_predictions = []
    for s in students:
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
            "total_students":   total_students,
            "high_risk_alerts": high_risk_count,
            "mcii_engagement":  mcii_engagement,
            "avg_progress":     avg_progress,
        },
        "students":      students_with_predictions,
        "page":          page,
        "total_pages":   total_pages,
        "total_filtered": total_filtered,
        "risk_filter":   risk_filter,
        "search":        search,
    })


@app.get("/admin/students/{student_id}", response_class=HTMLResponse)
async def admin_student_detail(
    student_id: int,
    request: Request,
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Render a detailed view for a single student including predictions, tasks, bundles,
    latest MCII intervention, and high-level task statistics for admin review.
    """
    student = db.query(Student).filter(Student.student_id == student_id).first()

    if not student:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": status.HTTP_404_NOT_FOUND,
                "title": status.HTTP_404_NOT_FOUND,
                "detail": "Student not found",
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )

    predictions = (
        db.query(Prediction)
        .filter(Prediction.student_id == student_id)
        .order_by(Prediction.prediction_date.desc())
        .limit(14)
        .all()
    )

    tasks = (
        db.query(Task)
        .filter(Task.student_id == student_id)
        .order_by(Task.due_date.desc())
        .limit(10)
        .all()
    )

    bundles = (
        db.query(WeeklyBundle)
        .filter(WeeklyBundle.student_id == student_id)
        .order_by(WeeklyBundle.week_number.desc())
        .all()
    )

    latest_intervention = (
        db.query(MCIIIntervention)
        .filter(MCIIIntervention.student_id == student_id)
        .order_by(MCIIIntervention.delivery_time.desc())
        .first()
    )

    total_tasks = db.query(Task).filter(Task.student_id == student_id).count()
    completed_tasks = (
        db.query(Task)
        .filter(Task.student_id == student_id, Task.status == "completed")
        .count()
    )
    overdue_tasks = (
        db.query(Task)
        .filter(Task.student_id == student_id, Task.status == "overdue")
        .count()
    )

    task_stats = {
        "total": total_tasks,
        "completed": completed_tasks,
        "overdue": overdue_tasks,
    }

    latest_prediction = predictions[0] if predictions else None

    return templates.TemplateResponse(
        "admin_student_detail.html",
        {
            "request": request,
            "current_user": current_user,
            "student": student,
            "predictions": predictions,
            "tasks": tasks,
            "bundles": bundles,
            "latest_intervention": latest_intervention,
            "task_stats": task_stats,
            "latest_prediction": latest_prediction,
        },
    )


# ── Auth Form Handlers 

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
        return RedirectResponse(url="/student/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    admin = db.query(Admin).filter(Admin.email == email).first()
    if admin and admin.password_hash == pw_hash:
        request.session["user"] = {
            "user_id": admin.admin_id,
            "email":   admin.email,
            "role":    "admin"
        }
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid email or password", "form_email": email},
        status_code=status.HTTP_400_BAD_REQUEST
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
            status_code=status.HTTP_400_BAD_REQUEST
        )

    student = Student(
        email=email,
        full_name=name,
        password_hash=hash_password(password),
        enrollment_date=date.today(),
        current_risk_level="low",
        prior_profile=prior_profile,
        days_active=0,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    create_initial_bundle(student.student_id, db)

    request.session["user"] = {
        "user_id": student.student_id,
        "email":   student.email,
        "role":    "student"
    }
    return RedirectResponse(url="/student/dashboard", status_code=status.HTTP_303_SEE_OTHER)


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
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


# ── Task Form Handlers 

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
    return RedirectResponse(url="/student/tasks", status_code=status.HTTP_303_SEE_OTHER)


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
    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)


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
    return RedirectResponse(url="/student/tasks", status_code=status.HTTP_303_SEE_OTHER)


# ── Prediction API 
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
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only generate predictions for yourself")

    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    result = compute_prediction(student, db)
    if not result:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not available or no active bundle")

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


@app.get("/student/mcii/tip")
async def get_mcii_tip(
    request: Request,
    current_user: dict = Depends(require_student),
    db: Session = Depends(get_db),
):
    """
    Return a personalized MCII tip for the logged-in student based on today's prediction.
    Reuses any intervention already delivered today, otherwise generates, stores, and returns a new one.
    """
    try:
        today = date.today()

        existing = (
            db.query(MCIIIntervention)
            .filter(
                MCIIIntervention.student_id == current_user["user_id"],
                func.date(MCIIIntervention.delivery_time) == today,
            )
            .order_by(MCIIIntervention.delivery_time.desc())
            .first()
        )

        if existing:
            prediction = (
                db.query(Prediction)
                .filter(Prediction.prediction_id == existing.prediction_id)
                .first()
            )
            if prediction:
                return {
                    "tip": existing.prompt_text,
                    "risk_level": prediction.risk_level,
                    "confidence": float(prediction.confidence_score),
                }
            return {
                "tip": existing.prompt_text,
                "risk_level": "unknown",
                "confidence": 0.0,
            }

        latest_prediction = (
            db.query(Prediction)
            .filter(Prediction.student_id == current_user["user_id"])
            .order_by(Prediction.prediction_date.desc())
            .first()
        )

        if not latest_prediction:
            return {
                "tip": (
                    "We are still building your profile. "
                    "Add tasks to your weekly bundle and check back tomorrow for a personalized strategy."
                ),
                "risk_level": "unknown",
                "confidence": 0.0,
            }

        tip_text = generate_mcii_tip(
            risk_level=latest_prediction.risk_level,
            confidence_score=float(latest_prediction.confidence_score),
        )

        try:
            intervention = MCIIIntervention(
                prediction_id=latest_prediction.prediction_id,
                student_id=current_user["user_id"],
                prompt_text=tip_text,
            )
            db.add(intervention)
            db.commit()
        except Exception:
            db.rollback()

        return {
            "tip": tip_text,
            "risk_level": latest_prediction.risk_level,
            "confidence": float(latest_prediction.confidence_score),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate MCII tip at this time.",
        )


# ── Health Check 

@app.get("/api/health")
def health_check():
    return {
        "status":        "ok",
        "model_3window": "loaded" if model_3window else "not_loaded",
        "model_7window": "loaded" if model_7window else "not_loaded",
        "timestamp":     datetime.now().isoformat()
    }



# for HTTP exceptions
@app.exception_handler(StarletteHTTPException)
async def general_http_exception_handler(request: Request, exception: StarletteHTTPException):
    message = (
        exception.detail
        if exception.detail
        else "An error occurred. Please check your request and try again."
    )
    if exception.status_code in (401, 302):
        return RedirectResponse(url="/login", status_code=303)
    if request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=exception.status_code,
            content={"detail": message},
        )
    return templates.TemplateResponse("error.html", {
        "request": request,
        "status_code": exception.status_code,
        "title": exception.status_code,
        "detail": message
    }, status_code=exception.status_code)

# for unexpected crashes or errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "status_code": 500,
        "detail": "Something went wrong"
    }, status_code=500)

from fastapi.exceptions import RequestValidationError

# catches bad form data submitted
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exception):
    if request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exception.errors()},
        )
    return templates.TemplateResponse("error.html", {
        "request": request,
        "status_code": 422,
        "detail": "Invalid form data submitted"
    }, status_code=422)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")