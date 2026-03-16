# ProActive — AI-Powered Procrastination Prediction and Intervention Platform

## Live Deployment

https://web-production-185b1.up.railway.app

## Demo Video

https://drive.google.com/file/d/18NolRuGXxa4hUc4FyRwRaje1omYAxHOf/view?usp=sharing

## Testing and Evaluation

See [TESTING.md](TESTING.md) for the full testing report, including functional testing results, performance benchmarks, cross-browser testing, data variation testing, ML model evaluation, analysis, discussion, and recommendations.

## Project Overview

ProActive is a full-stack web application that predicts student procrastination using a Bidirectional LSTM neural network with Bahdanau attention, and delivers MCII (Mental Contrasting with Implementation Intentions) interventions through an AI coaching chatbot powered by Anthropic Claude. It supports both individual students managing their own productivity and institutional use where an admin monitors a cohort of students.

### Core Features

- Daily AI risk predictions (low / medium / high) based on weekly behavioural bundles
- Prior profiles cold-start solution for new students with no behavioural history
- MCII chatbot with 48-hour conversation memory and live student context
- Admin cohort system with invite codes, task assignment, and risk monitoring
- Student dashboard with 14-day risk trend graph and daily rotating MCII tips
- Task management with due dates, task types, and admin-assigned tasks

### Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.12) |
| Database | MySQL |
| ML | TensorFlow / Keras — BiLSTM + Bahdanau Attention |
| AI chatbot | Anthropic Claude (claude-haiku-4-5-20251001) |
| Frontend | Jinja2 + Tailwind CSS |
| Scheduler | APScheduler |
| Deployment | Railway |

### ML Model Performance

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| BiLSTM 7-window (main) | 88.74% | 0.8874 | 0.9428 |
| BiLSTM 3-window (cold start) | 86.03% | 0.8569 | 0.9043 |
| SVM baseline | 68.16% | 0.6911 | 0.8231 |

Benchmarks from Memon et al. (2020): ANN 83.5%, XGBoost 87.0%.

---

## Installation and Setup

### Prerequisites

- Python 3.12
- MySQL 8.0 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/proactive.git
cd proactive
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create environment variables

Create a `.env` file in the project root:

```
SECRET_KEY=your_long_random_secret_key
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/proactivedb
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 5 — Set up the database

Create the database in MySQL:

```sql
CREATE DATABASE proactivedb;
```

Then run the schema:

```bash
mysql -u root -p proactivedb < schema.sql
```

Insert a default admin account:

```sql
INSERT INTO Admins (email, password_hash, department, invite_code)
VALUES ('admin@proactive.com', SHA2('admin123', 256), 'Computer Science', 'PROACTIVE1');
```

### Step 6 — Run the application

```bash
uvicorn main:app --reload
```

The application runs at http://localhost:8000.

### Step 7 — Seed initial data

Log in as the admin at http://localhost:8000/login and click Run Scheduler to generate the first round of predictions for any registered students.

---

## Default Login Credentials

| Role | Email | Password |
|---|---|---|
| Admin | admin@proactive.com | admin123 |

Students self-register at `/signup`. Use invite code `PROACTIVE1` to join the admin's cohort.

---

## Project Structure

```
proactive/
├── main.py                  # All routes, ML inference, scheduler
├── database.py              # SQLAlchemy engine and session
├── models.py                # ORM models
├── schemas.py               # Pydantic schemas
├── requirements.txt
├── Procfile                 # Railway deployment command
├── runtime.txt              # Python version for Railway
├── schema.sql               # Database schema
├── TESTING.md               # Full testing and evaluation report
├── templates/               # Jinja2 HTML templates
├── static/                  # CSS and static assets
├── media/                   # Uploaded profile pictures
├── models/saved_models/     # BiLSTM model files and scalers
│   ├── bilstm_7window.h5
│   ├── bilstm_3window.h5
│   ├── scaler_7window.pkl
│   ├── scaler_3window.pkl
│   ├── prior_profiles.json
│   └── feature_config.json
├── ml_notebooks/            # Jupyter notebooks for model training
└── tests/
    └── screenshots/         # All testing evidence screenshots
```

---

## Deployment

The application is deployed on Railway. See [TESTING.md](TESTING.md) section 3 for the full step-by-step deployment plan, environment configuration, and deployment verification results.

**Live URL:** https://web-production-185b1.up.railway.app
