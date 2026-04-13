# ProActive — AI-Powered Procrastination Prediction and Intervention Platform

ProActive is a full-stack web application that estimates early procrastination risk from weekly behavioural data using a BiLSTM with Bahdanau attention, and supports students through an MCII-style coaching chatbot (Anthropic Claude). It includes separate student and admin experiences: students track tasks and receive predictions and coaching; admins manage cohorts, invite codes, assignments, and risk monitoring.

---

## For reviewers and moderators

Use this section to **review the running product** without installing anything locally.

1. **Open the deployed application**  
   **Live site:** [https://web-production-185b1.up.railway.app](https://web-production-185b1.up.railway.app) (currently deactivated after capstone defence)

2. **What to try on the live site**
   - **Admin:** Sign in at `/login` with the default admin account (see [Default login credentials](#default-login-credentials)). Explore the admin dashboard, cohort overview, student detail views, and task assignment.
   - **Student:** Register at `/signup` using invite code `PROACTIVE1` (ties the student to the default admin cohort). Complete the weekly behavioural bundles and tasks as prompted, then open the student dashboard to view the risk trend and MCII chat.
   - **Scheduler/predictions:** After students have data, an admin can use **Run Scheduler** (from the admin UI) to trigger the batch prediction job so risk levels populate as designed.

3. **Evidence and formal evaluation**  
   Functional results, performance checks, deployment steps, ML evaluation, and discussion are documented in **[TESTING.md](TESTING.md)** — use that document for testing methodology and benchmarks rather than this README.

4. **Optional: verify the codebase locally**  
   If you need to run from source, follow [Installation and setup](#installation-and-setup) and [Run the application](#run-the-application). You will need Python 3.12, MySQL, a valid `ANTHROPIC_API_KEY`, and the ML artifacts under `models/saved_models/` (see [Machine learning artifacts](#machine-learning-artifacts)).

5. **Demo video (optional)**  
   [Google Drive — project demo](https://drive.google.com/file/d/18NolRuGXxa4hUc4FyRwRaje1omYAxHOf/view?usp=sharing)

---

## Live deployment

| | |
| --- | --- |
| **Production URL** | [https://web-production-185b1.up.railway.app](https://web-production-185b1.up.railway.app) |
| **Hosting** | Railway |
| **Testing & evaluation report** | [TESTING.md](TESTING.md) |

---

## Table of contents

- [For reviewers and moderators](#for-reviewers-and-moderators)
- [Live deployment](#live-deployment)
- [Project overview](#project-overview)
- [Tech stack](#tech-stack)
- [Machine learning artifacts](#machine-learning-artifacts)
- [Prerequisites](#prerequisites)
- [Installation and setup](#installation-and-setup)
- [Environment variables](#environment-variables)
- [Database setup](#database-setup)
- [Run the application](#run-the-application)
- [Default login credentials](#default-login-credentials)
- [Project structure](#project-structure)
- [Key dependencies](#key-dependencies)
- [Notebooks and training pipeline](#notebooks-and-training-pipeline)
- [Automated tests (pytest)](#automated-tests-pytest)
- [Load testing script](#load-testing-script)
- [Deployment (Railway)](#deployment-railway)

---

## Project overview

### Core features

- **Daily AI risk predictions** (low / medium / high) from weekly behavioural bundles using a 7-window BiLSTM; **cold-start** support via a 3-window model and **prior profiles** for new students.
- **MCII-oriented chatbot** with rolling conversation context (48-hour window) and live student state (risk, tasks, profile).
- **Admin cohort workflow:** invite codes, assigning tasks to the cohort, monitoring risk and per-student detail.
- **Student dashboard:** 14-day risk trend chart, tasks, profile, and rotating MCII tips.
- **Scheduled jobs:** APScheduler-driven batch work aligned with prediction generation (see admin “Run Scheduler” for manual triggering in development).

### Tech stack

| Layer | Technology |
| --- | --- |
| Backend | FastAPI (Python 3.12) |
| Database | MySQL (SQLAlchemy 2.x) |
| ML | TensorFlow / Keras — BiLSTM + Bahdanau attention |
| AI chat | Anthropic Claude |
| Frontend | HTML, Jinja2, Tailwind CSS (CDN), Chart.js where used |
| Scheduler | APScheduler |
| Deployment | Railway (`Procfile`, `runtime.txt`) |

### Model performance (reported)

| Model | Accuracy | F1 | AUC-ROC |
| --- | --- | --- | --- |
| BiLSTM 7-window (main) | 89.37% | 0.8874 | 0.9430 |
| BiLSTM 3-window (cold start) | 86.03% | 0.8569 | 0.9043 |
| SVM baseline | 68.16% | 0.6911 | 0.8231 |

Literature context: Memon et al. (2020) report an ANN at 83.5%; Mohammad & Gummadi (2025) provide additional benchmarks referenced in project materials.

---

## Machine learning artifacts

At startup, `main.py` loads weights, scalers, and JSON config from `models/saved_models/`:

- `bilstm_7window.h5`, `bilstm_3window.h5`
- `scaler_7window.pkl`, `scaler_3window.pkl`
- `prior_profiles.json`, `feature_config.json`

If `.h5` / `.pkl` files are missing, the app may start but ML inference will not work until those files are present. Generate or export them using the notebooks in `ml_notebooks/` (see [Notebooks and training pipeline](#notebooks-and-training-pipeline)), or use artifacts provided alongside the repository if your copy omits large binaries.

---

## Prerequisites

- **Python 3.12** (matches `runtime.txt` for Railway)
- **MySQL 8.x** (or compatible; deployed stack has used MySQL 9.x per project docs)
- **pip** (or compatible installer)
- **Anthropic API key** for the MCII chat feature (`ANTHROPIC_API_KEY`)

---

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/j-agbaje/procrastination_predictor_with_MCII_intervention.git
cd procrastination_predictor_with_MCII_intervention
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installing `requirements.txt` pulls in the web stack (FastAPI, uvicorn, SQLAlchemy, PyMySQL, Anthropic client), TensorFlow/Keras, APScheduler, Jinja2, scientific stack (NumPy, pandas, scikit-learn), Jupyter tooling used by the training notebooks, and pytest for automated tests. Expect a large download on first install (TensorFlow and notebook dependencies).

### 4. Configure environment variables

Create a `.env` file in the **project root** (same directory as `main.py`). See [Environment variables](#environment-variables).

### 5. Create the database and apply schema

See [Database setup](#database-setup).

### 6. Ensure ML artifacts are present

See [Machine learning artifacts](#machine-learning-artifacts).

### 7. Run the application

See [Run the application](#run-the-application).

---

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | SQLAlchemy URL, e.g. `mysql+pymysql://USER:PASSWORD@HOST:3306/DATABASE` |
| `SECRET_KEY` | Strongly recommended | Session signing secret (use a long random value in production) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for the coaching chatbot |

Example `.env` (adjust credentials and host):

```env
SECRET_KEY=your_long_random_secret_key
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/proactivedb
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Railway note:** If the platform injects `mysql://`, the application normalizes it to `mysql+pymysql://` in `database.py`. Details appear in [TESTING.md](TESTING.md).

---

## Database setup

1. Create a database in MySQL:

```sql
CREATE DATABASE proactivedb;
```

2. Load the schema from the repository (path is `database/schema.sql`, not the project root):

```bash
mysql -u root -p proactivedb < database/schema.sql
```

3. **Default admin (development / demo):** insert an admin row if your workflow requires the documented demo account (password hashing in the app uses SHA-256; align with how `main.py` stores passwords):

```sql
INSERT INTO Admins (email, password_hash, department, invite_code)
VALUES ('admin@proactive.com', SHA2('admin123', 256), 'Computer Science', 'PROACTIVE1');
```

If your schema column names differ, adjust to match `models.py` / `database/schema.sql`.

---

## Run the application

**Development (auto-reload):**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

**Production-style (fixed port):**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Railway uses the `Procfile` command `uvicorn main:app --host 0.0.0.0 --port $PORT`.

**First predictions:** Log in as admin, ensure students have registered and submitted bundles/tasks as needed, then use **Run Scheduler** in the admin UI to generate predictions where that workflow applies.

---

## Default login credentials

| Role | Email | Password |
| --- | --- | --- |
| Admin | admin@proactive.com | admin123 |

Students self-register at `/signup`. Use invite code **`PROACTIVE1`** to join the default admin cohort (when that admin and code exist in the database).

**Security:** Change default credentials and `SECRET_KEY` before any real deployment or public exposure.

---

## Project structure

```text
procrastination_predictor_with_MCII_intervention/
├── main.py                 # FastAPI app: routes, auth, ML inference, scheduler hooks
├── database.py             # Engine, session, DATABASE_URL handling
├── models.py               # SQLAlchemy ORM models
├── schemas.py              # Pydantic request/response schemas
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version pin (Railway)
├── Procfile                # web: uvicorn on $PORT
├── README.md               # This file
├── TESTING.md              # Testing report, deployment verification, ML evaluation
├── database/
│   └── schema.sql          # MySQL DDL
├── templates/              # Jinja2 pages (login, signup, dashboards, chat, admin, errors)
├── static/
│   └── css/                # Application styles
├── media/                  # Runtime uploads (e.g. profile images); created as needed
├── models/
│   └── saved_models/       # BiLSTM .h5, scalers .pkl, prior_profiles.json, feature_config.json
├── ml_notebooks/           # Training / experimentation Jupyter notebooks
├── tests/
│   ├── test_proactive.py   # Pytest suite (hashing, risk thresholds, helpers)
│   └── screenshots/        # Evidence assets referenced from TESTING.md
└── load_test.js            # k6-style load script example (points at production URL)
```

---

## Key dependencies

The following are representative of what `requirements.txt` installs; the file is authoritative for versions.

| Area | Packages |
| --- | --- |
| Web | `fastapi`, `uvicorn`, `starlette`, `jinja2`, `python-multipart`, `itsdangerous` |
| Data & ORM | `sqlalchemy`, `pymysql`, `pydantic`, `pydantic-settings` |
| ML | `tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `h5py` |
| AI | `anthropic`, `httpx` |
| Jobs | `apscheduler` |
| Security / auth helpers | `passlib`, `argon2-cffi`, `PyJWT`, `python-jose` (as pinned) |
| Dev / notebooks | `jupyter`, `jupyterlab`, `ipykernel`, `matplotlib`, `seaborn` |
| Tests | `pytest`, `pytest-asyncio` |

---

## Notebooks and training pipeline

- `ml_notebooks/procrastination-prediction-capstone-notebook.ipynb` — main capstone workflow for procrastination prediction.
- `ml_notebooks/hyperparameter-tuning-procrastination-prediction.ipynb` — hyperparameter exploration.

Use these to reproduce or update models and export artifacts into `models/saved_models/` in the shapes expected by `main.py`.

---

## Automated tests (pytest)

A small pytest suite lives in `tests/test_proactive.py` (risk thresholds, password hashing, invite-code helper behaviour). Run from the project root:

```bash
pytest tests/test_proactive.py -v
```

For full system-level evaluation, see **[TESTING.md](TESTING.md)**.

---

## Load testing script

`load_test.js` is configured against the production base URL for HTTP checks (e.g. login page). Run it with your preferred k6 or compatible runner if you need scripted load against the deployed instance.

---

## Deployment (Railway)

The application is deployed on **Railway** using `Procfile` and `runtime.txt`. Environment variables mirror [Environment variables](#environment-variables). Step-by-step provisioning, service wiring, and post-deploy checks are documented in **[TESTING.md](TESTING.md)** (deployment section).

**Live URL:** [https://web-production-185b1.up.railway.app](https://web-production-185b1.up.railway.app)
