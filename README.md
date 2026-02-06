# Procrastination Risk Prediction Platform
## Transfer Learning with Bi-LSTM and MCII-Based Interventions

**Author:** Jeremiah Agbaje  
**Supervisor:** Bernard Lamptey  
**Institution:** African Leadership University  
**Project Type:** BSc Software Engineering Capstone

---

## 📋 Project Description

This platform uses transfer learning to predict procrastination risk in online students, combining a Bi-LSTM neural network pre-trained on the Open University Learning Analytics Dataset (OULAD) with MCII-based interventions delivered through an AI-powered web interface.

**Key Features:**
- 🧠 Bi-LSTM model with attention mechanism
- 📊 Real-time procrastination risk prediction (Low/Medium/High)
- 💬 GPT-4 powered MCII intervention chatbot
- 📈 Student dashboard with task tracking
- 👨‍💼 Admin monitoring interface
- 🔄 Transfer learning approach for limited local data

---

## 🔗 Links

- **GitHub Repository:** [https://github.com/yourusername/procrastination-prediction](https://github.com/yourusername/procrastination-prediction)
- **Live Demo:** [Coming soon - will be deployed on Render]
- **Video Demo:** [Link to 5-10 min demo video]

---

## 🏗️ Project Structure

```
procrastination-prediction/
├── README.md
├── requirements.txt
├── .gitignore
│
├── ml_notebooks/
│   ├── oulad_procrastination_analysis.ipynb    # OULAD pre-training
│   ├── local_data_finetuning.ipynb             # Transfer learning
│   └── model_evaluation.ipynb                   # Performance metrics
│
├── backend/
│   ├── main.py                                  # FastAPI application
│   ├── models/
│   │   ├── bilstm_model.py                     # Model architecture
│   │   ├── prediction_service.py               # Inference logic
│   │   └── saved_models/
│   │       ├── procrastination_bilstm_model.h5
│   │       ├── scaler.pkl
│   │       └── label_encoder.pkl
│   ├── routes/
│   │   ├── auth.py                             # Authentication
│   │   ├── predictions.py                      # Prediction endpoints
│   │   ├── students.py                         # Student management
│   │   └── interventions.py                    # MCII chatbot
│   ├── database/
│   │   ├── db.py                               # Database connection
│   │   └── models.py                           # SQLAlchemy models
│   └── config.py                               # Configuration
│
├── frontend/
│   ├── index.html                              # Landing page
│   ├── student_dashboard.html                  # Student interface
│   ├── admin_dashboard.html                    # Admin interface
│   ├── css/
│   │   └── styles.css                          # Tailwind/custom styles
│   ├── js/
│   │   ├── api.js                              # API calls
│   │   ├── dashboard.js                        # Dashboard logic
│   │   └── charts.js                           # Chart rendering
│   └── assets/
│       └── images/
│
├── data/
│   ├── oulad/                                  # OULAD dataset (download)
│   ├── survey_responses.csv                    # Local survey data
│   └── processed/                              # Preprocessed data
│
├── docs/
│   ├── architecture_diagram.png
│   ├── deployment_plan.md
│   └── user_guide.md
│
└── tests/
    ├── test_api.py
    └── test_model.py
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- MySQL 8.0
- Node.js (optional, for frontend build tools)
- Google Colab account (for model training with GPU)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/procrastination-prediction.git
cd procrastination-prediction
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download OULAD Dataset
1. Visit: https://analyse.kmi.open.ac.uk/open_dataset
2. Download all CSV files
3. Place in `data/oulad/` directory

Alternative (automated):
```bash
cd data/oulad
wget https://analyse.kmi.open.ac.uk/open_dataset/download -O oulad.zip
unzip oulad.zip
```

### 5. Train ML Model (Google Colab)
1. Open `ml_notebooks/oulad_procrastination_analysis.ipynb` in Google Colab
2. Upload OULAD data to Google Drive
3. Run all cells to:
   - Create procrastination labels via K-Means clustering
   - Train Bi-LSTM model
   - Save model artifacts
4. Download trained model files to `backend/models/saved_models/`

### 6. Set Up Database
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE procrastination_db;
exit;

# Run migrations (create tables)
python backend/database/init_db.py
```

### 7. Configure Environment Variables
Create `.env` file in root directory:
```env
# Database
DATABASE_URL=mysql://root:password@localhost:3306/procrastination_db

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# JWT Secret
SECRET_KEY=your_secret_key_here

# Environment
ENVIRONMENT=development
```

### 8. Run Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

### 9. Serve Frontend
```bash
# Simple Python server
cd frontend
python -m http.server 3000
```

Frontend will be available at: `http://localhost:3000`

Alternative (use Live Server extension in VS Code)

### 10. Test API Endpoints
```bash
# Using curl
curl http://localhost:8000/api/health

# Using Postman
Import collection from docs/postman_collection.json
```

---

## 📊 ML Model Details

### Pre-training (OULAD)
- **Dataset:** 32,593 students, 10+ million VLE interactions
- **Features:** Late submission rate, study irregularity, last-minute activity, login gaps
- **Clustering:** K-Means (k=3) to create Low/Medium/High risk labels
- **Architecture:** Bi-LSTM with attention (128→64→32 units)
- **Performance:** ~75-85% accuracy (varies by sample)

### Transfer Learning (Local Data)
- **Fine-tuning:** Freeze early layers, train on local survey data
- **Local Dataset:** 100-120 students (survey responses)
- **Adaptation:** Institution-specific patterns

---

## 🎨 Design Files

### Figma Mockups
- [Student Dashboard](link-to-figma)
- [Admin Interface](link-to-figma)
- [Risk Display Components](link-to-figma)

### Screenshots
See `docs/screenshots/` for interface examples

---

## 🚢 Deployment Plan

### Current Status: ✅ Checkpoint Demo
- ML model pre-training complete
- Basic frontend structure ready
- FastAPI backend skeleton functional

### Phase 1: Initial Deployment (Feb 6, 2026)
- Deploy backend to **Render** (free tier)
- Deploy frontend to **Vercel/Netlify**
- Basic prediction endpoint working

### Phase 2: Full Deployment (Feb 27, 2026)
- Complete MCII chatbot integration
- Admin dashboard finalized
- Production database setup
- Security hardening (HTTPS, rate limiting)

### Deployment Commands
```bash
# Backend (Render)
# Connect GitHub repo to Render
# Build command: pip install -r requirements.txt
# Start command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT

# Frontend (Vercel)
vercel --prod
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/api/health

# Get prediction (requires auth token)
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"student_id": 123}'
```

---

## 📹 Video Demo

**Duration:** 5-10 minutes  
**Content:**
1. Problem statement (30s)
2. ML notebook walkthrough (2 min)
   - OULAD data loading
   - K-Means clustering
   - Bi-LSTM training
3. Web platform demo (3 min)
   - Student dashboard
   - Risk prediction
   - MCII chatbot interaction
4. Architecture overview (1 min)
5. Deployment (30s)

**Link:** [YouTube/Drive link here]

---

## 🔑 Key Technologies

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Deep learning
- **SQLAlchemy** - ORM for MySQL
- **OpenAI API** - GPT-4 for MCII interventions
- **Pydantic** - Data validation

### Frontend
- **HTML/CSS/JavaScript** - Core web technologies
- **Tailwind CSS** - Styling
- **Chart.js** - Data visualization
- **Fetch API** - HTTP requests

### ML/Data
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - K-Means clustering, preprocessing
- **Matplotlib/Seaborn** - Visualization

### DevOps
- **Git/GitHub** - Version control
- **Render** - Backend hosting
- **Vercel** - Frontend hosting
- **Google Colab** - Model training (free GPU)

---

## 📝 Development Checklist

### ML Track ✅
- [x] OULAD data preprocessing
- [x] Feature engineering (procrastination indicators)
- [x] K-Means clustering (create labels)
- [x] Bi-LSTM model architecture
- [x] Model training and evaluation
- [x] Save model artifacts
- [ ] Fine-tune on local data (Week 2)

### FullStack Track ⏳
- [x] FastAPI project setup
- [x] Database schema design
- [x] Frontend HTML/CSS structure
- [ ] Authentication system
- [ ] Prediction API endpoint
- [ ] MCII chatbot integration
- [ ] Student dashboard
- [ ] Admin dashboard

### Documentation 📚
- [x] README.md
- [x] Setup instructions
- [ ] Architecture diagram
- [ ] API documentation
- [ ] User guide
- [ ] Video demo

---

## 👥 Contributors

- **Jeremiah Agbaje** - Lead Developer
- **Bernard Lamptey** - Project Supervisor

---

## 📄 License

This project is for academic purposes (BSc Capstone).

---

## 📧 Contact

For questions or feedback:
- Email: jeremiah.agbaje@alustudent.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Last Updated:** February 6, 2026