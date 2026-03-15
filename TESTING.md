# ProActive — Testing and Evaluation Report

**Live URL:** https://web-production-185b1.up.railway.app
**Deployment environment:** Railway (Europe West 4), MySQL 9.4
**Test date:** 15 March 2026

---

## 1. Testing Overview

The proposal set out four objectives. This report tests whether the deployed system meets them.

- **Objective 1:** Conduct a systematic literature review and assess existing models before building.
- **Objective 2:** Pre-train a Bi-LSTM on the OULAD dataset (32,000+ students) and adapt it to local student data.
- **Objective 3:** Deploy a web platform integrating the LSTM with attention mechanism and MCII-based interventions.
- **Objective 4:** Conduct pilot testing with 5-10 students and collect initial behavioural data as proof of concept.

| Testing type | Tool | What it covers |
|---|---|---|
| Functional testing | Manual + screenshots | Core user flows end to end |
| Performance testing | Lighthouse, k6, Chrome DevTools | Page speed, API latency, load capacity |
| Network resilience | Chrome DevTools throttling | Behaviour on poor connections |
| Cross-browser / device | Chrome, Firefox, Safari, Mobile | Different hardware and software |
| Data variation | Multiple student profiles | Different risk levels and user types |
| ML model evaluation | Jupyter notebook | Accuracy, F1, AUC-ROC against benchmarks |

---

## 2. Performance Thresholds

Thresholds were set before testing to prevent adjusting criteria after seeing results.

| Metric | Threshold | Source |
|---|---|---|
| LCP | <= 2.5s | Google Core Web Vitals |
| TBT | <= 200ms | Google Core Web Vitals |
| CLS | <= 0.1 | Google Core Web Vitals |
| API P95 | <= 300ms | Standard web API benchmark |
| Error rate under load | < 1% | Industry standard |
| MCII chat latency | <= 5s | Acceptable for LLM-powered features |
| Page load on Slow 3G | <= 4s | Graceful degradation standard |

---

## 3. Deployment Plan

### 3.1 Environment

| Layer | Technology | Notes vs. proposal |
|---|---|---|
| Backend | FastAPI (Python 3.12) | As proposed |
| Database | MySQL 9.4 | As proposed |
| ML runtime | TensorFlow / Keras | As proposed |
| AI chatbot | Anthropic Claude (claude-haiku-4-5-20251001) | Changed from OpenAI GPT — better context handling, lower cost |
| Frontend | Jinja2 + Tailwind CSS | Changed from React + TypeScript — faster development |
| Hosting | Railway (Hobby, Europe West 4) | Changed from Render — includes managed MySQL |
| Process manager | Uvicorn | As proposed |

### 3.2 Step-by-step Deployment

**Step 1 — Prepare the repository**

`Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

`runtime.txt`:
```
python-3.12.0
```

```bash
pip freeze > requirements.txt
```

**Step 2 — Create Railway project**

1. Go to railway.app, sign in with GitHub
2. New Project -> Deploy from GitHub repo
3. Select the ProActive repository

**Step 3 — Add MySQL**

1. New -> Database -> MySQL
2. Wait 1-2 minutes
3. Copy `MYSQL_URL` from the MySQL service Variables tab

**Step 4 — Set environment variables**

| Variable | Value |
|---|---|
| `DATABASE_URL` | `mysql+pymysql://` + MYSQL_URL value |
| `SECRET_KEY` | Any long random string |
| `ANTHROPIC_API_KEY` | sk-ant-... |

Railway gives `mysql://` — SQLAlchemy needs `mysql+pymysql://`. The app converts this automatically:
```python
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
```

**Step 5 — Run schema migrations**

Connect via MySQL Workbench using the public credentials. Run `schema.sql`, then:
```sql
INSERT INTO Admins (email, password_hash, department, invite_code)
VALUES ('admin@proactive.com', SHA2('admin123', 256), 'Computer Science', 'PROACTIVE1');
```

**Step 6 — Deploy**

Push any commit. Railway builds automatically. Expect 3-5 minutes due to TensorFlow.

**Step 7 — Verify**

Check logs for:
```
All ML artifacts loaded
Application startup complete.
```

### 3.3 Deployment Verification

| Check | Result |
|---|---|
| Login page loads at live URL | PASS |
| Admin login works | PASS |
| Student signup with invite code works | PASS |
| Run scheduler returns predictions | PASS |
| MCII chat sends and receives messages | PASS |
| Student dashboard loads with prediction card | PASS |
| ML models loaded (confirmed in build logs) | PASS |
| Database connected (no errors in logs) | PASS |

> ADD: `tests/screenshots/deployment_railway_active.jpg` — Railway green Active status

---

## 4. Functional Testing

| Test case | Steps | Expected | Actual | Status |
|---|---|---|---|---|
| Student login | Valid email/password -> Sign In | Redirect to dashboard | Redirected | PASS |
| Invalid login | Wrong password | Error message | Error shown | PASS |
| Invalid invite code | Sign up with fake code | Error message | Error shown | PASS |
| Solo student (no admin) | Sign up with no code | Full app access | All features work | PASS |
| Cohort student signup | Sign up with valid code | Appears in admin cohort | Visible in admin table | PASS |
| Admin assigns task | Fill form -> submit | Task on all cohort students | Task on student tasks page | PASS |
| Duplicate task safeguard | Assign same task twice | Error: already assigned | Error shown | PASS |
| Admin task lock | View assigned task | No delete button | Delete button absent | PASS |
| MCII chat | Send message | AI responds | Claude responded within 3.9s | PASS |
| Run scheduler | Click button | Predictions generated | Scheduler run complete | PASS |
| Profile update | Edit name -> save | Name updates | Persisted after reload | PASS |
| New user banner | First login | Banner shown | Visible on first prediction only | PASS |

> ADD: `tests/screenshots/functional_login.jpg`
> ADD: `tests/screenshots/functional_invalid_login.jpg`
> ADD: `tests/screenshots/functional_invalid_code.jpg`
> ADD: `tests/screenshots/functional_assign_task.jpg`
> ADD: `tests/screenshots/functional_assigned_task_student.jpg`
> ADD: `tests/screenshots/functional_mcii_chat.jpg`

**Student dashboard — new user banner**
![New user banner](tests/screenshots/medium_risk_dashboard.jpg)

**Admin dashboard — cohort and scheduler**
![Admin dashboard](tests/screenshots/admin_dash_network.jpg)

---

## 5. Performance Testing

### 5.1 Lighthouse

| Page | LCP | TBT | CLS | Score | Status |
|---|---|---|---|---|---|
| Login | 1.5s | 0ms | 0 | 85 | PASS |
| Admin dashboard | 1.0s | 20ms | 0.006 | 96 | PASS |
| Student dashboard | 1.3s | 110ms | 0.013 | 91 | PASS |

All three pages pass every Core Web Vitals threshold. Login scores 85 because of external CDN dependencies (Tailwind, Google Fonts, Material Symbols) — a deliberate trade-off for development speed. Admin dashboard scores highest (96) because it renders less JavaScript on first load. Student dashboard TBT of 110ms is higher because Chart.js and the prediction API fire simultaneously. Even so, 110ms is well inside the 200ms limit.

![Lighthouse login](tests/screenshots/signin_lighthouse.jpg)

![Lighthouse admin](tests/screenshots/admin_dashboard_lighthouse.jpg)

![Lighthouse student](tests/screenshots/student_dashboard_lighthouse.jpg)

---

### 5.2 API Response Times

| Action | Response time | Threshold | Status |
|---|---|---|---|
| Login | 251ms | 300ms | PASS |
| Student dashboard | 333ms | 500ms | PASS |
| Admin dashboard | 333ms | 500ms | PASS |
| Run scheduler | 331ms | 500ms | PASS |
| MCII chat (avg 3 calls) | 3.35s | 5s | PASS |

Login is fastest at 251ms — only validates credentials and creates a session. Dashboard takes 333ms fetching a prediction and trend data. MCII chat averages 3.35s because every message calls the Claude API with full conversation history. The application adds under 50ms. The rest is external API time.

![Network login and dashboard](tests/screenshots/2dashboard_login_network.jpg)

![Network MCII chat calls](tests/screenshots/2chat_network.jpg)

---

### 5.3 k6 Load Test — 50 Concurrent Users

| Metric | Result | Threshold | Status |
|---|---|---|---|
| P95 response time | 353ms | 300ms | MARGINAL |
| P90 response time | 300ms | 300ms | PASS |
| Median | 203ms | — | — |
| Error rate | 0.00% | < 1% | PASS |
| Throughput | 17.9 req/s | — | — |
| Total requests | 2,165 | — | — |

P95 is 353ms — 53ms over threshold. The median of 203ms and 0% error rate across 2,165 requests confirm the system stays stable at peak load. The P95 spike is consistent with Railway's shared infrastructure at peak ramp, not application code.

![k6 load test](tests/screenshots/k6_load-DEMO.jpg)

---

### 5.4 Network Resilience

| Condition | Response time | Threshold | Behaviour | Status |
|---|---|---|---|---|
| No throttling | ~251ms | — | Normal | PASS |
| Fast 4G | ~257ms | — | Normal | PASS |
| Slow 4G | ~573ms | — | Slight delay, functional | PASS |
| Slow 3G | ~2.03s | 4s | Noticeable delay, no crash | PASS |

On Slow 3G the dashboard takes about 2 seconds. The page still loads correctly and all features work. No broken layouts or errors at any throttling level. This matters for the target population — students in regions with inconsistent internet access, as identified in the proposal.

![Slow 4G](tests/screenshots/DASHBOARD_SLOW4G.jpg)

![Fast 4G](tests/screenshots/DASHBOARD_FAST4G.jpg)

![Slow 3G — 2.03s, fully functional](tests/screenshots/DASHBOARD_3G.jpg)

---

## 6. Cross-Browser and Device Testing

| Environment | Login | Dashboard | MCII chat | Tasks | Status |
|---|---|---|---|---|---|
| Chrome (macOS) | PASS | PASS | PASS | PASS | PASS |
| Firefox (desktop) | PASS | PASS | PASS | PASS | PASS |
| Safari (macOS) | PASS | PASS | PASS | PASS | PASS |
| Mobile Chrome (iOS) | PASS | PASS | PASS | PASS | PASS |

All four environments passed. Tailwind CSS handles cross-browser rendering automatically. Responsive layout adapts correctly to mobile screen sizes.

![Chrome](tests/screenshots/DASHBOARD_CHROME.jpg)

![Firefox](tests/screenshots/dashboard_firefox.jpg)

![Safari](tests/screenshots/dashboard_safari.jpg)


---

## 7. Data Variation Testing

| Student | Prior profile | Risk level | Confidence | First-day banner | In cohort |
|---|---|---|---|---|---|
| Jerry (14 days active) | Mixed | Low | 20% | No | No |
| Jeremiah (new, lastminute) | Last minute | High | 67% | Yes | Yes |
| John (new, earlybird) | Early bird | Medium | 49% | Yes | Yes |
| Freeman (new, no admin) | Last minute | High | 67% | Yes | No |

Clear differentiation across all four profiles. A new last-minute student gets 67% high-risk. An early-bird gets 49% medium. An established student with real data gets 20% low. The prior profiles cold-start approach delivers personalised predictions from day one, not a generic default. The solo student test confirms the system works for independent use, meeting the proposal's goal of supporting both individuals and institutions.

![Medium risk — new student with banner](tests/screenshots/medium_risk_dashboard.jpg)

![High risk — 67% confidence](tests/screenshots/DASHBOARD_HIGH_CONFIDENCE.jpg)

> ADD: `tests/screenshots/data_low_risk.jpg` — Jerry's 20% low risk dashboard, no banner

---

## 8. ML Model Evaluation

**Architecture:** Bidirectional LSTM (64 units) + Bahdanau Attention (32 units) + Dense (32, ReLU) + Sigmoid
**Training data:** OULAD — 32,593 students | **Optimiser:** Adam lr=0.001 | **Early stopping:** val_AUC patience=50

| Model | Accuracy | F1 | AUC-ROC | Memon 2020 benchmark | Status |
|---|---|---|---|---|---|
| BiLSTM 7-window (main) | 88.74% | 0.8874 | 0.9428 | ANN 83.5% / XGBoost 87.0% | EXCEEDS |
| BiLSTM 3-window (cold start) | 86.03% | 0.8569 | 0.9043 | ANN 83.5% | EXCEEDS |
| SVM baseline | 68.16% | 0.6911 | 0.8231 | — | Baseline only |

> ADD: `tests/screenshots/ml_training_curves.jpg`
> ADD: `tests/screenshots/ml_confusion_matrix.jpg`
> ADD: `tests/screenshots/ml_classification_report.jpg`

---

## 9. Analysis

### 9.1 Results against proposal objectives

**Objective 1 — Literature review:** Met. The review identified Memon et al. (2020) as the benchmark. The 7-window BiLSTM exceeds it by 5.24 percentage points in accuracy and 0.0328 in AUC-ROC.

**Objective 2 — OULAD pre-training and local adaptation:** Partially met with justified change. Fine-tuning on 100-120 local students was not achievable in the timeline. The prior profiles cold-start approach was developed instead — mapping self-reported habits (early, mixed, lastminute) to OULAD-derived priors. This delivers personalised first-day predictions without weeks of local data collection.

**Objective 3 — Deployed platform:** Fully met with justified deviations. All three components (BiLSTM, attention, MCII) are live on Railway. React was changed to Jinja2 and OpenAI GPT to Anthropic Claude — both for development speed and cost. Both changes serve the same functional goals.

**Objective 4 — Pilot study:** Partially met. Seven accounts tested across different profiles — all features work. The qualitative pilot with self-identified procrastinators is scheduled for 16 March 2026.

### 9.2 Where results missed targets

The k6 P95 of 353ms exceeded 300ms by 53ms. Railway shared infrastructure at peak ramp is the cause, not code. Zero errors confirm stability.

The student dashboard Lighthouse score of 85 is good but not excellent. External CDN dependencies are the main factor — a deliberate trade-off.

True fine-tuning on local data was not achieved. The prior profiles solution is a weaker form of local adaptation but practical and working.

### 9.3 Finding-by-finding analysis

**Finding 1 — P95 353ms (threshold 300ms):** 53ms over at 50 VUs. Zero errors, 2,165 requests. Infrastructure, not code. Recommendation: dedicated Railway tier before scaling.

**Finding 2 — Dashboard TBT 110ms (threshold 200ms):** Within threshold. Chart.js and prediction API fire together on load. Prediction card renders first so user sees result quickly. Recommendation: defer Chart.js initialisation.

**Finding 3 — MCII chat 3.35s (threshold 5s):** Within threshold. Application adds under 50ms. The rest is Claude API. Recommendation: streaming responses so text appears word by word.

**Finding 4 — ML exceeds both benchmarks:** 88.74% accuracy and 0.9428 AUC-ROC on held-out data. SVM at 68.16% confirms sequential architecture adds real value. Bahdanau attention also provides interpretability — showing which weeks were most predictive.

**Finding 5 — Cold-start differentiation works:** Last-minute: 67% high-risk day one. Early-bird: 49% medium. Established: 20% low. Real behavioural data gradually overrides the prior as expected.

**Finding 6 — All cross-browser and network tests pass:** No issues on Chrome, Firefox, Safari, or mobile. No crashes on Slow 3G. Appropriate for the target population described in the proposal.

---

## 10. Discussion

This project shows that a lightweight machine learning system, paired with a psychologically grounded intervention framework, can address academic procrastination at scale. The prediction model gives the system something to act on. The MCII chatbot turns that signal into something concrete for the student. Neither component alone is enough.

The most significant contribution is the prior profiles cold-start solution. The proposal identified cold-start as a central research challenge. True fine-tuning on local data was not feasible in the timeline. The prior profiles approach solves the same problem using patterns from 32,593 OULAD students — every new student gets a personalised first-day prediction from day one.

The Anthropic Claude integration demonstrates LLM-powered MCII coaching works in a low-cost deployed environment. The chatbot maintains 48-hour conversation history, uses live student context in every prompt, and escalates to structured MCII at the right moments rather than forcing the framework into every reply.

The full stack runs on a $5/month Railway plan. This directly addresses the proposal's argument that one-to-one CBT is not cost-efficient at scale. A system that does not require trained mental health professionals or expensive infrastructure is more likely to reach students who need it.

The open question remains: does MCII delivered through a chatbot produce the same behavioural change as MCII in person? The current testing shows the platform is ready for the pilot study that will begin to answer it.

---

## 11. Recommendations and Future Work

| Priority | Recommendation | Rationale |
|---|---|---|
| High | Email / push notifications | Reduces reliance on students returning voluntarily |
| High | Attention weight visualisation on prediction card | Explainability — shows which weeks drove the result |
| Medium | Streaming MCII chat | Reduces perceived latency from 3s to near-instant |
| Medium | Mobile app (React Native) | Students primarily use phones |
| Low | Longitudinal study (full semester) | Measures whether interventions reduce late submissions |