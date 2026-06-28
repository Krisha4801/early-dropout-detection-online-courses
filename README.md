---
title: Early Dropout Detection in Online Courses
sdk: docker
app_port: 7860
---

# Early Dropout Detection in Online Courses

This project trains a machine learning model to detect students at risk of dropping out of an online course using early-course behavior.

The current data folder matches the OULAD-style CSV layout:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `courses.csv`

The loader supports both normal CSV files and the current layout where some names are folders containing a same-named CSV.

## Target

The model predicts:

```text
dropout = final_result == "Withdrawn"
```

By default, features are built using only information available up to course day 30.

## Features

The pipeline creates one row per student-course registration and includes:

- student profile: gender, region, education, age band, IMD band, disability
- registration features: registration timing and previous attempts
- course features: course length and progress by cutoff day
- VLE behavior: total clicks, active days, click timing, site diversity, activity-type clicks
- assessment behavior: submissions, missing due assessments, late/on-time submissions, scores

Outcome fields such as `final_result` and `date_unregistration` are not used as model inputs.

## Resume-Worthy ML Pieces

This project includes the pieces that make it stronger than a simple if-else demo:

- SHAP explainability for individual predictions.
- A transparent rule-based baseline for comparison.
- Recall-first thresholding because missing a real dropout is more costly than raising an extra alert.
- Honest evaluation with precision, recall, F1, F2, ROC AUC, average precision, and confusion matrices.

Current validation result for the selected random forest model:

```text
Precision: 0.552
Recall:    0.790
F2:        0.727
ROC AUC:   0.844
AP:        0.766
```

Current baseline comparison:

```text
Rule baseline recall: 0.720
Random forest recall: 0.790
```

Training also generates a local model card and evaluation reports under `reports/`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Explore the Data

```powershell
python scripts/explore_data.py --data-dir .
```

## Train

```powershell
python scripts/train_model.py --data-dir . --cutoff-day 30
```

Training writes:

- `models/dropout_model.joblib`
- `reports/metrics.json`
- `reports/feature_importance.csv`
- `reports/baseline_vs_model_report.csv`
- `reports/threshold_tradeoffs.csv`
- `reports/model_card.md`
- `reports/validation_predictions.csv`
- `reports/feature_columns.csv`

## Predict Dropout Risk

```powershell
python scripts/make_predictions.py --data-dir . --model-path models/dropout_model.joblib --output reports/dropout_risk_predictions.csv
```

The prediction file contains:

- student and course identifiers
- actual label when available
- dropout probability
- predicted dropout flag
- risk band: `low`, `watch`, `medium`, `high`

## Change the Early Window

Use a different cutoff day to predict earlier or later:

```powershell
python scripts/train_model.py --data-dir . --cutoff-day 14
```

Lower cutoff days make the task harder but closer to a true early-warning system.

## Web App

The project also includes a simple React + Express interface where a user enters one learner's early course behavior and gets a live dropout-risk prediction.

The visible form is intentionally small. It asks for learner ID, clicks, active days, inactivity, studied credits, assessments due/submitted, average score, and late submissions. Course module and presentation are intentionally excluded from the user flow and model inputs because their impact was small compared with behavior and assessment signals.

Folders:

- `client/` - Vite + React frontend
- `server/` - Express API
- Python bridge that calls `models/dropout_model.joblib`
- In-memory recent prediction history for the current server session

Install web dependencies:

```powershell
npm install
npm run install:all
```

Start the API and React app together:

```powershell
npm run dev
```

Or start them separately:

```powershell
npm run server
npm run client
```

Local URLs:

- Frontend: `http://localhost:5173`
- API health: `http://localhost:5000/api/health`

## Deployment

This app needs a backend that can run Node.js and Python because Express calls the saved scikit-learn model through `scripts/predict_manual.py`. Do not deploy it as a static-only site.

The repo includes a root `Dockerfile` for Hugging Face Spaces. It builds the React frontend, installs the Express API, installs Python ML dependencies, and serves everything from one container on port `7860`.

### Push to GitHub First

Create an empty GitHub repository named something like `early-dropout-detection-online-courses` under your GitHub account `Krisha4801`, then run:

```powershell
git init
git lfs install
git add .
git commit -m "Initial early dropout detection app"
git branch -M main
git remote add origin https://github.com/Krisha4801/early-dropout-detection-online-courses.git
git push -u origin main
```

`models/dropout_model.joblib` is intentionally tracked through Git LFS because the deployed app needs it for live predictions.

### Deploy on Hugging Face Spaces

1. Go to `https://huggingface.co/spaces`.
2. Create a new Space.
3. Select **Docker** as the SDK.
4. Use a public Space for a resume demo.
5. Push this same repository to the Space:

```powershell
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/early-dropout-detection-online-courses
git push space main
```

No external database is required. Recent predictions are kept in memory for the current running server session.

If the Space build fails, check the logs for these common issues:

- `models/dropout_model.joblib` was not pushed with Git LFS.
- Raw CSV files were accidentally committed.
- The app is not listening on port `7860`.
- Python dependencies failed to install.
