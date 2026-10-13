# FinSight AI — Production Deployment Guide
**Enterprise-Grade SME Financial Solvency, Runway Forecasting & Copilot Advisory System**

This guide provides step-by-step instructions for deploying FinSight AI across local, containerized, and production cloud environments.

---

## 🏗️ System Architecture Overview

FinSight AI consists of two primary operational deployment modes:

1. **Interactive Executive Web Application (`app.py`)**:
   - High-aesthetic Streamlit dashboard with real-time scenario stress-testing sliders, Plotly cash runway quantile trajectories, and a zero-hallucination tool-grounded AI copilot.
2. **Scheduled Batch & Risk Alerting Pipeline (`scripts/run_daily_inference.py`)**:
   - Nightly batch runner that ingests recent POS sales, computes 30-day quantile autoregressive forecasts, evaluates deterministic GAAP cashflow transitions, and exports structured risk JSON payloads.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FINSIGHT AI RUNTIME TOPOLOGY                     │
├───────────────────────────────┬─────────────────────────────────────────────┤
│   Interactive Dashboard (UI)  │       Scheduled Daily Batch Job             │
│   streamlit run app.py        │       python scripts/run_daily_inference.py │
│   Port: 8501                  │       Runs: Cron / GitHub Actions @ 00:00   │
└───────────────┬───────────────┴──────────────────────┬──────────────────────┘
                │                                      │
                ▼                                      ▼
     [Deterministic Cashflow Engine] ◄────► [LightGBM Forecast Engine]
                │                                      │
                ▼                                      ▼
     [Pydantic Tool-Calling Agent]          [JSON Daily Risk Report]
```

---

## 🚀 Deployment Methods

### Method 1: Local & Internal Network (Fastest)

Run the dashboard directly on your workstation or internal company server using your existing Python environment:

```bash
# 1. Activate your Python environment (Python 3.10+)
# Example with Anaconda:
conda activate base

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

- **Local URL:** `http://localhost:8501`
- **Network URL:** `http://<YOUR_LAN_IP>:8501`

---

### Method 2: Docker Containerization (Recommended for Cloud)

A production-ready multi-stage `Dockerfile` and `docker-compose.yml` are provided in the repository root.

#### Single Container Run:
```bash
# Build the Docker image
docker build -t finsight-ai:latest .

# Run container with port forwarding
docker run -d --name finsight_app -p 8501:8501 finsight-ai:latest

# Check container status
docker ps
```

#### Docker Compose Orchestration:
```bash
# Launch with healthcheck and volume mounting
docker compose up -d

# View real-time container logs
docker compose logs -f

# Stop container
docker compose down
```

---

### Method 3: 1-Click Free Cloud Deployment (Streamlit Community Cloud)

You can host FinSight AI publicly or privately for free with continuous Git deployment:

1. Push this repository to **GitHub** (public or private).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Configure deployment settings:
   - **Repository:** `<your-github-username>/<your-repo-name>`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
5. Click **"Deploy!"**.
   - Streamlit Cloud automatically parses `requirements.txt`, provisions the container, and assigns an SSL-secured URL (e.g., `https://finsight-ai.streamlit.app`).

---

### Method 4: Hugging Face Spaces (Free Cloud Alternative)

1. Create a new Space on **[huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Select **Streamlit** (or **Docker**) as the SDK.
3. Push the repository files to the Hugging Face Git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git push hf main
   ```
4. Hugging Face automatically compiles and exposes the live interactive application.

---

### Method 5: Enterprise Cloud Deployment (Google Cloud Run / AWS ECS)

#### Deploy to Google Cloud Run (Serverless, Autoscaling to Zero):
```bash
# Set GCP Project
gcloud config set project <YOUR_PROJECT_ID>

# Build image on Google Cloud Build
gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/finsight-ai:latest

# Deploy to Cloud Run (Managed)
gcloud run deploy finsight-ai \
    --image gcr.io/<YOUR_PROJECT_ID>/finsight-ai:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --cpu 2
```

#### Deploy to AWS App Runner:
1. Push Docker image to Amazon ECR:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URI>
   docker tag finsight-ai:latest <ECR_URI>:latest
   docker push <ECR_URI>:latest
   ```
2. Create an **App Runner Service** pointing to the ECR image on port `8501`.

---

### Method 6: Automated Nightly Batch Monitoring (GitHub Actions / Cron)

To run unattended risk scoring every night and generate daily alert snapshots:

#### Via Local Cron (Linux / macOS / WSL):
```bash
# Open crontab editor
crontab -e

# Run every night at midnight (00:00)
0 0 * * * cd /path/to/finsight && /path/to/venv/bin/python scripts/run_daily_inference.py >> /var/log/finsight_batch.log 2>&1
```

#### Via GitHub Actions (Built-in):
The workflow `.github/workflows/daily_monitoring.yml` is already committed:
- Runs automatically at **00:00 UTC every day**.
- Ingests recent transactions, executes `run_daily_inference.py`.
- Generates `daily_risk_report_YYYY-MM-DD.json` and attaches it as a GitHub Artifact.

---

## 🔒 Production Invariants & Pre-Flight Checks

Before deploying to live production, run the automated verification suite:

```bash
# 1. Run unit test suite
pytest tests -v

# 2. Run data provenance & license audit
python scripts/download_data.py

# 3. Test daily batch runner
python scripts/run_daily_inference.py
```

### Key Environment Settings
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `STREAMLIT_SERVER_PORT` | `8501` | HTTP port for web dashboard |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind IP for incoming requests |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Prevents browser auto-open in headless containers |
| `PYTHONUNBUFFERED` | `1` | Forces stdout flushing for real-time Docker logging |
