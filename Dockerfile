# FinSight AI — Production Container Deployment
# Lightweight Python 3.11 base
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies (libgomp1 is required for LightGBM OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, processed data, and web dashboard
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/
COPY app.py /app/app.py
COPY server.py /app/server.py
COPY dashboard/ /app/dashboard/
COPY static/ /app/static/

# Expose default port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/health || exit 1

# Launch executive web dashboard
ENTRYPOINT ["python", "server.py"]
