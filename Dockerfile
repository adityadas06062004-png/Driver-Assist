# Runs the dashboard (and the tests). The webcam monitor is meant to run natively:
# Docker Desktop on macOS and Windows cannot pass a webcam through to a container.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir ".[dashboard]"

# Session data lives here, so mount a volume to keep it between runs.
ENV HOME=/data
VOLUME /data

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
