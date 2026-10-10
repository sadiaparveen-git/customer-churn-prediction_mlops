# Serving image for the churn prediction API (FastAPI + XGBoost).
#
# Build from the project root (the build context must be the root):
#   docker build -f docker/Dockerfile -t churn-api .
# Run:
#   docker run --rm -p 8000:8000 churn-api

# Same Python minor version the model was trained with (3.11)
FROM python:3.11-slim

# No .pyc files; unbuffered output so logs appear in `docker logs`;
# no pip cache kept in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# XGBoost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run the service as a non-root user
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Install dependencies first: this layer is reused until the pins change,
# so code changes do not trigger a reinstall
# (nvidia-nccl-cu12 is a GPU library the ARM xgboost wheel pulls in; CPU
# serving never loads it, so it is removed in the same layer to save ~470 MB)
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt \
    && pip uninstall -y nvidia-nccl-cu12

# Application code and the model bundle (whole folders). They stay owned by
# root and read-only to the service user.
COPY app/ ./app/
COPY src/ ./src/
COPY model/ ./model/

# `app` and `src` are imported as top-level packages. The model location is
# set explicitly instead of relying on the default relative path.
ENV PYTHONPATH=/app \
    MODEL_DIR=/app/model

USER appuser
EXPOSE 8000

# The slim image has no curl, so probe /health with Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 0.0.0.0 makes the port reachable from outside the container. No --reload
# (development only). One worker per container: each worker loads its own
# copy of the model, so scale by running more containers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
