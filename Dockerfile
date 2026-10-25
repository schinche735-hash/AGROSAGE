FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates
COPY static ./static
COPY models/plantdisease_expanded_frozen_old_candidate.keras ./models/plantdisease_expanded_frozen_old_candidate.keras
COPY models/labels_plantdisease_expanded_frozen_old_candidate.json ./models/labels_plantdisease_expanded_frozen_old_candidate.json
COPY models/labels_pv_original.json ./models/labels_pv_original.json

RUN mkdir -p /app/models /app/data /app/static/uploads && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000
CMD ["sh", "-c", "exec python app.py"]

COPY research.py .
COPY vision_verification.py .
