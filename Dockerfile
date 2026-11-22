FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY data/generate_dataset.py data/generate_dataset.py
COPY streamlit_app/ streamlit_app/
COPY benchmark_report.json .
RUN mkdir -p models data var && chown -R app:app /app

USER app
EXPOSE 8000 8501
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
