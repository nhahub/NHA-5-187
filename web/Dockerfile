# Streamlit web app (apps/web_interface.py). Build context = repo root.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY web/requirements.txt /app/web-requirements.txt
RUN pip install -r /app/web-requirements.txt

# web_interface.py does `from apps.m2_t05_viewer import ...`, so copy the whole apps/ package
COPY apps /app/apps

# Streamlit server talks to the API container by service name
ENV BRAINMRI_API_URL=http://api:8000

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "apps/web_interface_m4_t03.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.maxUploadSize=512"]
