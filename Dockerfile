FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SIMS_DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/sims
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
RUN useradd --create-home --uid 10001 sims && mkdir /data && chown sims:sims /data
COPY --chown=sims:sims app ./app
COPY --chown=sims:sims scripts ./scripts
USER sims
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
