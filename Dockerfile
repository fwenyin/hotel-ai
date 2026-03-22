FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN chmod +x docker-entrypoint.sh

RUN mkdir -p \
    models/preprocessors \
    models/champion \
    logs \
    outputs \
    mlruns \
    data/processed \
    reports

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["./docker-entrypoint.sh"]

