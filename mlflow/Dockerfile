FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mlflow>=2.14.0,<3" \
    psycopg2-binary \
    boto3 \
    gunicorn

WORKDIR /mlflow
EXPOSE 5000
ENV PYTHONUNBUFFERED=1

ENV POSTGRES_USER=mlops
ENV POSTGRES_PASSWORD=mlopspassword
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=mlflow

CMD mlflow server \
    --backend-store-uri "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}" \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
