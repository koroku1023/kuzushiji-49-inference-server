FROM python:3.10.10-slim

WORKDIR /app

COPY infra/inference_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /data/raw

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
