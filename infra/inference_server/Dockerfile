FROM python:3.10.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY infra/jupyter/requirements.txt /home/jovyan/requirements.txt
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt

COPY infra/inference_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /data/raw

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
