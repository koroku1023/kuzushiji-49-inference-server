FROM jupyter/base-notebook:latest

COPY infra/jupyter/requirements.txt /home/jovyan/requirements.txt
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt
