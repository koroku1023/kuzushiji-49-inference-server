FROM jupyter/base-notebook:latest

WORKDIR /home/jovyan

USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

USER jovyan

COPY infra/jupyter/requirements.txt /home/jovyan/requirements.txt
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt

RUN mkdir -p /home/jovyan/data/raw
