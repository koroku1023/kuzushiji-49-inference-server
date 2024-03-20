import io
import os
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import pytz

from app.inference.handlers.predict_handler import predict_handler


app = FastAPI()

# create logfile
logging.basicConfig(
    level=logging.INFO,
    filename="log/inference/inference.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# TODO: add asynchronous and batch
@app.post("/predict/{model_name}")
async def predict(model_name: str, upload_file: UploadFile = File(...)):

    jst = pytz.timezone("Asia/Tokyo")
    start_timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")

    results, log_text = await predict_handler(
        model_name, upload_file, start_timestamp
    )
    logging.info(log_text)

    return results
