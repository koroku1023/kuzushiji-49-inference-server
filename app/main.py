import io
import os
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import pytz

from app.inference.inferences.cnn import cnn_inference


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

    file_extension = upload_file.filename.split(".")[-1]
    images = await upload_file.read()

    # TODO: handle multi file extensions
    if file_extension == "npz":
        images = np.load(io.BytesIO(images))["arr_0"]
        if images.ndim == 2:
            images = np.expand_dims(images, axis=0)
    else:
        logging.info(
            "error: Unsupported file format. Only npz files are allowed."
        )
        return {
            "error": "Unsupported file format. Only npz files are allowed."
        }

    # save uploaded file
    np.savez_compressed(
        os.path.join(f"data/upload/{start_timestamp}_upload_images.npz"),
        arr_0=images,
    )

    # inference
    if model_name == "cnn":
        predict_probas, predictions = cnn_inference("simple_cnn", images[:10])
    else:
        logging.info("error: Unsupported model name.")
        return {"error": "Unsupported model name."}

    df_classmap = pd.read_csv("data/raw/k49_classmap.csv", index_col=0)
    results = {
        "results": [
            {
                "index": idx,
                "prediction": df_classmap.iloc[predictions[idx], 1],
                "prediction_label": str(predictions[idx]),
                "probability": str(predict_probas[idx]),
            }
            for idx in range(len(predictions))
        ]
    }
    logging.info(f"results: {results}")

    return {"model": model_name, **results}
