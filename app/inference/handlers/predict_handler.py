import io
import os
import sys

from fastapi import File, UploadFile
import pandas as pd
import numpy as np

sys.path.append("app/inference")
from inferences.cnn import cnn_inference


async def predict_handler(
    model_name: str, upload_file: UploadFile, start_timestamp: str
):

    file_extension = upload_file.filename.split(".")[-1]
    images = await upload_file.read()

    if file_extension == "npz":
        images = np.load(io.BytesIO(images))["arr_0"]
        if images.ndim == 2:
            images = np.expand_dims(images, axis=0)
    else:
        log_text = (
            "error: Unsupported file format. Only npz files are allowed."
        )
        return {
            "error": "Unsupported file format. Only npz files are allowed."
        }, log_text

    # save uploaded file
    np.savez_compressed(
        os.path.join(f"data/upload/{start_timestamp}_upload_images.npz"),
        arr_0=images,
    )

    # inference
    if model_name == "cnn":
        predict_probas, predictions = cnn_inference("simple_cnn", images[:10])
    else:
        log_text = "error: Unsupported model name."
        return {"error": "Unsupported model name."}, log_text

    df_classmap = pd.read_csv("data/raw/k49_classmap.csv", index_col=0)
    results = {
        "results": [
            {
                "index": idx,
                "start_timestamp": start_timestamp,
                "prediction": df_classmap.iloc[predictions[idx], 1],
                "prediction_label": str(predictions[idx]),
                "probability": str(predict_probas[idx]),
            }
            for idx in range(len(predictions))
        ]
    }
    log_text = f"results: {results}"

    return {"model": model_name, **results}, log_text
