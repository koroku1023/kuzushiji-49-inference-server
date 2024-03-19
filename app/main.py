import io

from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np

from app.inference.inferences.cnn import cnn_inference


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# TODO: add asynchronous and batch
@app.post("/predict/{model_name}")
async def predict(model_name: str, upload_file: UploadFile = File(...)):

    file_extension = upload_file.filename.split(".")[-1]
    images = await upload_file.read()

    # TODO: handle multi file extensions
    if file_extension == "npz":
        images = np.load(io.BytesIO(images))["arr_0"]
        # TODO: save image npz
        # TODO: handle uploaded only one image
    else:
        return {
            "error": "Unsupported file format. Only npz files are allowed."
        }

    if model_name == "cnn":
        predict_probas, predictions = cnn_inference("simple_cnn", images[:10])
    else:
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

    return {"model": model_name, **results}
