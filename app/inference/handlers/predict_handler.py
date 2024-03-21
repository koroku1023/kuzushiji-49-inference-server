import io
import os
import sys
import json
from typing import Dict

from fastapi import UploadFile
import pandas as pd
import numpy as np

sys.path.append("app/inference")
from inferences.cnn import cnn_inference
from inferences.densenet import densenet_inference


# exec sync prediction
async def predict_handler(
    model_name: str,
    upload_file: UploadFile,
    start_timestamp: str,
    task_id: str,
):

    file_extension = upload_file.filename.split(".")[-1]

    if file_extension == "npz":
        images = await _async_load_npz_file(upload_file)
    else:
        log_text = (
            "error: Unsupported file format. Only npz files are allowed."
        )
        return {
            "error": "Unsupported file format. Only npz files are allowed."
        }, log_text

    # save uploaded file
    np.savez_compressed(
        os.path.join(f"data/input/{task_id}_input_images.npz"),
        arr_0=images,
    )

    # inference
    if model_name == "cnn":
        predict_probas, predictions = cnn_inference("simple_cnn", images)
    elif model_name == "densenet":
        predict_probas, predictions = densenet_inference("densenet", images)
    else:
        log_text = "error: Unsupported model name."
        return {"error": "Unsupported model name."}, log_text

    results = _fetch_results(
        predict_probas, predictions, start_timestamp, task_id
    )
    log_text = f"results: {results}"

    _save_results(results, task_id, "sync")

    return {"model": model_name, **results}, log_text


# exec async prediction
async def async_predict_handler(
    model_name: str,
    upload_file: UploadFile,
    start_timestamp: str,
    task_id: str,
    inference_type: str = "async",
):

    file_extension = upload_file.filename.split(".")[-1]

    if file_extension == "npz":
        images = await _async_load_npz_file(upload_file)
    else:
        log_text = (
            "error: Unsupported file format. Only npz files are allowed."
        )
        return {
            "error": "Unsupported file format. Only npz files are allowed."
        }, log_text

    # save uploaded file
    np.savez_compressed(
        os.path.join(f"data/input/{task_id}_input_images.npz"),
        arr_0=images,
    )

    # inference
    if model_name == "cnn":
        predict_probas, predictions = await _async_cnn_inference(
            "simple_cnn", images
        )
    elif model_name == "densenet":
        predict_probas, predictions = await _async_densenet_inference(
            "densenet", images
        )
    else:
        log_text = "error: Unsupported model name."
        return {"error": "Unsupported model name."}, log_text

    results = _fetch_results(
        predict_probas, predictions, start_timestamp, task_id
    )
    log_text = f"results: {results}"

    _save_results(results, task_id, inference_type)

    return {"model": model_name, **results}, log_text


# exec batch prediction
def batch_predict_handler(model_name: str, start_timestamp: str, task_id: str):

    images = _load_npz_file(f"{task_id}_input_images.npz")

    if model_name == "cnn":
        predict_probas, predictions = cnn_inference("simple_cnn", images)
    elif model_name == "densenet":
        predict_probas, predictions = densenet_inference("densenet", images)
    else:
        log_text = "error: Unsupported model name."
        return {"error": "Unsupported model name."}, log_text

    results = _fetch_results(
        predict_probas, predictions, start_timestamp, task_id
    )
    log_text = f"results: {results}"

    _save_results(results, task_id, "batch")

    return {"model": model_name, **results}, log_text


# exec batch data uploading
async def async_batch_data_upload(
    upload_file: UploadFile,
    task_id: str,
):

    file_extension = upload_file.filename.split(".")[-1]

    if file_extension == "npz":
        images = await _async_load_npz_file(upload_file)
    else:
        log_text = (
            "error: Unsupported file format. Only npz files are allowed."
        )
        return (
            {"error": "Unsupported file format. Only npz files are allowed."},
            log_text,
            False,
        )

    # save uploaded file
    np.savez_compressed(
        os.path.join(f"data/input/{task_id}_input_images.npz"),
        arr_0=images,
    )
    return None, None, True


async def _async_cnn_inference(model_name: str, images: np.array):

    predict_probas, predictions = cnn_inference(model_name, images)
    return predict_probas, predictions


async def _async_densenet_inference(model_name: str, images: np.array):

    predict_probas, predictions = densenet_inference(model_name, images)
    return predict_probas, predictions


async def _async_load_npz_file(file: UploadFile):

    images = await file.read()
    images = np.load(io.BytesIO(images))["arr_0"]
    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)
    return images


def _load_npz_file(file_name: str):

    images = np.load(os.path.join("data/input", file_name))["arr_0"]
    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)
    return images


def _fetch_results(
    predict_probas: np.array,
    predictions: np.array,
    start_timestamp: str,
    task_id: id,
):

    df_classmap = pd.read_csv("data/raw/k49_classmap.csv", index_col=0)
    results = {
        "task_id": task_id,
        "results": [
            {
                "index": idx,
                "start_timestamp": start_timestamp,
                "prediction": df_classmap.iloc[predictions[idx], 1],
                "prediction_label": str(predictions[idx]),
                "probability": str(predict_probas[idx]),
            }
            for idx in range(len(predictions))
        ],
    }
    return results


def _save_results(results: Dict, task_id: str, inference_type: str):

    file_name = f"results_{task_id}.json"
    with open(
        os.path.join(f"data/output/", inference_type, file_name), "w"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
