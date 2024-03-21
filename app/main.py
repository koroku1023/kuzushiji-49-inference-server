from datetime import datetime
import logging
import random

from fastapi import FastAPI, File, UploadFile
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.inference.handlers.predict_handler import (
    predict_handler,
    async_predict_handler,
    batch_predict_handler,
    async_batch_data_upload,
)

random.seed(42)
app = FastAPI()
scheduler = AsyncIOScheduler()


# create logfile
logging.basicConfig(
    level=logging.INFO,
    filename="log/inference/inference.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@app.on_event("startup")
async def start_scheduler():
    scheduler.start()


@app.on_event("shutdown")
async def shutdown_scheduler():
    scheduler.shutdown()


# sync prediction
@app.post("/predict/{model_name}")
async def predict(model_name: str, upload_file: UploadFile = File(...)):

    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(0, 99999)
    task_id = f"{start_timestamp}_{random_number:05d}"

    # prediction
    results, log_text = await predict_handler(
        model_name, upload_file, start_timestamp, task_id
    )
    logging.info(f"{task_id}: {log_text}")

    return results


# async prediction
@app.post("/async_predict/{model_name}")
async def async_predict(model_name: str, upload_file: UploadFile = File(...)):

    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(0, 99999)
    task_id = f"{start_timestamp}_{random_number:05d}"

    # prediction
    results, log_text = await async_predict_handler(
        model_name, upload_file, start_timestamp, task_id
    )
    logging.info(f"{task_id}: {log_text}")

    return results


# batch prediction
@app.post("/schedule/{model_name}")
async def batch_scheduler(
    model_name: str, execute_at: datetime, upload_file: UploadFile = File(...)
):

    start_timestamp = execute_at.strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(0, 99999)
    task_id = f"{start_timestamp}_{random_number:05d}"

    if model_name not in ("cnn", "densenet"):
        logging.info(f"{task_id}: error: Unsupported model name.")
        return {"error": "Unsupported model name."}

    # data upload in the data dir
    results, log_text, tf = await async_batch_data_upload(upload_file, task_id)

    if tf:
        # add scheduling
        scheduler.add_job(
            func=_execute_scheduled_task,
            trigger="date",
            run_date=execute_at,
            args=[model_name, start_timestamp, task_id],
        )
    else:
        logging.info(f"{task_id}: {log_text}")

        return results

    return {
        "message": f"Batch prediction task {task_id},  scheduled at {execute_at}"
    }


def _execute_scheduled_task(
    model_name: str, start_timestamp: str, task_id: str
):

    _, log_text = batch_predict_handler(model_name, start_timestamp, task_id)
    logging.info(f"{task_id}: {log_text}")
