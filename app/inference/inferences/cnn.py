import os
import sys
import random
import copy
import logging
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("app/inference")
from utils.dataset import CustomDataset
from utils.preprocess import img_transformer
from architectures.cnn import SimpleCNN
from predict import predict


ARGS = {
    "SEED": 42,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "DATA_DIR": "data/raw",
    "MODEL_DIR": "model",
    "LOG_DIR": "log",
    "UNDER_SAMPLING": True,
    "IMAGE_SIZE": (28, 28),
    "BATCH_SIZE": 512,
    "NUM_CLASSES": 49,
    "MODEL_NAME": "simple_cnn",
}


def worker_init_fn(worker_id: int = 0):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ["PYTHONHASHSEED"] = str(worker_id)


def set_seed(seed: int = 0):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(ARGS["SEED"])

# TODO logging


def cnn_inference(model_name, images):

    transformer = img_transformer(image_size=ARGS["IMAGE_SIZE"])

    # Create Dataset and DataLoader
    inference_dataset = CustomDataset(images, transformer)
    inference_loader = DataLoader(
        inference_dataset, batch_size=ARGS["BATCH_SIZE"], shuffle=False
    )

    # Load Model
    model = SimpleCNN(num_classes=ARGS["NUM_CLASSES"])
    model.load_state_dict(
        torch.load(os.path.join(ARGS["MODEL_DIR"], "simple_cnn.pth"))
    )
    model.to(ARGS["DEVICE"])

    # Inference
    predictions = predict(model, inference_loader, ARGS["DEVICE"])

    return predictions
