import os
import sys
import random
import copy
import logging
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset


sys.path.append("/home/jovyan/training")
from utils.dataset import CustomDataset
from preprocessing.preprocess import img_transformer
from model.architectures.cnn import SimpleCNN
from predict import predict

ARGS = {
    "SEED": 42,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "DATA_DIR": "data/raw",
    "MODEL_DIR": "model",
    "LOG_DIR": "log/test",
    "UNDER_SAMPLING": False,
    "IMAGE_SIZE": (28, 28),
    "BATCH_SIZE": 512,
    "NUM_CLASSES": 49,
    "MODEL_NAME": "simple_cnn",
    "CRITERION": "CrossEntropyLoss",
    "OPTIMIZER": "AdamW",
    "LR": 1e-05,
    "T_MAX": 500,
    "MIN_LR": 1e-06,
    "EPOCH": 40,
}


def worker_init_fn(worker_id=0):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ["PYTHONHASHSEED"] = str(worker_id)


def set_seed(seed=0):
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

# create logfile
start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(
        ARGS["LOG_DIR"], f"{start_timestamp}_{ARGS['MODEL_NAME']}.log"
    ),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():

    transformer = img_transformer(image_size=ARGS["IMAGE_SIZE"])

    # Create Dataset
    test_dataset = CustomDataset(
        images_npz=os.path.join(ARGS["DATA_DIR"], "k49-test-imgs.npz"),
        labels_npz=os.path.join(ARGS["DATA_DIR"], "k49-test-labels.npz"),
        transformer=transformer,
        exec_under_sampling=ARGS["UNDER_SAMPLING"],
        num_classes=ARGS["NUM_CLASSES"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=ARGS["BATCH_SIZE"], shuffle=True
    )

    # Model
    device = ARGS["DEVICE"]
    model = SimpleCNN(num_classes=ARGS["NUM_CLASSES"])
    model.load_state_dict(
        torch.load(os.path.join(ARGS["MODEL_DIR"], "simple_cnn.pth"))
    )
    model.to(device)

    # Test
    test_scores = predict(model, test_loader, device)
    test_accuracy = test_scores["accuracy"]
    test_precision = test_scores["precision"]
    test_recall = test_scores["recall"]
    test_f1_score = test_scores["f1_score"]

    print(
        f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}  F1_Score: {test_f1_score:.4f}"
    )
    logging.info(
        f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}  F1_Score: {test_f1_score:.4f}"
    )


if __name__ == "__main__":
    main()
