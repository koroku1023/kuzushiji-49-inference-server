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
import pytz


sys.path.append("/home/jovyan/training")
from utils.dataset import CustomDataset
from preprocessing.preprocess import img_transformer
from model.architectures.cnn import SimpleCNN
from utils.criterion import fetch_criterion
from utils.optimizer import fetch_optimizer
from utils.scheduler import fetch_scheduler
from trainer import train_one_epoch
from evaluator import evaluate

ARGS = {
    "SEED": 42,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "DATA_DIR": "data/raw",
    "MODEL_DIR": "model",
    "LOG_DIR": "log/training",
    "UNDER_SAMPLING": True,
    "IMAGE_SIZE": (28, 28),
    "BATCH_SIZE": 512,
    "NUM_CLASSES": 49,
    "MODEL_NAME": "simple_cnn",
    "CRITERION": "CrossEntropyLoss",
    "OPTIMIZER": "AdamW",
    "LR": 1e-05,
    "T_MAX": 500,
    "MIN_LR": 1e-06,
    "EPOCH": 5,
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
jst = pytz.timezone("Asia/Tokyo")
start_timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
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
    full_dataset = CustomDataset(
        train_images_npz=os.path.join(ARGS["DATA_DIR"], "k49-train-imgs.npz"),
        train_labels_npz=os.path.join(
            ARGS["DATA_DIR"], "k49-train-labels.npz"
        ),
        transformer=transformer,
        exec_under_sampling=ARGS["UNDER_SAMPLING"],
        num_classes=ARGS["NUM_CLASSES"],
    )

    # Create Train・Val Dataset and DataLoader
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.3,
        stratify=full_dataset.labels,
        random_state=ARGS["SEED"],
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=ARGS["BATCH_SIZE"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ARGS["BATCH_SIZE"], shuffle=False
    )

    # Model
    device = ARGS["DEVICE"]
    model = SimpleCNN(num_classes=ARGS["NUM_CLASSES"])
    model.to(device)

    # Criterion, Optimizer, and Scheduler
    loss_fn = fetch_criterion(ARGS["CRITERION"])
    optimizer = fetch_optimizer(
        ARGS["OPTIMIZER"],
        model=model,
        lr=ARGS["LR"],
    )
    scheduler = fetch_scheduler(
        optimizer, T_max=ARGS["T_MAX"], eta_min=ARGS["MIN_LR"]
    )

    best_epoch_loss = np.inf
    best_accuracy = 0.0
    best_f1_score = 0.0
    not_updated_time = 0
    for epoch in range(1, ARGS["EPOCH"] + 1):

        # Train
        model, train_epoch_loss, train_scores = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_accuracy = train_scores["accuracy"]
        train_f1_score = train_scores["f1_score"]
        print()
        print(
            f"<Train> Epoch: {epoch}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:4f}, F1_Score: {train_f1_score:.4f}"
        )
        logging.info(
            f"<Train> Epoch: {epoch}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1_Score: {train_f1_score:.4f}"
        )

        # Val
        val_epoch_loss, val_scores = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )
        val_accuracy = val_scores["accuracy"]
        val_f1_score = val_scores["f1_score"]
        print(
            f"<Val> Epoch: {epoch}, Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:4f}, F1_Score: {val_f1_score:.4f}"
        )
        print()
        logging.info(
            f"<Val> Epoch: {epoch}, Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1_Score: {val_f1_score:.4f}"
        )

        # Early Stopping Check
        if val_epoch_loss < best_epoch_loss:
            print(
                f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            print()
            logging.info(
                f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            best_epoch_loss = val_epoch_loss
            best_accuracy = val_accuracy
            best_f1_score = val_f1_score
            best_model_wts = copy.deepcopy(model.state_dict())
            if epoch == ARGS["EPOCH"]:
                torch.save(
                    best_model_wts,
                    os.path.join(
                        ARGS["MODEL_DIR"], f"{ARGS['MODEL_NAME']}.pth"
                    ),
                )
        else:
            not_updated_time += 1
            if not_updated_time == 2:
                torch.save(
                    best_model_wts,
                    os.path.join(
                        ARGS["MODEL_DIR"], f"{ARGS['MODEL_NAME']}.pth"
                    ),
                )
                print("Execute Early Stopping")
                logging.info("Execute Early Stopping")
                break

    print(
        f"<Best Val Score> Accuracy: {best_accuracy:4f}, F1_Score: {best_f1_score:4f}"
    )
    logging.info(
        f"<Best Val Score> Accuracy: {best_accuracy:4f}, F1_Score: {best_f1_score:4f}"
    )


if __name__ == "__main__":
    main()
