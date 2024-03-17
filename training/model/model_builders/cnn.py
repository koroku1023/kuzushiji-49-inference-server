import os
import sys
import random

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset


sys.path.append("/home/jovyan/training")
from utils.dataset import CustomDataset
from preprocessing.preprocess import img_transformer
from model.architectures.cnn import SimpleCNN
from utils.criterion import fetch_criterion
from utils.optimizer import fetch_optimizer
from utils.scheduler import fetch_scheduler
from trainer import train_one_epoch

ARGS = {
    "SEED": 42,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "DATA_DIR": "data/raw",
    "IMAGE_SIZE": (28, 28),
    "BATCH_SIZE": 64,
    "NUM_CLASSES": 49,
    "CRITERION": "CrossEntropyLoss",
    "OPTIMIZER": "AdamW",
    "LR": 1e-05,
    "T_MAX": 500,
    "MIN_LR": 1e-06,
    "EPOCH": 2,
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


def main():

    transformer = img_transformer(image_size=ARGS["IMAGE_SIZE"])

    full_dataset = CustomDataset(
        train_imgs_npz=os.path.join(ARGS["DATA_DIR"], "k49-train-imgs.npz"),
        train_labels_npz=os.path.join(
            ARGS["DATA_DIR"], "k49-train-labels.npz"
        ),
        transformer=transformer,
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

    for epoch in range(1, ARGS["EPOCH"] + 1):

        model, train_epoch_loss, train_scores = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )


if __name__ == "__main__":
    main()
