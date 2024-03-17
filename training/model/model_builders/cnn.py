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

ARGS = {
    "SEED": 42,
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "DATA_DIR": "data/raw",
    "BATCH_SIZE": 64,
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

    transformer = img_transformer()

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

    device = ARGS["DEVICE"]


if __name__ == "__main__":
    main()
