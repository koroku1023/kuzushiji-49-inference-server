import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("/home/jovyan/training")
from preprocessing.preprocess import under_sampling


class CustomDataset(Dataset):

    def __init__(
        self,
        train_images_npz=None,
        train_labels_npz=None,
        transformer=None,
        exec_under_sampling=False,
        num_classes=None,
    ):

        self.images = np.load(train_images_npz)["arr_0"]
        self.labels = np.load(train_labels_npz)["arr_0"]
        self.transformer = transformer

        if exec_under_sampling:
            self.images, self.labels = under_sampling(
                self.images, self.labels, num_classes
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        label = self.labels[idx]

        image = self.transformer(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)
