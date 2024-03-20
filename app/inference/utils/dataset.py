import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(
        self,
        images: np.array = None,
        transformer: Any = None,
    ):

        self.images = images
        self.transformer = transformer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):

        image = self.images[idx]
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = self.transformer(image=image)["image"]

        return image
