import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(
        self, train_imgs_npz=None, train_labels_npz=None, transformer=None
    ):

        self.imgs = np.load(train_imgs_npz)["arr_0"]
        self.labels = np.load(train_labels_npz)["arr_0"]
        self.transformer = transformer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = np.repeat(img[..., np.newaxis], 3, -1)
        label = self.labels[idx]

        img = self.transformer(img)

        return img, torch.tensor(label, dtype=torch.long)
