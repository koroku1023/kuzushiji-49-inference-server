import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(
        self, train_imgs_npz=None, train_labels_npz=None, transformer=None
    ):

        self.imgs = np.load(train_imgs_npz)["arr_0"]
        self.imgs = self.imgs[:5000]
        self.labels = np.load(train_labels_npz)["arr_0"]
        self.labels = self.labels[:5000]
        self.transformer = transformer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.imgs[idx]
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        label = self.labels[idx]

        img = self.transformer(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)
