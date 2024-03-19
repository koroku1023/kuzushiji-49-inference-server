import torch.nn as nn


def fetch_criterion(criterion_name):

    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")
