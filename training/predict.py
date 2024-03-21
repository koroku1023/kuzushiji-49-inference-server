import gc
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.append("/home/jovyan/training")
from utils.score import cal_scores


def predict(
    model,
    test_loader: DataLoader,
    device: str,
):

    model.eval()
    true_labels = []
    pred_probs = []

    for batch in tqdm(test_loader, desc="Test", leave=False):

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        true_labels.append(labels.detach().cpu().numpy())
        pred_probs.append(outputs.detach().cpu().numpy())

    scores = cal_scores(true_labels, pred_probs)

    gc.collect()
    return scores
