import sys
import gc
from tqdm import tqdm

import torch

sys.path.append("/home/jovyan/training")
from utils.score import cal_scores


def evaluate(model, val_loader, loss_fn, device):

    model.eval()
    dataset_size = 0
    running_loss = 0.0
    true_labels = []
    pred_probs = []

    for batch in tqdm(val_loader, desc="Val", leave=False):

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        true_labels.append(labels.detach().cpu().numpy())
        pred_probs.append(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / dataset_size
    scores = cal_scores(true_labels, pred_probs)

    gc.collect()

    return epoch_loss, scores
