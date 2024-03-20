import gc

import numpy as np
import torch
from torch.utils.data import DataLoader


def predict(
    model,
    inference_loader: DataLoader,
    device: str,
):

    model.eval()
    predictions = []

    for images in inference_loader:

        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            predicted = outputs.softmax(dim=1)
            predicted = predicted.cpu().detach().numpy()
        predictions.append(predicted)

    predictions = np.concatenate(predictions, axis=0)
    predict_probas = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)

    gc.collect()
    return predict_probas, predictions
