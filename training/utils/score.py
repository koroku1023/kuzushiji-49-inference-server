import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def cal_scores(true_labels, pred_probs):

    true_labels = np.concatenate(true_labels)
    pred_probs = np.concatenate(pred_probs)
    pred_labels = np.argmax(pred_probs, axis=1)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    recall = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return scores
