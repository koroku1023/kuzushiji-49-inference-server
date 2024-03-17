import sys
import gc
from tqdm import tqdm

sys.path.append("/home/jovyan/training")
from utils.score import cal_scores


def train_one_epoch(
    model, train_loader, optimizer, scheduler, loss_fn, device
):

    model.train()
    dataset_size = 0
    running_loss = 0.0
    train_loss = []
    true_labels = []
    pred_probs = []

    for batch in tqdm(train_loader, desc="Training", leave=False):

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        train_loss.append(loss.item())
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        true_labels.append(labels.detach().cpu().numpy())
        pred_probs.append(outputs.detach().cpu().numpy())

        scores = cal_scores(true_labels, pred_probs)

    gc.collect()

    return model, epoch_loss, scores
