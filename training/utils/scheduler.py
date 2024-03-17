from torch.optim import lr_scheduler


def fetch_scheduler(optimizer, T_max=10, eta_min=0):

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    return scheduler
