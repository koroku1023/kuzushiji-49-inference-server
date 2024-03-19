from torch.optim import Adam, AdamW


def fetch_optimizer(optimizer_name, model, lr, betas=(0.9, 0.999)):

    if optimizer_name == "Adam":
        return Adam(model.parameters(), lr=lr, betas=betas)
    elif optimizer_name == "AdamW":
        return AdamW(model.parameters(), lr=lr, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
