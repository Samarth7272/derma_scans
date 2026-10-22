import torch


def get_device():
    """Select an available computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")