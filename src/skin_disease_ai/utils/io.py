from pathlib import Path
import torch


def load_checkpoint(path, map_location="cpu"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location=map_location,
        weights_only=False
    )