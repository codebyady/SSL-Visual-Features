# mocomotion/utils/checkpoint.py

import os
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], filename: str) -> None:
    """
    Save training state to a file.

    state should typically contain:
      - "epoch": int
      - "model": model.state_dict()
      - "optimizer": optimizer.state_dict()
      - (later) "scheduler": scheduler.state_dict(), etc.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Load training state from a file.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    return torch.load(filename, map_location=map_location)


def get_model_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Return the correct state_dict, handling both plain nn.Module
    and DistributedDataParallel-wrapped models.
    """
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()
