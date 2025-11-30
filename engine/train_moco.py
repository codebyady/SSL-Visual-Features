# mocomotion/engine/train_moco.py

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_moco_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run ONE epoch of MoCo training (no DDP, single process).

    Returns a small dict of stats (e.g., avg loss).
    """
    model.train()
    running_loss = 0.0
    num_steps = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for (im_q, im_k) in progress_bar:
        im_q = im_q.to(device, non_blocking=True)
        im_k = im_k.to(device, non_blocking=True)

        # forward through MoCo
        logits, labels = model(im_q, im_k)

        # InfoNCE = cross-entropy over (1+K) logits, labels all zeros
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_scalar = loss.item()
        running_loss += loss_scalar
        num_steps += 1

        progress_bar.set_postfix({"loss": f"{loss_scalar:.3f}"})

    avg_loss = running_loss / max(1, num_steps)
    return {"loss": avg_loss}


def train_moco(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    """
    Simple multi-epoch training loop around train_moco_one_epoch.

    This is intentionally minimal for Phase 3:
    - no scheduler
    - no checkpointing
    - no distributed
    """
    for epoch in range(epochs):
        stats = train_moco_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        print(f"[Epoch {epoch}] avg loss: {stats['loss']:.3f}")
