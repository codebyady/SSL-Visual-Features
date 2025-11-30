# mocomotion/engine/train_moco.py

from typing import Dict, Optional
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import save_checkpoint, get_model_state_dict
from utils.distributed import is_main_process


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    base_lr: float,
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
) -> float:
    """
    Epoch-based learning rate schedule with:
      - linear warmup for first warmup_epochs
      - cosine decay from base_lr -> min_lr afterwards

    Returns the lr used for this epoch.
    """
    if epoch < warmup_epochs and warmup_epochs > 0:
        # linear warmup from 0 -> base_lr
        lr = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        # cosine decay from base_lr -> min_lr over [warmup_epochs, epochs)
        if epochs <= warmup_epochs:
            # degenerate case: no decay, just use base_lr
            lr = base_lr
        else:
            t = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            # cosine from 0 -> pi
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def train_moco_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run ONE epoch of MoCo training.

    Works for both:
      - single process
      - multi-GPU (DDP) – we only show progress bar on main process.
    """
    model.train()
    running_loss = 0.0
    num_steps = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        leave=False,
        disable=not is_main_process(),  # only main process shows progress
    )

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

        if is_main_process():
            progress_bar.set_postfix({"loss": f"{loss_scalar:.3f}"})

    avg_loss = running_loss / max(1, num_steps)
    return {"loss": avg_loss}


def train_moco(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    start_epoch: int = 0,
    checkpoint_path: Optional[str] = None,
    sampler: Optional[torch.utils.data.Sampler] = None,
    base_lr: float = 0.03,
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
) -> None:
    """
    Multi-epoch training loop.

    - start_epoch: allows resuming from a checkpoint.
    - checkpoint_path: if not None, save after each epoch (main process only).
    - sampler: if a DistributedSampler is used, we call set_epoch(epoch).
    - base_lr, warmup_epochs, min_lr: define the LR schedule.
    """
    for epoch in range(start_epoch, epochs):
        # adjust LR for this epoch
        lr = adjust_learning_rate(
            optimizer=optimizer,
            epoch=epoch,
            epochs=epochs,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )

        # important for DistributedSampler shuffling
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        stats = train_moco_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        if is_main_process():
            print(
                f"[Epoch {epoch}] lr: {lr:.6f} | avg loss: {stats['loss']:.3f}"
            )

            if checkpoint_path is not None:
                state = {
                    "epoch": epoch,
                    "model": get_model_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(state, checkpoint_path)
                print(f"Saved checkpoint to: {checkpoint_path}")
