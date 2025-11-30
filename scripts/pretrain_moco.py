# mocomotion/scripts/pretrain_moco.py

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.datasets import build_pretrain_dataset
from engine.train_moco import train_moco
from moco.builder import MoCo
from utils.checkpoint import load_checkpoint
from utils.distributed import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="MoCo v2 pretraining (Phase 6: LR scheduling)")

    # basic training hyperparams
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--lr", type=float, default=0.03, help="base learning rate (before scaling/decay)"
    )

    # LR schedule
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="minimum learning rate (cosine decay end)",
    )

    # model / MoCo params
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
    )
    parser.add_argument("--dim", type=int, default=128, help="projection dim")
    parser.add_argument("--K", type=int, default=65536, help="queue size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum for encoder_k")
    parser.add_argument("--T", type=float, default=0.2, help="softmax temperature")
    parser.add_argument("--no-mlp", action="store_true", help="disable MLP head")

    # data
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/pretrain",
        help="path to pretrain data root",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="input image resolution (must stay 96 for course)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help=(
            "OPTIONAL: limit number of images for local debugging. "
            "Set to -1 to use full dataset."
        ),
    )

    # checkpoint / output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "path to a checkpoint to resume from. "
            "If empty, will look for <output-dir>/checkpoint_latest.pth."
        ),
    )

    # distributed training
    parser.add_argument(
        "--dist-url",
        type=str,
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="distributed backend",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of processes participating in the job",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="rank of this process",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # initialize (or decide not to use) distributed mode
    init_distributed_mode(args)

    # device
    if torch.cuda.is_available() and getattr(args, "gpu", None) is not None:
        device = torch.device("cuda", args.gpu)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_main_process():
        print(f"Using device: {device}")

    # dataset + dataloader
    dataset = build_pretrain_dataset(
        root=args.data_root,
        image_size=args.image_size,
    )

    if args.max_samples > 0:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        if is_main_process():
            print(f"Using a subset of {len(dataset)} images for debugging.")

    # sampler: DistributedSampler in DDP, else None
    if getattr(args, "distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # build MoCo model
    model = MoCo(
        backbone=args.backbone,
        dim=args.dim,
        K=args.K,
        m=args.m,
        T=args.T,
        mlp=not args.no_mlp,
    )
    model.to(device)

    # wrap with DDP if distributed
    if getattr(args, "distributed", False) and torch.cuda.is_available():
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=False,
        )

    # optimizer (MoCo v2 uses SGD + momentum)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,          # base LR (we'll schedule it per epoch)
        momentum=0.9,
        weight_decay=1e-4,
    )

    # checkpoint path logic
    if args.resume:
        checkpoint_path = args.resume
    else:
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")

    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        if is_main_process():
            print(f"Found checkpoint at {checkpoint_path}, loading to resume...")
        ckpt = load_checkpoint(checkpoint_path, map_location=device)
        model_state = ckpt["model"]
        model.load_state_dict(model_state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main_process():
            print(f"Resuming from epoch {start_epoch}.")
    else:
        if args.resume and is_main_process():
            print(f"WARNING: --resume given but file not found: {checkpoint_path}")
        elif is_main_process():
            print("No existing checkpoint found, training from scratch.")

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process():
        print("Starting training...")
        print(
            f"LR schedule: base_lr={args.lr}, warmup_epochs={args.warmup_epochs}, "
            f"min_lr={args.min_lr}, total_epochs={args.epochs}"
        )

    train_moco(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path,
        sampler=sampler,
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    if is_main_process():
        print("Training finished (Phase 6 LR scheduling).")


if __name__ == "__main__":
    main()
