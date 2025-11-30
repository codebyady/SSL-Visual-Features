# mocomotion/scripts/pretrain_moco.py

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from data.datasets import build_pretrain_dataset
from engine.train_moco import train_moco
from moco.builder import MoCo
from utils.checkpoint import load_checkpoint
from utils.distributed import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoCo v2 pretraining (Phase 7: YAML config)"
    )

    # main switch: which config file to use
    parser.add_argument(
        "--config",
        type=str,
        default="configs/moco_r50_96.yaml",
        help="path to YAML config file",
    )

    # light overrides for quick debugging
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override epochs from config (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override batch size from config (optional)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override base LR from config (optional)",
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
        default=None,
        help="override output dir from config (optional)",
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

    # distributed training (same as before)
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


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()

    # load YAML config
    cfg = load_config(args.config)

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
        print(f"Using config: {args.config}")

    # ----- read values from config -----
    # model
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "resnet50")
    dim = model_cfg.get("dim", 128)
    K = model_cfg.get("K", 65536)
    m = model_cfg.get("m", 0.999)
    T = model_cfg.get("T", 0.2)
    mlp = model_cfg.get("mlp", True)

    # data
    data_cfg = cfg.get("data", {})
    data_root = data_cfg.get("root", "data/pretrain")
    image_size = data_cfg.get("image_size", 96)
    cfg_batch_size = data_cfg.get("batch_size", 256)

    # optim
    optim_cfg = cfg.get("optim", {})
    cfg_epochs = optim_cfg.get("epochs", 200)
    cfg_lr = optim_cfg.get("lr", 0.03)
    warmup_epochs = optim_cfg.get("warmup_epochs", 10)
    min_lr = optim_cfg.get("min_lr", 0.0)
    weight_decay = optim_cfg.get("weight_decay", 1.0e-4)
    momentum = optim_cfg.get("momentum", 0.9)

    # checkpoint
    ckpt_cfg = cfg.get("checkpoint", {})
    cfg_output_dir = ckpt_cfg.get("output_dir", "checkpoints")

    # apply CLI overrides (if provided)
    epochs = args.epochs if args.epochs is not None else cfg_epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg_batch_size
    base_lr = args.lr if args.lr is not None else cfg_lr
    output_dir = args.output_dir if args.output_dir is not None else cfg_output_dir

    # dataset + dataloader
    dataset = build_pretrain_dataset(
        root=data_root,
        image_size=image_size,
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
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # build MoCo model
    model = MoCo(
        backbone=backbone,
        dim=dim,
        K=K,
        m=m,
        T=T,
        mlp=mlp,
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
        lr=base_lr,          # base LR (scheduled per epoch in train_moco)
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # checkpoint path logic
    if args.resume:
        checkpoint_path = args.resume
    else:
        checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")

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

    os.makedirs(output_dir, exist_ok=True)

    if is_main_process():
        print("Starting training...")
        print(
            f"LR schedule: base_lr={base_lr}, warmup_epochs={warmup_epochs}, "
            f"min_lr={min_lr}, total_epochs={epochs}"
        )
        print(
            f"Config: backbone={backbone}, dim={dim}, K={K}, m={m}, T={T}, mlp={mlp}"
        )
        print(
            f"Data: root={data_root}, image_size={image_size}, batch_size={batch_size}"
        )
        print(f"Checkpoints: output_dir={output_dir}")

    train_moco(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path,
        sampler=sampler,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
    )

    if is_main_process():
        print("Training finished (Phase 7 with YAML config).")


if __name__ == "__main__":
    main()
