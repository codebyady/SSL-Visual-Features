# mocomotion/scripts/pretrain_moco.py

import argparse

import torch
from torch.utils.data import DataLoader

from data.datasets import build_pretrain_dataset
from moco.builder import MoCo
from engine.train_moco import train_moco


def parse_args():
    parser = argparse.ArgumentParser(description="MoCo v2 pretraining (Phase 3 minimal)")

    # basic training hyperparams
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--lr", type=float, default=0.03, help="learning rate (SGD)"
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

    return parser.parse_args()


def main():
    args = parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset + dataloader
    dataset = build_pretrain_dataset(
        root=args.data_root,
        image_size=args.image_size,
    )

    if args.max_samples > 0:
        # for fast local testing; don't do this on the cluster
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        print(f"Using a subset of {len(dataset)} images for debugging.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    # optimizer (MoCo v2 uses SGD + momentum)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("Starting training...")
    train_moco(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )
    print("Training finished (Phase 3 minimal).")


if __name__ == "__main__":
    main()
