import argparse
from pathlib import Path

import math
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.neighbors import KNeighborsClassifier

from moco.builder import MoCo
from utils.checkpoint import load_checkpoint


# ============================================================================
#                       FEATURE EXTRACTOR (MoCo encoder)
# ============================================================================

class MoCoFeatureExtractor:
    """
    Wraps your trained MoCo model and exposes a simple
    `extract_batch_features(tensor_batch)` API that returns numpy features.

    We use ONLY the backbone encoder_q (no projection head, no queue),
    and we keep everything frozen.
    """

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet50",
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T_moco: float = 0.2,
        mlp: bool = True,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Build MoCo model with same hyperparameters as pretraining
        model = MoCo(
            backbone=backbone,
            dim=dim,
            K=K,
            m=m,
            T=T_moco,
            mlp=mlp,
        )

        # Load checkpoint
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        # Freeze all params
        for p in model.parameters():
            p.requires_grad = False

        # Keep reference; we will only use encoder_q
        self.model = model

    @torch.no_grad()
    def extract_batch_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Args:
            images: Tensor of shape [B, 3, H, W] on CPU (we move to device)

        Returns:
            features: numpy array [B, D] from encoder_q backbone, L2-normalized
        """
        images = images.to(self.device, non_blocking=True)

        # Unwrap DDP if ever wrapped (for safety)
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        encoder_q = model.encoder_q  # backbone (ResNet)
        encoder_q.eval()

        feats = encoder_q(images)  # [B, D]
        feats = torch.nn.functional.normalize(feats, dim=1)

        return feats.cpu().numpy()


# ============================================================================
#                       DATASET FOR CUB (train/val/test)
# ============================================================================

class CUBImageDataset(Dataset):
    """
    Simple dataset that reads images + labels from the CUB competition layout
    using the CSV files (train_labels.csv, val_labels.csv, test_images.csv).
    """

    def __init__(self, image_dir, filenames, labels=None, image_size=96):
        self.image_dir = Path(image_dir)
        self.filenames = list(filenames)
        self.labels = None if labels is None else list(labels)

        # 96x96, same ImageNet normalization as during MoCo pretraining
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = self.image_dir / img_name

        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, label, img_name
        else:
            return img, img_name


def collate_fn(batch):
    """
    Custom collate function:
    - train/val: (image, label, filename)
    - test: (image, filename)
    """
    if len(batch[0]) == 3:
        images = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        filenames = [b[2] for b in batch]
        return images, labels, filenames
    else:
        images = torch.stack([b[0] for b in batch], dim=0)
        filenames = [b[1] for b in batch]
        return images, filenames


# ============================================================================
#                  FEATURE EXTRACTION + KNN + SUBMISSION
# ============================================================================

def extract_features_from_loader(feature_extractor, dataloader, split_name="train"):
    all_features = []
    all_labels = []
    all_filenames = []

    print(f"\nExtracting features from {split_name} set.")
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels.cpu().tolist())
        else:  # test
            images, filenames = batch

        feats = feature_extractor.extract_batch_features(images)
        all_features.append(feats)
        all_filenames.extend(filenames)

    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None

    print(
        f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}"
    )
    return features, labels, all_filenames


def train_knn_classifier(
    train_features,
    train_labels,
    val_features,
    val_labels,
    k: int = 5,
):
    print(f"\nTraining KNN classifier (k={k}).")

    clf = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="cosine",
        n_jobs=-1,
    )

    clf.fit(train_features, train_labels)

    train_acc = clf.score(train_features, train_labels)
    val_acc = clf.score(val_features, val_labels)

    print("\nKNN results:")
    print(f"  Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")

    return clf

# ============================================================================
#                        MoCo-style weighted kNN
# ============================================================================

def moco_knn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    k: int = 20,
    T: float = 0.07,
    num_classes: Optional[int] = None,
    device: str = "cuda",
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    MoCo-style weighted kNN classifier.

    Args:
        train_features: [N_train, D] numpy array (feature bank, from train split)
        train_labels:   [N_train] numpy array of int class labels
        query_features: [N_query, D] numpy array (features we want predictions for)
        k:              number of neighbors
        T:              temperature for exp(sim / T) weighting
        num_classes:    number of classes (if None, inferred from train_labels)
        device:         "cuda" or "cpu"
        chunk_size:     process queries in chunks to save memory

    Returns:
        preds: [N_query] numpy array of predicted class ids
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"\n[MoCo kNN] Using device: {device}")

    # convert to torch tensors
    train_features_t = torch.from_numpy(train_features).float().to(device)
    query_features_t = torch.from_numpy(query_features).float().to(device)
    train_labels_t   = torch.from_numpy(train_labels).long().to(device)

    # L2-normalize (cheap and safe even if already normalized)
    train_features_t = torch.nn.functional.normalize(train_features_t, dim=1)
    query_features_t = torch.nn.functional.normalize(query_features_t, dim=1)

    if num_classes is None:
        num_classes = int(train_labels.max()) + 1

    N_query = query_features_t.shape[0]
    all_preds = []

    num_chunks = math.ceil(N_query / chunk_size)
    print(f"[MoCo kNN] Processing {N_query} queries in {num_chunks} chunks (k={k}, T={T}).")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end   = min((chunk_idx + 1) * chunk_size, N_query)

        q = query_features_t[start:end]  # [B, D]

        # cosine similarity (features are normalized)
        sim = torch.mm(q, train_features_t.t())  # [B, N_train]

        # top-k neighbors for each query
        sim_vals, sim_idx = sim.topk(k, dim=1, largest=True, sorted=True)  # [B, k]

        # neighbor labels and weights
        neighbor_labels = train_labels_t[sim_idx]           # [B, k]
        weights = torch.exp(sim_vals / T)                   # [B, k]

        # weighted votes per class
        B = q.shape[0]
        probs = torch.zeros(B, num_classes, device=device)  # [B, C]
        probs.scatter_add_(dim=1, index=neighbor_labels, src=weights)

        preds_chunk = probs.argmax(dim=1)                   # [B]
        all_preds.append(preds_chunk.cpu())

        del q, sim, sim_vals, sim_idx, neighbor_labels, weights, probs
        if device == "cuda":
            torch.cuda.empty_cache()

    preds = torch.cat(all_preds, dim=0)  # [N_query]
    return preds.numpy()


def moco_knn_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    k: int = 20,
    T: float = 0.07,
    device: str = "cuda",
    chunk_size: int = 1024,
) -> float:
    """
    Run MoCo-style kNN on val set and return top-1 accuracy.
    """
    preds = moco_knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        query_features=val_features,
        k=k,
        T=T,
        device=device,
        chunk_size=chunk_size,
    )

    val_labels = np.asarray(val_labels)
    acc = (preds == val_labels).mean()

    print(f"\n[MoCo kNN] Val accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    return acc


def create_submission(test_features, test_filenames, classifier, output_path: str):
    print("\nGenerating predictions on test set.")
    preds = classifier.predict(test_features)

    df = pd.DataFrame({"id": test_filenames, "class_id": preds})
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Submission file created: {output_path}")
    print("=" * 60)
    print(f"Total predictions: {len(df)}")
    print("\nFirst 10 predictions:")
    print(df.head(10))

    print("\nClass distribution (top 10 classes):")
    print(df["class_id"].value_counts().head(10))

    # quick sanity checks
    assert list(df.columns) == ["id", "class_id"], "Columns must be ['id', 'class_id']"
    assert df["class_id"].min() >= 0, "class_id must be >= 0"
    assert df["class_id"].max() <= 199, "class_id must be <= 199"
    assert df.isnull().sum().sum() == 0, "NaNs found in submission!"
    print("\n✓ Submission format looks valid.")


def create_submission_from_preds(test_filenames, preds, output_path: str):
    print("\nGenerating predictions on test set (from MoCo kNN).")

    df = pd.DataFrame({"id": test_filenames, "class_id": preds})
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Submission file created: {output_path}")
    print("=" * 60)
    print(f"Total predictions: {len(df)}")
    print("\nFirst 10 predictions:")
    print(df.head(10))

    print("\nClass distribution (top 10 classes):")
    print(df["class_id"].value_counts().head(10))

    # quick sanity checks
    assert list(df.columns) == ["id", "class_id"], "Columns must be ['id', 'class_id']"
    assert df["class_id"].min() >= 0, "class_id must be >= 0"
    assert df["class_id"].max() <= 199, "class_id must be <= 199"
    assert df.isnull().sum().sum() == 0, "NaNs found in submission!"
    print("\n✓ Submission format looks valid.")


# ============================================================================
#                                  MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Create Kaggle submission using MoCo features + KNN"
    )

    # data & output
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with CUB data (train/val/test + CSVs)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="submission_moco_knn.csv",
        help="Output submission CSV path",
    )

    # MoCo model / checkpoint
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_latest.pth",
        help="Path to pretrained MoCo checkpoint",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Backbone architecture used during SSL pretraining",
    )
    p.add_argument("--dim", type=int, default=128, help="Projection dim used in MoCo")
    p.add_argument("--K", type=int, default=65536, help="Queue size used in MoCo")
    p.add_argument(
        "--m",
        type=float,
        default=0.999,
        help="Momentum used for key encoder in MoCo",
    )
    p.add_argument(
        "--T-moco",
        type=float,
        default=0.2,
        help="Softmax temperature used in MoCo (pretraining)",
    )
    p.add_argument(
        "--no-mlp",
        action="store_true",
        help="Set if you trained MoCo WITHOUT MLP head",
    )

    # eval hyperparams
    p.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Input image resolution (competition uses 96)",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--k", type=int, default=5, help="K for KNN classifier")

    # device
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    print("\nLoading CUB metadata CSVs.")
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    test_df = pd.read_csv(data_dir / "test_images.csv")

    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")

    # ------------------------------------------------------------------
    # Build datasets + dataloaders
    # ------------------------------------------------------------------
    print(f"\nCreating datasets (resolution={args.image_size}px).")
    train_ds = CUBImageDataset(
        image_dir=data_dir / "train",
        filenames=train_df["filename"].tolist(),
        labels=train_df["class_id"].tolist(),
        image_size=args.image_size,
    )
    val_ds = CUBImageDataset(
        image_dir=data_dir / "val",
        filenames=val_df["filename"].tolist(),
        labels=val_df["class_id"].tolist(),
        image_size=args.image_size,
    )
    test_ds = CUBImageDataset(
        image_dir=data_dir / "test",
        filenames=test_df["filename"].tolist(),
        labels=None,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Initialize MoCo feature extractor
    # ------------------------------------------------------------------
    feature_extractor = MoCoFeatureExtractor(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        dim=args.dim,
        K=args.K,
        m=args.m,
        T_moco=args.T_moco,
        mlp=not args.no_mlp,
        device=device,
    )

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    train_feats, train_labels, _ = extract_features_from_loader(
        feature_extractor, train_loader, "train"
    )
    val_feats, val_labels, _ = extract_features_from_loader(
        feature_extractor, val_loader, "val"
    )
    test_feats, _, test_filenames = extract_features_from_loader(
        feature_extractor, test_loader, "test"
    )

    # ------------------------------------------------------------------
    # MoCo-style kNN: eval on val, predict on test + create submission
    # ------------------------------------------------------------------
    train_labels_np = np.array(train_labels)
    val_labels_np   = np.array(val_labels)

    # 1) Hyperparam search on val
    k_values = [10, 20]
    T_values = [0.05, 0.07, 0.1, 0.2]

    best_acc = -1.0
    best_k = None
    best_T = None

    print("\n=== MoCo kNN hyperparam search on val set ===")
    for k in k_values:
        for T in T_values:
            print(f"\n>>> Evaluating k={k}, T={T}")
            acc = moco_knn_eval(
                train_features=train_feats,
                train_labels=train_labels_np,
                val_features=val_feats,
                val_labels=val_labels_np,
                k=k,
                T=T,
                device=device,
                chunk_size=1024,
            )
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_T = T

    print("\n=== Best hyperparams on val ===")
    print(f"  k = {best_k}, T = {best_T}, val acc = {best_acc:.4f} ({best_acc * 100:.2f}%)")

    # 2) Build feature bank on train+val (OK now that hyperparams are fixed)
    all_train_feats = np.concatenate([train_feats, val_feats], axis=0)
    all_train_labels = np.concatenate([train_labels_np, val_labels_np], axis=0)

    # 3) Predict on test with best (k, T)
    print("\nRunning MoCo kNN on test set with best hyperparams...")
    test_preds = moco_knn_predict(
        train_features=all_train_feats,
        train_labels=all_train_labels,
        query_features=test_feats,
        k=best_k,
        T=best_T,
        device=device,
        chunk_size=1024,
    )

    # 4) Create submission CSV
    create_submission_from_preds(test_filenames, test_preds, args.output)

    print("\nDone! Upload the submission file to Kaggle.")


if __name__ == "__main__":
    main()
