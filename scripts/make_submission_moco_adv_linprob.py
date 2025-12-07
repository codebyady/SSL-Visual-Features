import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from moco.builder import MoCo
from utils.checkpoint import load_checkpoint


# ============================================================================
#                           MoCo ENCODER WRAPPER
# ============================================================================

class MoCoEncoder(nn.Module):
    """
    Wraps your MoCo model and exposes encoder_q as a frozen feature extractor.
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
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # Build MoCo model with same hyperparameters used in pretraining
        model = MoCo(
            backbone=backbone,
            dim=dim,
            K=K,
            m=m,
            T=T_moco,
            mlp=mlp,
        )

        # Load checkpoint in your format: ckpt["model"]
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        # Freeze params
        for p in model.parameters():
            p.requires_grad = False

        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] tensor on self.device
        Returns:
            feats: [B, D] tensor of L2-normalized features
        """
        m = self.model
        if hasattr(m, "module"):
            m = m.module
        encoder_q = m.encoder_q
        encoder_q.eval()

        feats = encoder_q(x)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats


# ============================================================================
#                       MULTI-VIEW FEATURE EXTRACTOR
# ============================================================================

class MultiViewFeatureExtractor:
    """
    Handles center-crop, 5-crop, and full TTA (center + flip + 5-crop)
    for a given MoCoEncoder.
    """

    def __init__(self, encoder: MoCoEncoder, resolution: int = 96):
        self.encoder = encoder
        self.device = encoder.device
        self.resolution = resolution

        self.center_transform = self._build_transform(center=True, flip=False)
        self.flip_transform = self._build_transform(center=True, flip=True)
        self.fivecrop_transform = self._build_fivecrop_transform()

    def _build_transform(self, center: bool = True, flip: bool = False):
        resize_size = int(self.resolution * 256 / 224)
        t_list = [T.Resize(resize_size)]
        if center:
            t_list.append(T.CenterCrop(self.resolution))
        if flip:
            t_list.append(T.RandomHorizontalFlip(p=1.0))
        t_list.extend(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return T.Compose(t_list)

    def _build_fivecrop_transform(self):
        resize_size = int(self.resolution * 256 / 224)
        return T.Compose(
            [
                T.Resize(resize_size),
                T.FiveCrop(self.resolution),
                T.Lambda(
                    lambda crops: torch.stack(
                        [
                            T.Compose(
                                [
                                    T.ToTensor(),
                                    T.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                    ),
                                ]
                            )(crop)
                            for crop in crops
                        ]
                    )
                ),
            ]
        )

    @torch.no_grad()
    def _encode_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        feats = self.encoder(batch_tensor)
        return feats.cpu().numpy()

    @torch.no_grad()
    def extract_center(self, pil_images):
        tensors = [self.center_transform(img) for img in pil_images]
        batch = torch.stack(tensors, dim=0)
        return self._encode_batch(batch)

    @torch.no_grad()
    def extract_flip(self, pil_images):
        tensors = [self.flip_transform(img) for img in pil_images]
        batch = torch.stack(tensors, dim=0)
        return self._encode_batch(batch)

    @torch.no_grad()
    def extract_fivecrop(self, pil_images):
        all_feats = []
        for img in pil_images:
            crops = self.fivecrop_transform(img)  # [5, 3, H, W]
            feats = self.encoder(crops.to(self.device))  # [5, D]
            avg_feat = feats.mean(dim=0, keepdim=True)  # [1, D]
            all_feats.append(avg_feat.cpu())
        all_feats = torch.cat(all_feats, dim=0)  # [B, D]
        return all_feats.numpy()

    @torch.no_grad()
    def extract(self, pil_images, mode: str = "center"):
        """
        mode: 'center', 'multicrop', or 'tta'
        """
        if mode == "center":
            return self.extract_center(pil_images)
        elif mode == "multicrop":
            return self.extract_fivecrop(pil_images)
        elif mode == "tta":
            feats_center = self.extract_center(pil_images)
            feats_flip = self.extract_flip(pil_images)
            feats_multi = self.extract_fivecrop(pil_images)
            return (feats_center + feats_flip + feats_multi) / 3.0
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
#                            DATASET & COLLATE
# ============================================================================

class CUBImageRawDataset(Dataset):
    """
    Returns PIL images + labels/filenames.
    Uses the Kaggle CUB layout with train/val/test and CSVs.
    """

    def __init__(self, image_dir, filenames, labels=None):
        self.image_dir = Path(image_dir)
        self.filenames = list(filenames)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = self.image_dir / img_name
        img = Image.open(img_path).convert("RGB")

        if self.labels is not None:
            label = self.labels[idx]
            return img, label, img_name
        else:
            return img, img_name


def collate_fn_train(batch):
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    filenames = [b[2] for b in batch]
    return images, labels, filenames


def collate_fn_test(batch):
    images = [b[0] for b in batch]
    filenames = [b[1] for b in batch]
    return images, filenames


# ============================================================================
#                      FEATURE EXTRACTION HELPERS
# ============================================================================

def sanitize_features(feats: np.ndarray) -> np.ndarray:
    return np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)


def l2_normalize(feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, eps)


def extract_features_split(
    extractor: MultiViewFeatureExtractor,
    dataloader: DataLoader,
    split_name: str,
    mode: str = "center",
    with_labels: bool = True,
):
    all_feats = []
    all_labels = []
    all_fnames = []

    print(f"\n[Extract] {split_name} features using mode='{mode}'")
    for batch in tqdm(dataloader, desc=split_name, unit="batch"):
        if with_labels:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch

        feats = extractor.extract(images, mode=mode)
        all_feats.append(feats)
        all_fnames.extend(filenames)

    feats = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels) if with_labels else None

    print(f"[Extract] {split_name}: {feats.shape[0]} x {feats.shape[1]}")
    return feats, labels, all_fnames


# ============================================================================
#                      C TUNING / LINEAR PROBE HELPERS
# ============================================================================

def tune_C_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_values,
    n_splits: int = 5,
    max_iter: int = 1000,
    solver: str = "lbfgs",
):
    print(f"\n[CV] Tuning C with {n_splits}-fold StratifiedKFold...")
    best_C = None
    best_score = -1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for C in C_values:
        scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            clf = LogisticRegression(
                penalty="l2",
                C=C,
                multi_class="multinomial",
                solver=solver,
                max_iter=max_iter,
                n_jobs=-1,
                verbose=0,
            )
            clf.fit(X_tr, y_tr)
            score = clf.score(X_val, y_val)
            scores.append(score)

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        print(f"  C={C:8.4f}: {mean_score:.4f} ± {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    print(f"\n[CV] Best C: {best_C} (mean val acc={best_score:.4f})")
    return best_C


def tune_C_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values,
    max_iter: int = 1000,
    solver: str = "lbfgs",
):
    print(f"\n[Linear] Tuning C on train/val split: {C_values}")
    best_C = None
    best_val_acc = -1.0

    for C in C_values:
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            multi_class="multinomial",
            solver=solver,
            max_iter=max_iter,
            n_jobs=-1,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  C={C:8.4f}: Val Acc = {val_acc:.4f} ({val_acc*100:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C

    print(
        f"\n[Linear] Best C: {best_C}, "
        f"Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)"
    )
    return best_C


def create_submission(test_feats, test_filenames, classifier, output_path: str):
    print("\n[Submit] Predicting on test set...")
    preds = classifier.predict(test_feats)

    df = pd.DataFrame({"id": test_filenames, "class_id": preds})
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"[Submit] Saved submission to: {output_path}")
    print("=" * 60)
    print(f"Total predictions: {len(df)}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nClass distribution (top 10):")
    print(df["class_id"].value_counts().head(10))

    assert list(df.columns) == ["id", "class_id"], "Columns must be ['id', 'class_id']"
    assert df.isnull().sum().sum() == 0, "NaNs found in submission!"
    print("\n[Submit] ✓ Submission format looks valid.")


# ============================================================================
#                                  MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Advanced MoCo linear probe with TTA + C tuning for CUB Kaggle"
    )

    # Data & paths
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_latest.pth",
        help="Path to pretrained MoCo checkpoint",
    )
    p.add_argument(
        "--output",
        type=str,
        default="submission_moco_linear_advanced.csv",
        help="Output CSV path",
    )

    # MoCo / model params (must match training)
    p.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
    )
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--K", type=int, default=65536)
    p.add_argument("--m", type=float, default=0.999)
    p.add_argument("--T-moco", type=float, default=0.2)
    p.add_argument(
        "--no-mlp",
        action="store_true",
        help="Set if you trained MoCo WITHOUT MLP head",
    )

    # Eval / feature extraction
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--mode",
        type=str,
        default="center",
        choices=["center", "multicrop", "tta"],
        help="Feature extraction mode",
    )

    # Preprocessing
    p.add_argument(
        "--use-power-transform",
        action="store_true",
        help="Apply PowerTransformer to features",
    )
    p.add_argument(
        "--use-standard-scaler",
        action="store_true",
        help="Apply StandardScaler to features",
    )

    # C tuning
    p.add_argument(
        "--use-cv",
        action="store_true",
        help="Use cross-validation on train to tune C",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--C",
        type=float,
        default=None,
        help="Single C (if set and --C-values not used)",
    )
    p.add_argument(
        "--C-values",
        type=float,
        nargs="*",
        default=None,
        help="List of C values to tune over",
    )

    # Classifier
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "saga"],
    )

    # Device
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for feature extraction: 'cuda' or 'cpu'",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Device] cuda requested but not available; using cpu.")
        device = "cpu"
    print(f"[Setup] Using device: {device}")

    data_dir = Path(args.data-dir if hasattr(args, "data-dir") else args.data_dir)

    # ------------------------------------------------------------------
    # Load metadata CSVs
    # ------------------------------------------------------------------
    print("\n[Data] Loading CUB CSVs...")
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    test_df = pd.read_csv(data_dir / "test_images.csv")

    img_col = "filename"
    label_col = "class_id"

    print(f"[Data] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"[Data] Classes: {train_df[label_col].nunique()}")

    # ------------------------------------------------------------------
    # Build datasets & dataloaders (PIL images)
    # ------------------------------------------------------------------
    train_ds = CUBImageRawDataset(
        image_dir=data_dir / "train",
        filenames=train_df[img_col].tolist(),
        labels=train_df[label_col].tolist(),
    )
    val_ds = CUBImageRawDataset(
        image_dir=data_dir / "val",
        filenames=val_df[img_col].tolist(),
        labels=val_df[label_col].tolist(),
    )
    test_ds = CUBImageRawDataset(
        image_dir=data_dir / "test",
        filenames=test_df[img_col].tolist(),
        labels=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_test,
    )

    # ------------------------------------------------------------------
    # Initialize encoder + multiview extractor
    # ------------------------------------------------------------------
    encoder = MoCoEncoder(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        dim=args.dim,
        K=args.K,
        m=args.m,
        T_moco=args.T_moco,
        mlp=not args.no_mlp,
        device=device,
    )
    extractor = MultiViewFeatureExtractor(
        encoder=encoder, resolution=args.image_size
    )

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    train_feats, train_labels, _ = extract_features_split(
        extractor, train_loader, "train", mode=args.mode, with_labels=True
    )
    val_feats, val_labels, _ = extract_features_split(
        extractor, val_loader, "val", mode=args.mode, with_labels=True
    )
    test_feats, _, test_filenames = extract_features_split(
        extractor, test_loader, "test", mode=args.mode, with_labels=False
    )

    # ------------------------------------------------------------------
    # Sanitize + L2 normalize
    # ------------------------------------------------------------------
    print("\n[Preprocess] Sanitizing + L2 normalizing...")
    train_feats = l2_normalize(sanitize_features(train_feats))
    val_feats = l2_normalize(sanitize_features(val_feats))
    test_feats = l2_normalize(sanitize_features(test_feats))

    # ------------------------------------------------------------------
    # Optional feature preprocessing (Power / Standard scaling)
    # ------------------------------------------------------------------
    scaler = None
    if args.use_power_transform:
        print("[Preprocess] Applying PowerTransformer...")
        scaler = PowerTransformer()
        train_feats = scaler.fit_transform(train_feats)
        val_feats = scaler.transform(val_feats)
        test_feats = scaler.transform(test_feats)
    elif args.use_standard_scaler:
        print("[Preprocess] Applying StandardScaler...")
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        val_feats = scaler.transform(val_feats)
        test_feats = scaler.transform(test_feats)

    train_labels = np.array(train_labels, dtype=np.int64)
    val_labels = np.array(val_labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # Determine C grid
    # ------------------------------------------------------------------
    if args.C_values is not None:
        C_values = args.C_values
        print(f"\n[Linear] Using user-provided C grid: {C_values}")
    else:
        # Reasonable default for 200-class CUB
        C_values = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        print(f"\n[Linear] Using default C grid: {C_values}")

    # ------------------------------------------------------------------
    # Tune C (CV or simple val)
    # ------------------------------------------------------------------
    if args.use_cv:
        best_C = tune_C_with_cv(
            train_feats,
            train_labels,
            C_values,
            n_splits=args.cv_folds,
            max_iter=args.max_iter,
            solver=args.solver,
        )
    else:
        best_C = tune_C_on_val(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            C_values,
            max_iter=args.max_iter,
            solver=args.solver,
        )

    if best_C is None:
        best_C = args.C if args.C is not None else 1.0
        print(f"[Linear] Fallback: using C={best_C}")

    # ------------------------------------------------------------------
    # Final model on TRAIN + VAL
    # ------------------------------------------------------------------
    print(f"\n[Linear] Training final model on TRAIN+VAL with C={best_C}...")
    all_feats = np.concatenate([train_feats, val_feats], axis=0)
    all_labels = np.concatenate([train_labels, val_labels], axis=0)

    final_clf = LogisticRegression(
        penalty="l2",
        C=best_C,
        multi_class="multinomial",
        solver=args.solver,
        max_iter=args.max_iter,
        n_jobs=-1,
        verbose=1,
    )
    final_clf.fit(all_feats, all_labels)

    # Sanity check
    train_val_pred = final_clf.predict(all_feats)
    train_val_acc = accuracy_score(all_labels, train_val_pred)
    print(
        f"[Linear] Train+Val Accuracy: {train_val_acc:.4f} ({train_val_acc*100:.2f}%)"
    )

    # ------------------------------------------------------------------
    # Predict on TEST and save submission
    # ------------------------------------------------------------------
    create_submission(test_feats, test_filenames, final_clf, args.output)

    print("\nDone! Upload the submission file to Kaggle.")


if __name__ == "__main__":
    main()
