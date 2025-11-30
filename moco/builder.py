# mocomotion/moco/builder.py

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import build_resnet


class MoCo(nn.Module):
    """
    Minimal MoCo v2 implementation (single-GPU friendly).

    - encoder_q: query encoder
    - encoder_k: key encoder (momentum updated)
    - proj_q / proj_k: projection MLPs
    - queue: memory bank of negative keys

    Forward:
        im_q, im_k -> logits, labels
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.2,
        mlp: bool = True,
    ):
        """
        Args:
            backbone: resnet18 / resnet34 / resnet50
            dim: feature dim of the projection head
            K: queue size (number of negatives)
            m: momentum for updating key encoder
            T: softmax temperature
            mlp: if True, use 2-layer MLP for projection (MoCo v2 style)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # encoder_q and encoder_k are the same architecture
        encoder_q, feat_dim = build_resnet(backbone)
        encoder_k, _ = build_resnet(backbone)

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # projection heads (MLP or linear)
        if mlp:
            self.proj_q = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            )
            self.proj_k = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            )
        else:
            self.proj_q = nn.Linear(feat_dim, dim)
            self.proj_k = nn.Linear(feat_dim, dim)

        # initialize key encoder weights to match query encoder
        self._init_key_encoder()

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)  # normalize along feature dim

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_key_encoder(self) -> None:
        """
        Initialize key encoder as a copy of query encoder, and
        set requires_grad=False for all key encoder params.
        """
        # copy encoder weights
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # copy projection head weights
        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder:

            theta_k = m * theta_k + (1 - m) * theta_q
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Update the queue:

        - keys: [batch_size, dim] tensor
        - queue: [dim, K] tensor
        """
        keys = keys.detach()
        batch_size = keys.shape[0]

        K = self.K
        ptr = int(self.queue_ptr)

        assert (
            batch_size <= K
        ), f"Batch size {batch_size} must be <= queue size {K}"

        # if batch doesn't divide K, last few entries will be overwritten a bit unevenly;
        # that's fine for a first implementation.
        end = ptr + batch_size
        if end <= K:
            self.queue[:, ptr:end] = keys.T
        else:
            # wrap-around
            first_part = K - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, : end - K] = keys[first_part:].T

        ptr = (ptr + batch_size) % K
        self.queue_ptr[0] = ptr

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            im_q: a batch of query images  [B, 3, H, W]
            im_k: a batch of key images    [B, 3, H, W]

        Output:
            logits: [B, 1 + K]
            labels: [B] (all zeros, index of positive key)
        """
        # compute query features
        q_feat = self.encoder_q(im_q)          # [B, feat_dim]
        q = self.proj_q(q_feat)                # [B, dim]
        q = F.normalize(q, dim=1)              # L2-normalize

        # compute key features
        with torch.no_grad():
            # update key encoder
            self._momentum_update_key_encoder()

            k_feat = self.encoder_k(im_k)      # [B, feat_dim]
            k = self.proj_k(k_feat)            # [B, dim]
            k = F.normalize(k, dim=1)

        # positive logits: dot product between q and k
        l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)  # [B, 1]

        # negative logits: dot product between q and all entries in the queue
        # queue: [dim, K] -> q @ queue = [B, K]
        l_neg = torch.einsum("bd,dk->bk", [q, self.queue.clone().detach()])

        # logits: [B, 1 + K]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits = logits / self.T

        # labels: positives are the first logit
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # update the queue with current keys
        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        return logits, labels


if __name__ == "__main__":
    # Quick sanity check: random forward pass
    model = MoCo(
        backbone="resnet50",
        dim=128,
        K=1024,     # smaller queue for local test
        m=0.999,
        T=0.2,
        mlp=True,
    )

    model.eval()
    x_q = torch.randn(4, 3, 96, 96)
    x_k = torch.randn(4, 3, 96, 96)
    with torch.no_grad():
        logits, labels = model(x_q, x_k)
    print("logits shape:", logits.shape)  # expect [4, 1 + 1024]
    print("labels shape:", labels.shape)  # expect [4]
