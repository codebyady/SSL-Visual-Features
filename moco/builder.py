# mocomotion/moco/builder.py

import torch.nn as nn

class MoCo(nn.Module):
    """Stub: MoCo model (encoder_q, encoder_k, queue, etc.)."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, im_q, im_k):
        raise NotImplementedError("MoCo.forward not implemented yet.")
