# mocomotion/utils/distributed.py

import os
from typing import Optional

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(args) -> None:
    """
    Initialize torch.distributed if running in distributed mode.

    This function:
      - sets args.distributed = True/False
      - sets args.gpu
      - initializes process group when needed

    On your Mac (CPU dev), this will almost always decide:
      args.distributed = False
      args.gpu = None
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched with torchrun / torch.distributed.run
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif getattr(args, "world_size", 1) > 1:
        # world_size > 1 but env vars not set: not handling this case here
        print("Warning: world_size > 1 but RANK/WORLD_SIZE not set in env; "
              "falling back to non-distributed mode.")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = None
        return
    else:
        # single-process, non-distributed
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = None
        print("Not using distributed mode.")
        return

    # If we got here, we have RANK and WORLD_SIZE: use distributed
    args.distributed = True

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
    else:
        # CPU-only fallback (e.g., Mac dev) – not ideal for real DDP but harmless
        device = torch.device("cpu")

    dist_backend = getattr(args, "dist_backend", "nccl")
    print(f"| distributed init (rank {args.rank}): backend={dist_backend}, gpu={args.gpu}")
    dist.init_process_group(
        backend=dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()

    # Just to keep a reference if needed
    args.device = device
