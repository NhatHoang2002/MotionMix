"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp4

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = -1

def setup_dist(device=0, distributed=False):
    """
    Setup a distributed process group.
    """
    global used_device
    if not distributed:
        used_device = device
    else:
        dist.init_process_group(backend="nccl", init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if dist.is_initialized():
        return th.device(f"cuda:{dist.get_rank()}")
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")

def get_dist_info():
    global used_device
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = used_device
        world_size = 1
    return rank, world_size

def is_main_process():
    return (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
