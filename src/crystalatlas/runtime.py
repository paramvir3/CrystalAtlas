"""Deterministic local and distributed runtime setup."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._optional import require_torch
from .errors import DomainError


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Process identity read from torchrun-compatible environment variables."""

    world_size: int
    rank: int
    local_rank: int

    @classmethod
    def from_environment(cls) -> DistributedContext:
        context = cls(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        )
        if context.world_size < 1:
            raise DomainError("WORLD_SIZE must be positive")
        if not 0 <= context.rank < context.world_size:
            raise DomainError("RANK lies outside WORLD_SIZE")
        if context.local_rank < 0:
            raise DomainError("LOCAL_RANK must be nonnegative")
        return context


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed Python, NumPy, and—when installed—PyTorch.

    The returned NumPy generator should be passed explicitly to exact samplers;
    global state is seeded only for third-party code that cannot accept a
    generator.
    """

    if seed < 0:
        raise DomainError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)
    try:
        torch = require_torch()
    except ImportError:
        return generator
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return generator


def distributed_seed(
    root_seed: int,
    context: DistributedContext,
    *,
    worker_id: int = 0,
) -> int:
    """Derive a reproducible, rank/worker-separated unsigned 32-bit seed."""

    if root_seed < 0 or worker_id < 0:
        raise DomainError("root_seed and worker_id must be nonnegative")
    sequence = np.random.SeedSequence([root_seed, context.rank, worker_id])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def initialize_distributed(context: DistributedContext | None = None) -> Any:
    """Initialize torch.distributed only when more than one process is requested."""

    context = context or DistributedContext.from_environment()
    if context.world_size == 1:
        return None
    torch = require_torch()
    distributed = torch.distributed
    if not distributed.is_available():
        raise RuntimeError("PyTorch was built without distributed support")
    if torch.cuda.is_available():
        if context.local_rank >= torch.cuda.device_count():
            raise DomainError("LOCAL_RANK lies outside the visible CUDA devices")
        torch.cuda.set_device(context.local_rank)
    if not distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend=backend, init_method="env://")
    return distributed
