"""Safe performance defaults for CUDA and CPU training."""
from __future__ import annotations

import os
import torch


def configure_runtime(device: str = "auto", cpu_threads: int = 0,
                      amp: bool = True, compile_model: bool = False):
    """Configure backends without requiring CUDA or a large-memory GPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 4)))
    else:
        # Leave room for NumPy/NetworkX and avoid oversubscription.
        threads = min(os.cpu_count() or 1, 16)
        torch.set_num_threads(threads)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return torch.device(device), bool(amp and device.startswith("cuda")), compile_model


def maybe_compile(model, enabled: bool):
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception as exc:
        print(f"torch.compile disabled: {exc}")
        return model