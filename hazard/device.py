"""Device selection and the memory guards that keep a 16 GB Mac responsive.

The failure mode on Apple Silicon is not an out-of-memory error, it is macOS
quietly swapping to SSD and handing you a twenty-minute beachball. The guards
here make PyTorch raise a clean, catchable OOM *before* the system starts
swapping. `apply_memory_guards()` has to run before torch is imported, because
the MPS allocator reads its watermark once at initialisation.
"""

from __future__ import annotations

import os
import platform

# Fraction of the GPU's "recommended max working set" that PyTorch may claim
# before it refuses. On a 16 GB machine the recommended max is around 10.6 GB,
# so 0.8 caps us near 8.5 GB and leaves the OS and your browser some air.
DEFAULT_HIGH_WATERMARK = "0.8"

# The level at which the allocator starts returning cached blocks to the system.
# It MUST be <= the high watermark: PyTorch ships a default of 1.4, so lowering
# only the high watermark throws "invalid low watermark ratio" at the first
# .to(device) call. Both have to move together.
DEFAULT_LOW_WATERMARK = "0.6"


def apply_memory_guards(
    high_watermark: str = DEFAULT_HIGH_WATERMARK,
    low_watermark: str = DEFAULT_LOW_WATERMARK,
) -> None:
    """Set the MPS env vars. Must be called before `import torch`."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", low_watermark)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", high_watermark)
    # Keep tokenizer / BLAS thread pools from fighting the GPU for cores.
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def pick_device(requested: str = "auto"):
    """Return a torch.device, preferring MPS on Apple Silicon."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_cache(device) -> None:
    """Release cached allocations. Cheap to call once per chunk, not per block."""
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def describe(device) -> str:
    """One line about what we are running on, for the log header."""
    import torch

    bits = [f"torch {torch.__version__}", f"device {device.type}"]
    if device.type == "mps":
        bits.append(f"{platform.machine()} / macOS {platform.mac_ver()[0]}")
        try:
            total = torch.mps.recommended_max_memory() / 1e9
            bits.append(f"MPS budget {total:.1f} GB")
        except Exception:
            pass
    return " | ".join(bits)


def memory_used_gb(device) -> float:
    """Current device allocation in GB, for the per-chunk log line."""
    import torch

    try:
        if device.type == "mps":
            return torch.mps.current_allocated_memory() / 1e9
        if device.type == "cuda":
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0
