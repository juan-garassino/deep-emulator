"""Device selection — the single source of truth for `mps -> cuda -> cpu`.

Juan trains on an M-series Mac (MPS) locally and on RTX / RunPod (CUDA)
remotely, so NO torch code in this repo may hardcode `.cuda()`. Everything
routes through `get_device()`:

    from deepEmulator.utils.device import get_device
    device = get_device()          # torch.device, best available
    net = net.to(device)

Preference order (best available wins):
    1. MPS  — Apple-Silicon Metal backend (local M-series dev)
    2. CUDA — NVIDIA (RTX box / RunPod pods)
    3. CPU  — universal fallback (CI, x86 macOS, anything else)

`get_device()` returns a `torch.device`; `get_device_str()` returns the plain
string the DDQN agent / training scripts thread through (they store the device
as a `str`). An explicit `prefer=` override always wins when that backend is
actually usable — otherwise it degrades to the normal preference chain.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _mps_ok() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:  # torch missing / older build without the mps namespace
        return False


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_device_str(prefer: str | None = None) -> str:
    """Return the best available torch device as a plain string.

    `prefer` (one of "mps" / "cuda" / "cpu") is honored iff that backend is
    actually usable on this host; otherwise the standard mps -> cuda -> cpu
    chain applies. This keeps a RunPod override (`prefer="cuda"`) from crashing
    a laptop run when CUDA isn't there.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "mps" and _mps_ok():
            return "mps"
        if prefer == "cuda" and _cuda_ok():
            return "cuda"
        if prefer == "cpu":
            return "cpu"
        logger.warning("preferred device %r unavailable — falling back to auto-select", prefer)

    if _mps_ok():
        return "mps"
    if _cuda_ok():
        return "cuda"
    return "cpu"


def get_device(prefer: str | None = None):
    """Return the best available `torch.device` (mps -> cuda -> cpu).

    See `get_device_str` for the `prefer` semantics. Importing torch lazily so
    modules that only need the helper's *name* (docs, CLI wiring) don't pay the
    torch import at module load.
    """
    import torch

    return torch.device(get_device_str(prefer))


def is_cuda(device: object) -> bool:
    """True iff `device` (a torch.device or str) targets CUDA.

    Guards the CUDA-only fast paths (AMP autocast, GradScaler) so they stay
    no-ops on MPS/CPU instead of raising.
    """
    return str(device).startswith("cuda")
