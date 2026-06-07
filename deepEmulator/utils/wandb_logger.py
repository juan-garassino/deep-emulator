"""Graceful-degradation wandb wrapper.

Training scripts call this instead of `wandb` directly so a missing key,
missing dep, or wandb-side network failure never crashes a multi-hour run.

`entrypoint.sh` sets the env-var contract:
    - WANDB_API_KEY  → presence triggers `wandb login` before training.
    - WANDB_MODE     → `disabled` forces this wrapper to no-op even if installed.
    - WANDB_PROJECT  → default project name (we do not override here).
    - WANDB_NAME     → default run name (we do not override here).
"""
from __future__ import annotations

import os
import sys
from typing import Any


class WandbLogger:
    """Wrap wandb.{init,log,finish} behind a graceful interface.

    Any side-effect failure (ImportError, network, auth) degrades the logger
    to a no-op for the rest of the run rather than raising.
    """

    def __init__(self, **init_kwargs: Any) -> None:
        self._run = None
        mode = os.environ.get("WANDB_MODE", "").lower()
        if mode == "disabled":
            return
        # Without an explicit mode AND without an API key, wandb.init() would
        # block waiting for interactive auth — disastrous in CI and test envs.
        # No credentials + no mode override → stay silent.
        if not os.environ.get("WANDB_API_KEY") and mode not in (
            "online", "offline", "shared", "dryrun"
        ):
            return
        try:
            import wandb  # type: ignore
        except Exception as exc:
            print(f"[wandb] import failed (logging disabled): {exc}", file=sys.stderr)
            return
        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            print(f"[wandb] init failed (logging disabled): {exc}", file=sys.stderr)
            self._run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            self._run.log(data, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            import wandb  # type: ignore

            wandb.finish()
        except Exception:
            pass
        self._run = None
