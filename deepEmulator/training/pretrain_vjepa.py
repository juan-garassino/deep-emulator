"""V-JEPA pretraining — Phase 1 placeholder.

The `deepemu-pretrain-vjepa` CLI is registered ahead of the implementation so
the entrypoint/Makefile contract is stable. Until the encoder modules land
(`encoders/vjepa.py`, `vit_spatiotemporal.py`, `sequence_augmentations.py`),
this stub exits with a clear message instead of a raw ModuleNotFoundError —
critically BEFORE any caller stages gigabytes of corpus from GCS
(entrypoint.sh checks `IMPLEMENTED` first).
"""
from __future__ import annotations

import sys

IMPLEMENTED = False


def main(argv: list[str] | None = None) -> int:
    print(
        "[pretrain-vjepa] V-JEPA is Phase 1 — not implemented yet. "
        "Use deepemu-pretrain-dino / MODE=pretrain_dino instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
