"""Cartridge adapters. `load_all()` populates the registry with every shipped adapter."""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_CARTRIDGE_MODULES = (
    "deepEmulator.cartridges.pokemon_red",
    "deepEmulator.cartridges.pokemon_crystal",
    "deepEmulator.cartridges.pokemon_coral",
    "deepEmulator.cartridges.generic_gb",
    "deepEmulator.cartridges.atari.pong",
)


def load_all() -> list[str]:
    """Import every shipped cartridge module so the registry is populated.

    Idempotent (imports are cached). A module that fails to import (e.g. a
    missing optional dependency like ale-py) is logged and skipped — never
    silently swallowed.
    """
    for mod in _CARTRIDGE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            logger.warning("cartridge module %s failed to import — skipping", mod, exc_info=True)
    from deepEmulator.core import registry

    return registry.available()
