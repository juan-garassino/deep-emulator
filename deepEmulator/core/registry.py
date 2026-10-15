"""Cartridge registry — maps cartridge_title (or override id) to adapter class."""
from __future__ import annotations

from typing import Callable, Type

from deepEmulator.core.cartridge import CartridgeAdapter

_REGISTRY: dict[str, Type[CartridgeAdapter]] = {}


def register(cartridge_title: str) -> Callable[[Type[CartridgeAdapter]], Type[CartridgeAdapter]]:
    """Class decorator registering an adapter under `cartridge_title`.

    Re-registering a title replaces the previous class.
    """
    def deco(cls: Type[CartridgeAdapter]) -> Type[CartridgeAdapter]:
        _REGISTRY[cartridge_title] = cls
        return cls
    return deco


def get(cartridge_title: str) -> Type[CartridgeAdapter]:
    """Return the adapter class for `cartridge_title`; raise `KeyError` listing the registered titles if absent."""
    if cartridge_title not in _REGISTRY:
        raise KeyError(f"no cartridge registered for {cartridge_title!r}; have {list(_REGISTRY)}")
    return _REGISTRY[cartridge_title]


def available() -> list[str]:
    """Return the sorted list of registered cartridge titles."""
    return sorted(_REGISTRY)
