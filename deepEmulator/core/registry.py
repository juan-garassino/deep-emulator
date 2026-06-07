"""Cartridge registry — maps cartridge_title (or override id) to adapter class."""
from __future__ import annotations

from typing import Type

from deepEmulator.core.cartridge import CartridgeAdapter

_REGISTRY: dict[str, Type[CartridgeAdapter]] = {}


def register(cartridge_title: str):
    def deco(cls: Type[CartridgeAdapter]):
        _REGISTRY[cartridge_title] = cls
        return cls
    return deco


def get(cartridge_title: str) -> Type[CartridgeAdapter]:
    if cartridge_title not in _REGISTRY:
        raise KeyError(f"no cartridge registered for {cartridge_title!r}; have {list(_REGISTRY)}")
    return _REGISTRY[cartridge_title]


def available() -> list[str]:
    return sorted(_REGISTRY)
