"""Pokemon Coral cartridge adapter — minimal subclass of PokemonCrystalAdapter.

Coral is a fan-made romhack built on Pokemon Crystal's engine, so the Gen 2
RAM layout (party struct, badges, map_group/map_number, event flags) is
shared. Same reward shaping applies. Cartridge title in the ROM header is
`POKECORALV1PKC`.

This file is intentionally tiny — it's the **flexibility test**: does the
architecture support a new Gen 2 game via subclass + title rebind, without
re-doing the RAM map? Answer: yes.

If Coral diverges from Crystal in some specific way (e.g., adds badges, moves
event flags), override the relevant constants here. For now we assume they're
identical.
"""
from __future__ import annotations

from dataclasses import dataclass

from deepEmulator.cartridges.pokemon_crystal import PokemonCrystalAdapter
from deepEmulator.core.registry import register


@dataclass
class PokemonCoralAdapter(PokemonCrystalAdapter):
    cartridge_title: str = "POKEMON CORAL"


register("POKEMON CORAL")(PokemonCoralAdapter)
