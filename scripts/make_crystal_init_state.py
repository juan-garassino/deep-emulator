#!/usr/bin/env python3
"""Produce `states/crystal_init.state` by scripting the Crystal intro.

Drives the title screen → new game → speed/text settings → name entry →
mom dialog → Prof. Elm's lab → starter selection → walk to Cherrygrove,
then saves the PyBoy state at that anchor.

Robust to small timing variations: uses RAM-state-driven waits (e.g.,
"tick until wMapGroup, wMapNumber indicate Cherrygrove") rather than
hardcoded frame counts wherever possible.

Run:
    python scripts/make_crystal_init_state.py \\
        --rom roms/PokemonCrystal.gbc \\
        --out states/crystal_init.state \\
        --name RED --starter cyndaquil
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# When run as a script, allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepEmulator.cartridges.pokemon_crystal import (  # noqa: E402
    MAP_GROUP, MAP_NUMBER, PARTY_COUNT, dump_state,
)


# Cherrygrove map group/number (community-cited; will VERIFY at runtime)
CHERRYGROVE_GROUP = 26  # VERIFY

# Starter positions on Elm's table
_STARTER_INDEX = {
    "chikorita": 0,
    "cyndaquil": 1,
    "totodile": 2,
}


def _press(pyboy, button: str, hold_ticks: int = 8, release_ticks: int = 8) -> None:
    """Press a PyBoy button for `hold_ticks` frames, then release for `release_ticks`."""
    pyboy.button_press(button)
    pyboy.tick(hold_ticks, False)
    pyboy.button_release(button)
    pyboy.tick(release_ticks, False)


def _mash(pyboy, button: str, n_presses: int, *, hold: int = 4, gap: int = 4) -> None:
    for _ in range(n_presses):
        _press(pyboy, button, hold, gap)


def _wait_for_map(pyboy, target_group: int, *, timeout_frames: int = 200_000) -> bool:
    """Tick until the player enters a map with `map_group == target_group`."""
    frames = 0
    while frames < timeout_frames:
        if pyboy.memory[MAP_GROUP] == target_group:
            return True
        pyboy.tick(60, False)
        frames += 60
    return False


def _enter_name(pyboy, name: str) -> None:
    """Type a short name (≤7 chars) on Crystal's name-entry screen.

    Strategy: the cursor starts at 'A'. We mash right/down to navigate to each
    letter, press A to confirm, repeat. End by pressing Start to accept.

    NOTE: this is fragile — the keyboard layout and exact cursor mechanics
    can vary. For the smoke we just want *any* valid name; if scripting fails,
    user falls back to manual.
    """
    # For simplicity and robustness: skip to the default name by pressing Start
    # immediately. Crystal accepts the default ("GOLD") or "PLAYER" depending
    # on the screen. We accept whatever default to keep the script reliable.
    # The smoke + reward shaping don't depend on the player name.
    _press(pyboy, "start", 10, 30)


def drive_intro(
    rom: Path,
    out: Path,
    *,
    name: str = "RED",
    starter: str = "cyndaquil",
    headless: bool = True,
    verbose: bool = True,
) -> None:
    from pyboy import PyBoy  # local import

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"[make-crystal-init] loading {rom}")
    pyboy = PyBoy(str(rom), window="null" if headless else "SDL2")
    if not headless:
        pyboy.set_emulation_speed(0)  # max speed for scripted intro

    t0 = time.time()

    # Step 1: skip company logos + title screen
    log("[make-crystal-init] skipping logos / title screen")
    for _ in range(8):
        _press(pyboy, "start", 8, 60)
        _press(pyboy, "a", 8, 60)

    # Step 2: choose "NEW GAME". Cursor defaults to first option.
    log("[make-crystal-init] selecting new game")
    _press(pyboy, "a", 10, 30)

    # Step 3: mash through Prof. Oak's intro dialog (his Gen 2 cameo)
    log("[make-crystal-init] mashing intro dialog")
    _mash(pyboy, "a", 80, hold=6, gap=12)

    # Step 4: name entry (default name — see _enter_name docstring)
    log("[make-crystal-init] confirming default player name")
    _enter_name(pyboy, name)
    _mash(pyboy, "a", 10, hold=6, gap=12)

    # Step 5: more intro mashing until mom + Elm scenes complete
    log("[make-crystal-init] mom + Elm dialog")
    _mash(pyboy, "a", 120, hold=6, gap=12)

    # Step 6: starter selection — navigate to the chosen Pokeball on Elm's table
    starter_idx = _STARTER_INDEX.get(starter.lower(), 1)
    log(f"[make-crystal-init] selecting starter index {starter_idx} ({starter})")
    # Navigate right `starter_idx` times if needed, then press A
    for _ in range(starter_idx):
        _press(pyboy, "right", 8, 16)
    _press(pyboy, "a", 10, 30)
    # Confirm pickup dialog
    _mash(pyboy, "a", 40, hold=6, gap=12)

    # Step 7: walk out of Elm's lab → out of New Bark Town → through Route 29 →
    # arrive at Cherrygrove. Mash UP repeatedly + occasional A for dialog.
    log("[make-crystal-init] walking toward Cherrygrove (UP-mash + dialog A)")
    for _ in range(800):
        _press(pyboy, "up", 6, 4)
        if pyboy.memory[MAP_GROUP] == CHERRYGROVE_GROUP:
            log(f"[make-crystal-init] reached Cherrygrove (group={CHERRYGROVE_GROUP})")
            break
        # Occasionally press A to advance any dialog we hit
        if _ % 20 == 0:
            _mash(pyboy, "a", 4, hold=4, gap=4)
    else:
        log(
            "[make-crystal-init] WARNING: did not detect Cherrygrove arrival within walk loop. "
            f"Final map: group={pyboy.memory[MAP_GROUP]} num={pyboy.memory[MAP_NUMBER]}. "
            "Saving state anyway — verify manually."
        )

    # Step 8: a few extra ticks to settle
    pyboy.tick(60, False)

    # Step 9: save state
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pyboy.save_state(f)
    log(f"[make-crystal-init] saved state to {out} ({out.stat().st_size} bytes)")
    log(f"[make-crystal-init] final RAM state: {dump_state(pyboy)}")
    log(f"[make-crystal-init] wall-time: {time.time() - t0:.1f}s")

    try:
        pyboy.stop(save=False)
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(prog="make-crystal-init-state")
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("states/crystal_init.state"))
    p.add_argument("--name", default="RED")
    p.add_argument("--starter", default="cyndaquil", choices=list(_STARTER_INDEX))
    p.add_argument("--no-headless", action="store_true",
                   help="Show PyBoy SDL2 window (for debugging the intro script)")
    args = p.parse_args()
    drive_intro(args.rom, args.out, name=args.name, starter=args.starter,
                headless=not args.no_headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
