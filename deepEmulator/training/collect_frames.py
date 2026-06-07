"""Frame corpus collector CLI.

Builds one or more envs (via the cartridge registry), runs them under a
random or near-greedy trained policy, and writes a `FrameStorage` corpus
ready for `deepemu-pretrain-dino`.

    deepemu-collect-frames \\
        --cartridges "POKEMON RED" \\
        --rom roms/PokemonRed.gb \\
        --init-state states/init.state \\
        --frames 5000 \\
        --out data/frames/pokemon_local

Multi-cartridge mix (round-robin sampling):

    deepemu-collect-frames \\
        --cartridges "POKEMON RED" "ATARI PONG" \\
        --rom roms/PokemonRed.gb roms/pong.bin \\
        --frames 50000 --out data/frames/multi
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from deepEmulator.data.frame_corpus import FrameCollector, FrameStorage


def _import_cartridge_modules() -> None:
    for mod in (
        "deepEmulator.cartridges.pokemon_red",
        "deepEmulator.cartridges.atari.pong",
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            pass


def _build_env(cartridge: str, rom: Path | None, init_state: Path | None):
    """Build an env for the given cartridge using the registry.

    Dispatches on `platform` declared by the registered adapter.
    """
    from deepEmulator.core import registry

    AdapterCls = registry.get(cartridge)
    adapter = AdapterCls(init_state=init_state) if init_state else AdapterCls()
    platform = getattr(adapter, "platform", "gameboy")
    if platform == "gameboy":
        from deepEmulator.platforms.gameboy import PyBoyEnv

        return PyBoyEnv(adapter, rom_path=rom, init_state=init_state, headless=True)
    if platform == "atari":
        from deepEmulator.platforms.atari import AtariEnv

        return AtariEnv(adapter, rom_path=rom, headless=True)
    raise ValueError(f"unsupported platform {platform!r} for cartridge {cartridge!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-collect-frames")
    p.add_argument(
        "--cartridges",
        nargs="+",
        required=True,
        help='One or more registered cartridge titles, e.g. "POKEMON RED" "ATARI PONG"',
    )
    p.add_argument(
        "--rom",
        type=Path,
        nargs="+",
        default=None,
        help="One ROM path per cartridge (in the same order). Atari ROMs are optional if "
        "ale-py bundles them.",
    )
    p.add_argument("--init-state", type=Path, default=None, help="PyBoy save-state (GB only)")
    p.add_argument("--frames", type=int, default=5_000)
    p.add_argument("--out", type=Path, required=True, help="FrameStorage root directory")
    p.add_argument("--chunk-size", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _import_cartridge_modules()

    roms = list(args.rom) if args.rom else [None] * len(args.cartridges)
    if len(roms) < len(args.cartridges):
        roms += [None] * (len(args.cartridges) - len(roms))

    envs = []
    for cart, rom in zip(args.cartridges, roms):
        envs.append(_build_env(cart, rom, args.init_state))
    print(f"[collect-frames] built {len(envs)} envs: {', '.join(args.cartridges)}")

    storage = FrameStorage(args.out, chunk_size=args.chunk_size)
    collector = FrameCollector(envs, storage, seed=args.seed)

    t0 = time.time()
    last_report = t0
    for i in range(args.frames):
        collector.step()
        now = time.time()
        if now - last_report > 5.0:
            rate = (i + 1) / (now - t0)
            print(f"  {i + 1:>7d}/{args.frames} frames | {rate:.0f}/s")
            last_report = now
    storage.flush()
    elapsed = time.time() - t0

    for env in envs:
        try:
            env.close()
        except Exception:
            pass

    print(
        f"[collect-frames] done: {storage.total_frames:,} frames in {elapsed:.1f}s "
        f"({storage.total_frames / max(elapsed, 1e-3):.0f}/s) at {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
