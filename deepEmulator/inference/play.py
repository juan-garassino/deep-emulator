"""Local visible inference: load a DDQN bundle, play with a visible PyBoy window.

Optional live overlays (pygame, in `[viz]` extra):
- `--arrows`: secondary pygame surface showing the cumulative trajectory as
  direction-colored arrows on a small grid.
- `--attention`: secondary pygame surface showing the encoder's CLS-attention
  rollout alpha-blended onto the current frame (only if the bundle's
  metadata records an `encoder`).

Human-takeover toggle: while running, edit `agent_enabled.txt` and set its
contents to `0` to pause the agent (you keep the visible PyBoy window and can
play manually), `1` to resume.

    deepemu-play \\
        --cartridge "POKEMON RED" \\
        --rom roms/PokemonRed.gb \\
        --init-state states/init.state \\
        --ckpt checkpoints/pokemon_red/<run> \\
        --visible --arrows --attention
"""
from __future__ import annotations

import argparse
import importlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepEmulator.agents.ddqn_torch import DDQNAgent
from deepEmulator.cartridges import load_all as _load_cartridges
from deepEmulator.utils.checkpoints import load_bundle


def _build_inner_env(cartridge: str, rom: Path, init_state: Path | None, visible: bool):
    from deepEmulator.core import registry

    AdapterCls = registry.get(cartridge)
    adapter = AdapterCls(init_state=init_state) if init_state else AdapterCls()
    platform = getattr(adapter, "platform", "gameboy")
    if platform == "gameboy":
        from deepEmulator.platforms.gameboy import PyBoyEnv

        return PyBoyEnv(adapter, rom_path=rom, init_state=init_state, headless=not visible)
    if platform == "atari":
        from deepEmulator.platforms.atari import AtariEnv

        return AtariEnv(adapter, rom_path=rom, headless=not visible)
    raise ValueError(f"unsupported platform {platform!r}")


# --- live overlays ---------------------------------------------------------
class _ArrowsOverlay:
    """pygame surface that accumulates trajectory arrows in a small grid."""

    def __init__(self, cell_size: int = 8, grid_radius: int = 32, title: str = "arrows"):
        import pygame  # type: ignore

        self.pygame = pygame
        self.cell_size = cell_size
        self.grid_radius = grid_radius
        side = (2 * grid_radius + 1) * cell_size
        self.surface = pygame.display.set_mode((side, side))
        pygame.display.set_caption(title)
        self.flows: dict[tuple[int, int], np.ndarray] = {}
        self._prev: tuple[int, int, int] | None = None
        self._center: tuple[int, int] | None = None

    def update(self, x: int, y: int, map_id: int) -> None:
        # Use the first sample's map_id as origin; ignore cross-map jumps for now
        if self._prev is None:
            self._prev = (x, y, map_id)
            self._center = (x, y)
            return
        px, py, pm = self._prev
        if pm == map_id and abs(x - px) + abs(y - py) <= 2:
            key = (py, px)
            diff = np.array([y - py, x - px], dtype=np.int32)
            if key in self.flows:
                self.flows[key] = self.flows[key] + diff
            else:
                self.flows[key] = diff
        self._prev = (x, y, map_id)

    def draw(self) -> None:
        cs = self.cell_size
        side = (2 * self.grid_radius + 1) * cs
        self.surface.fill((20, 20, 20))
        if self._center is None:
            self.pygame.display.flip()
            return
        cx, cy = self._center
        for (gy, gx), vec in self.flows.items():
            rx = gx - cx + self.grid_radius
            ry = gy - cy + self.grid_radius
            if not (0 <= rx <= 2 * self.grid_radius and 0 <= ry <= 2 * self.grid_radius):
                continue
            dy, dx = float(vec[0]), float(vec[1])
            if dx == 0 and dy == 0:
                continue
            angle = math.atan2(-dy, dx)
            hue = 0.5 * angle / math.pi + 0.5
            color = _hsv_to_rgb_255(hue)
            x0 = rx * cs + cs // 2
            y0 = ry * cs + cs // 2
            x1 = int(x0 + math.cos(angle) * cs * 0.45)
            y1 = int(y0 - math.sin(angle) * cs * 0.45)
            self.pygame.draw.line(self.surface, color, (x0, y0), (x1, y1), 2)
        # Mark current position
        x0 = self.grid_radius * cs + cs // 2
        y0 = self.grid_radius * cs + cs // 2
        self.pygame.draw.circle(self.surface, (255, 255, 255), (x0, y0), max(2, cs // 4))
        self.pygame.display.flip()


def _hsv_to_rgb_255(h: float, s: float = 0.85, v: float = 0.95) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


class _AttentionOverlay:
    """pygame surface showing attention rollout alpha-blended on the frame."""

    def __init__(self, encoder: Any, alpha: float = 0.55, title: str = "attention"):
        import pygame  # type: ignore

        self.pygame = pygame
        self.encoder = encoder
        self.alpha = alpha
        # Reserve a 160×144-ish surface; will resize on first draw if needed
        self.surface = pygame.display.set_mode((320, 288))
        pygame.display.set_caption(title)
        self._size = (320, 288)

    def draw(self, frame_rgb: np.ndarray) -> None:
        from deepEmulator.visualization.attention import attention_for_frame

        overlay, _ = attention_for_frame(self.encoder, frame_rgb, alpha=self.alpha, device="cpu")
        # Upscale 2x for visibility
        H, W, _ = overlay.shape
        target = (W * 2, H * 2)
        if target != self._size:
            self.surface = self.pygame.display.set_mode(target)
            self._size = target
        # pygame surfarray expects (W, H, 3), so transpose
        arr = np.transpose(overlay, (1, 0, 2))
        # Nearest-neighbor 2x upscale
        big = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        self.pygame.surfarray.blit_array(self.surface, big)
        self.pygame.display.flip()


# --- agent_enabled toggle --------------------------------------------------
def _toggle_path(run_dir: Path) -> Path:
    return run_dir / "agent_enabled.txt"


def _read_agent_enabled(toggle: Path, default: bool = True) -> bool:
    if not toggle.exists():
        return default
    try:
        return toggle.read_text().strip() not in ("0", "false", "False", "")
    except Exception:
        return default


# --- CLI -------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-play")
    p.add_argument("--cartridge", required=True)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--init-state", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, required=True, help="Trained DDQN bundle dir")
    p.add_argument("--visible", action="store_true", help="Open PyBoy SDL2 window")
    p.add_argument("--arrows", action="store_true", help="Live arrow trajectory overlay window")
    p.add_argument(
        "--attention",
        action="store_true",
        help="Live attention heatmap overlay (only if bundle has an encoder)",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=4096)
    p.add_argument("--epsilon", type=float, default=0.05)
    return p.parse_args(argv)


def run_play(
    *,
    cartridge: str,
    rom: Path,
    init_state: Path | None,
    ckpt: Path,
    visible: bool,
    arrows: bool,
    attention: bool,
    episodes: int,
    max_episode_steps: int,
    epsilon: float,
    env_factory=None,  # for tests: inject a fake env factory
) -> int:
    _load_cartridges()

    agent_state, metadata = load_bundle(ckpt)
    encoder_path = None
    if "encoder" in metadata and metadata["encoder"].get("path"):
        encoder_path = Path(metadata["encoder"]["path"])

    # Build env (real or injected)
    if env_factory is not None:
        env = env_factory()
    else:
        env = _build_inner_env(cartridge, rom, init_state, visible=visible)
        if encoder_path is not None:
            from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder

            encoder, _ = load_frozen_encoder(encoder_path)
            env = FrozenEncoderEnv(env, encoder)

    agent = DDQNAgent(
        obs_shape=tuple(metadata["observation_shape"]),
        n_actions=len(metadata["action_set"]),
        device="cpu",
    )
    agent.load_state_dict(agent_state)
    # fixed inference epsilon — no decay, no 0.05 floor clamp (act(explore=False))
    agent.eval_epsilon = epsilon

    # Optional overlays
    arrows_overlay = None
    attention_overlay = None
    pygame = None
    if arrows or (attention and encoder_path is not None):
        pygame = importlib.import_module("pygame")
        pygame.init()
    if arrows:
        arrows_overlay = _ArrowsOverlay(title=f"{cartridge} arrows")
    if attention:
        if encoder_path is None:
            print("[deepemu-play] --attention requested but bundle has no encoder; skipping")
        else:
            # Load the encoder for the overlay (separate from the agent's env wrapper)
            from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder

            enc, _ = load_frozen_encoder(encoder_path)
            attention_overlay = _AttentionOverlay(enc, title=f"{cartridge} attention")

    toggle = _toggle_path(ckpt)
    print(f"[deepemu-play] cartridge={cartridge} ckpt={ckpt} encoder={encoder_path or 'none'}")
    print(f"[deepemu-play] toggle file: {toggle} (write '0' to pause agent)")

    total_steps = 0
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            for step in range(max_episode_steps):
                # Human takeover check (cheap; just stat the file)
                agent_on = _read_agent_enabled(toggle)
                if agent_on:
                    action = agent.act(obs, explore=False)
                else:
                    # idle while paused. On Game Boy, action 0 is "down" — only
                    # ALE has a true NOOP at index 0.
                    is_gb = getattr(env.cartridge, "platform", "gameboy") == "gameboy"
                    action = None if is_gb else 0
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                total_steps += 1

                # Live arrows
                if arrows_overlay is not None and "trajectory" in info:
                    x, y, m = info["trajectory"]
                    arrows_overlay.update(int(x), int(y), int(m))
                    if step % 4 == 0:
                        arrows_overlay.draw()

                # Live attention
                if attention_overlay is not None and step % 4 == 0:
                    try:
                        # The current frame from the underlying env
                        if hasattr(env, "inner"):
                            frame = env.inner.render()
                        else:
                            frame = env.render()
                        if frame is not None:
                            attention_overlay.draw(frame[:, :, :3])
                    except Exception as e:  # don't crash play loop on viz errors
                        print(f"[deepemu-play] attention overlay error: {e}")

                # Handle pygame quit events
                if pygame is not None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                if term or trunc:
                    break
            print(
                f"  episode {ep:3d}: reward={ep_reward:8.3f} steps={step + 1:5d} "
                f"toggle={'agent' if agent_on else 'human'}"
            )
    except KeyboardInterrupt:
        print("[deepemu-play] interrupted by user")
    finally:
        if pygame is not None:
            pygame.quit()
        try:
            env.close()
        except Exception:
            pass

    print(f"[deepemu-play] total env steps: {total_steps}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_play(
        cartridge=args.cartridge,
        rom=args.rom,
        init_state=args.init_state,
        ckpt=args.ckpt,
        visible=args.visible,
        arrows=args.arrows,
        attention=args.attention,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
