"""Synthetic emulator — bouncing-blob env with no external deps.

Promoted from scripts/nano_e2e.py so tests and the vectorized runner can
exercise the full pipeline (multiprocessing included) on CI without ROMs.
The 'game' is a bouncing 8x8 blob on a 144x160 canvas; actions nudge its
velocity and the reward is the +x velocity component (the agent learns to
move right). Registered as "SYNTH BLOB".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.env import EmulatorEnv
from deepEmulator.core.registry import register
from deepEmulator.core.spaces import Box, Discrete


class SyntheticEmulatorEnv(EmulatorEnv):
    """A pixel-CNN-compatible env with no external deps. Done (truncated)
    after `episode_length` steps."""

    def __init__(self, cartridge: CartridgeAdapter, *, episode_length: int = 50, seed: int = 0):
        super().__init__(cartridge)
        self.h, self.w = 144, 160
        self.frame_stack = 3
        self.episode_length = episode_length
        self.action_space = Discrete(len(cartridge.action_set) or 7)
        self.observation_space = Box(0, 255, (self.frame_stack, self.h // 2, self.w // 2))
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.x = self.w // 2
        self.y = self.h // 2
        self.vx = 0
        self.vy = 0
        self.t = 0
        self._stack = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _render_frame_color(self) -> np.ndarray:
        canvas = np.full((self.h, self.w, 3), 30, dtype=np.uint8)
        # Background gradient (gives the encoder texture)
        canvas[..., 0] += (np.linspace(0, 60, self.h)[:, None]).astype(np.uint8)
        canvas[..., 1] += (np.linspace(0, 60, self.w)[None, :]).astype(np.uint8)
        # Bouncing blob
        x0, y0 = max(0, int(self.x) - 4), max(0, int(self.y) - 4)
        x1, y1 = min(self.w, int(self.x) + 4), min(self.h, int(self.y) + 4)
        canvas[y0:y1, x0:x1] = [240, 220, 80]
        return canvas

    def _downscaled_gray(self) -> np.ndarray:
        rgb = self._render_frame_color()
        gray = rgb.mean(axis=-1).astype(np.uint8)
        return gray.reshape(self.h // 2, 2, self.w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)

    def _push(self, frame: np.ndarray) -> None:
        for i in range(self.frame_stack - 1, 0, -1):
            self._stack[i] = self._stack[i - 1]
        self._stack[0] = frame

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        first = self._downscaled_gray()
        for _ in range(self.frame_stack):
            self._push(first)
        return self._stack.copy(), {"game_state": {"x": self.x, "y": self.y, "map_id": 0}}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Action 0=noop, 1=left, 2=right, 3=up, 4=down, 5/6=ignored
        dvx = {1: -1, 2: 1}.get(action, 0)
        dvy = {3: -1, 4: 1}.get(action, 0)
        self.vx = int(np.clip(self.vx + dvx, -3, 3))
        self.vy = int(np.clip(self.vy + dvy, -3, 3))
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x <= 4 or self.x >= self.w - 4:
            self.vx = -self.vx
        if self.y <= 4 or self.y >= self.h - 4:
            self.vy = -self.vy
        self.x = int(np.clip(self.x, 4, self.w - 4))
        self.y = int(np.clip(self.y, 4, self.h - 4))
        self._push(self._downscaled_gray())
        self.t += 1
        reward = float(self.vx) * 0.1
        done = self.t >= self.episode_length
        return (
            self._stack.copy(),
            reward,
            False,
            done,
            {
                "game_state": {"x": self.x, "y": self.y, "map_id": 0},
                "trajectory": (self.x, self.y, 0),
            },
        )

    def render(self) -> np.ndarray:
        return self._render_frame_color()

    def close(self) -> None:
        pass


@dataclass
class SyntheticCartridge(CartridgeAdapter):
    cartridge_title: str = "SYNTH BLOB"
    platform: str = "gameboy"
    action_set: list = field(default_factory=lambda: list(range(7)))
    observation_shape: tuple = (3, 72, 80)

    def reset_episode(self, emulator) -> None:
        pass

    def read_game_state(self, emulator) -> dict:
        return {"x": 0, "y": 0, "map_id": 0}

    def compute_reward(self, prev_state, curr_state, emulator) -> float:
        return 0.0

    def is_done(self, state: dict) -> bool:
        return False


register("SYNTH BLOB")(SyntheticCartridge)
