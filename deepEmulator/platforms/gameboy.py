"""PyBoy backend — Gymnasium-free EmulatorEnv.

Observation: stacked grayscale screen (frame_stack, H, W) downscaled 2x from
(144, 160) to (72, 80). Action: integer index into the cartridge's action_set
(PyBoy button name strings like "a", "left").

Ported from PWhiddy v2/red_gym_env_v2.py step/run_action_on_emulator logic.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.env import EmulatorEnv
from deepEmulator.core.spaces import Box, Discrete


def downscale_luma_2x(frame: np.ndarray) -> np.ndarray:
    """(144, 160, 3|4) uint8 RGB(A) -> (72, 80) uint8 grayscale, fused.

    Single uint16 pass: sum 2x2 blocks across the 3 channels (12 samples,
    max 3060 < 65535) then integer-divide. Replaces two float64 mean passes
    (~3.15 ms -> ~1.15 ms per frame; |diff| <= 1 vs the float reference).
    """
    h2, w2 = frame.shape[0] // 2, frame.shape[1] // 2
    summed = frame[: h2 * 2, : w2 * 2, :3].reshape(h2, 2, w2, 2, 3).sum(
        axis=(1, 3, 4), dtype=np.uint16
    )
    return (summed // 12).astype(np.uint8)


class PyBoyEnv(EmulatorEnv):
    """Game Boy / GBC env wrapping PyBoy 2.x.

    Args:
        cartridge: a CartridgeAdapter (e.g. PokemonRedAdapter)
        rom_path: path to .gb / .gbc ROM
        init_state: optional PyBoy save-state to load on reset
        headless: True for "null" window, False for visible "SDL2"
        action_freq: total ticks per env step
        press_ticks: ticks the button is held before release
        max_steps: episode horizon
        frame_stack: number of stacked grayscale frames in the observation
    """

    def __init__(
        self,
        cartridge: CartridgeAdapter,
        rom_path: str | Path,
        init_state: str | Path | None = None,
        headless: bool = True,
        action_freq: int = 24,
        press_ticks: int = 8,
        max_steps: int = 20_480,
        frame_stack: int = 3,
        boot_ticks: int = 60,
        reward_clip: float | None = 5.0,
    ):
        super().__init__(cartridge)
        from pyboy import PyBoy  # local import so import-only tests don't require pyboy

        self.rom_path = str(rom_path)
        self.init_state = Path(init_state) if init_state else cartridge.init_state
        self.headless = headless
        self.action_freq = action_freq
        self.press_ticks = press_ticks
        self.max_steps = max_steps
        self.frame_stack = frame_stack
        # GB reward deltas span ~0.01 (boot tick) to 10+ (badge) — clamp so a
        # single transition can't dominate the Huber targets. None disables.
        self.reward_clip = reward_clip if (reward_clip or 0) > 0 else None

        self.pyboy = PyBoy(self.rom_path, window="null" if headless else "SDL2")
        if not headless:
            self.pyboy.set_emulation_speed(6)

        # Snapshot the post-boot state so reset() without an init_state actually
        # resets the emulator. Without this, episode N+1 continues from wherever
        # episode N truncated while the adapter wipes its exploration bookkeeping
        # — re-earning explore reward for the same tiles every episode.
        self._boot_state: io.BytesIO | None = None
        if self.init_state is None:
            self.pyboy.tick(boot_ticks, False)
            self._boot_state = io.BytesIO()
            self.pyboy.save_state(self._boot_state)

        self._frame_h, self._frame_w = 72, 80  # 144/2, 160/2
        self.observation_space = Box(
            low=0.0,
            high=255.0,
            shape=(frame_stack, self._frame_h, self._frame_w),
            dtype=np.dtype(np.uint8),
        )
        self.action_space = Discrete(len(cartridge.action_set))

        self._screen_stack = np.zeros(
            (frame_stack, self._frame_h, self._frame_w), dtype=np.uint8
        )
        self._step_count = 0
        self._prev_state: dict = {}

    # --- helpers ------------------------------------------------------------
    def _grab_frame(self) -> np.ndarray:
        # PyBoy 2.x: pyboy.screen.ndarray -> (144, 160, 3 or 4).
        # Channel mean = luminance (no-op on DMG, correct on GBC), fused with
        # the 2x block-mean downscale in one uint16 pass — this is the hottest
        # python line in the training loop.
        return downscale_luma_2x(self.pyboy.screen.ndarray)

    def _push_frame(self, frame: np.ndarray) -> None:
        # in-place reversed shift — np.roll allocated a full stack per push
        for i in range(self.frame_stack - 1, 0, -1):
            self._screen_stack[i] = self._screen_stack[i - 1]
        self._screen_stack[0] = frame

    def _send_action(self, action: int | None) -> None:
        render = not self.headless
        if action is None:
            # idle step: advance time without pressing anything (human takeover)
            self.pyboy.tick(self.action_freq - 1, render)
            self.pyboy.tick(1, True)
            return
        button = self.cartridge.action_set[action]
        self.pyboy.button_press(button)
        self.pyboy.tick(self.press_ticks, render)
        self.pyboy.button_release(button)
        self.pyboy.tick(self.action_freq - self.press_ticks - 1, render)
        self.pyboy.tick(1, True)  # final tick always renders for observation

    # --- EmulatorEnv API ----------------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        # `seed` accepted for Env-protocol compatibility but unused: PyBoy is
        # deterministic from a loaded state.
        if self.init_state is not None:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
        elif self._boot_state is not None:
            self._boot_state.seek(0)
            self.pyboy.load_state(self._boot_state)
        # refresh the framebuffer — load_state alone can leave the previous
        # episode's last frame on screen, making the first obs stale
        self.pyboy.tick(1, True)
        self.cartridge.reset_episode(self.pyboy)
        self._step_count = 0
        self._screen_stack[:] = 0
        frame = self._grab_frame()
        for _ in range(self.frame_stack):
            self._push_frame(frame)
        self._prev_state = self.cartridge.read_game_state(self.pyboy)
        return self._screen_stack.copy(), {"game_state": self._prev_state}

    def step(self, action: int | None) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._send_action(action)
        curr_state = self.cartridge.read_game_state(self.pyboy)
        reward = self.cartridge.compute_reward(self._prev_state, curr_state, self.pyboy)
        if self.reward_clip is not None:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))
        self._push_frame(self._grab_frame())

        self._step_count += 1
        terminated = self.cartridge.is_done(curr_state)
        truncated = self._step_count >= self.max_steps
        self._prev_state = curr_state

        info = {
            "game_state": curr_state,
            "trajectory": self.cartridge.get_trajectory_coords(curr_state),
        }
        return self._screen_stack.copy(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.pyboy.screen.ndarray.copy()

    def close(self) -> None:
        self.pyboy.stop(save=False)
