"""Frame corpus collection for SSL pretraining.

Two modes:
- **Stream** (`FrameRing`): in-memory ring buffer; collector pushes frames during
  pretraining, no disk persistence. Cheap, non-reproducible.
- **Buffered** (`FrameStorage`): append-only on-disk chunks (.npz). Resumeable
  across Colab sessions, reproducible, costs Drive space.

All frames are normalized to (1, 96, 96) uint8 grayscale regardless of source
platform (GB 144×160, Atari 210×160, SEGA 320×224). The normalization step
(grayscale + center-crop to square + bilinear resize) is the only thing the
encoder ever sees — RL training uses the raw per-platform observations.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# --- normalization ----------------------------------------------------------
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return (frame.astype(np.float32) @ _GRAY_WEIGHTS).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        if frame.shape[0] == 1:
            return frame[0].astype(np.uint8)
        return (frame[:3].astype(np.float32).transpose(1, 2, 0) @ _GRAY_WEIGHTS).astype(np.uint8)
    raise ValueError(f"unsupported frame shape {frame.shape}")


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _bilinear_resize(img: np.ndarray, size: int) -> np.ndarray:
    src_h, src_w = img.shape
    if src_h == size and src_w == size:
        return img.astype(np.uint8)
    y = np.linspace(0, src_h - 1, size)
    x = np.linspace(0, src_w - 1, size)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    fy = (y - y0).astype(np.float32)
    fx = (x - x0).astype(np.float32)
    f = img.astype(np.float32)
    a = f[y0[:, None], x0[None, :]]
    b = f[y0[:, None], x1[None, :]]
    c = f[y1[:, None], x0[None, :]]
    d = f[y1[:, None], x1[None, :]]
    out = (
        a * (1 - fy[:, None]) * (1 - fx[None, :])
        + b * (1 - fy[:, None]) * fx[None, :]
        + c * fy[:, None] * (1 - fx[None, :])
        + d * fy[:, None] * fx[None, :]
    )
    return out.astype(np.uint8)


def normalize_to_96x96(frame: np.ndarray) -> np.ndarray:
    """Any platform frame → (1, 96, 96) uint8 grayscale."""
    g = _to_grayscale(frame)
    s = _center_crop_square(g)
    r = _bilinear_resize(s, 96)
    return r[None, :, :]


# --- in-memory ring ---------------------------------------------------------
class FrameRing:
    """Fixed-size FIFO of normalized frames."""

    def __init__(self, capacity: int = 50_000, *, rng: int | None = None):
        self.capacity = int(capacity)
        self._buf: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(rng)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, frame_norm: np.ndarray) -> None:
        if frame_norm.shape != (1, 96, 96):
            raise ValueError(f"expected (1, 96, 96), got {frame_norm.shape}")
        self._buf.append(frame_norm.astype(np.uint8, copy=False))

    def sample(self, batch_size: int) -> np.ndarray:
        if len(self._buf) == 0:
            raise RuntimeError("ring is empty")
        idx = self._rng.integers(0, len(self._buf), size=batch_size)
        return np.stack([self._buf[int(i)] for i in idx])

    def iter_batches(self, batch_size: int, n_batches: int) -> Iterator[np.ndarray]:
        for _ in range(n_batches):
            yield self.sample(batch_size)


# --- disk-backed chunks -----------------------------------------------------
class FrameStorage:
    """Append-only chunks on disk: {root}/chunk_{i:06d}.npz + index.json."""

    def __init__(self, root: Path | str, chunk_size: int = 10_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self._buffer: list[np.ndarray] = []
        self._index_path = self.root / "index.json"
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())
        else:
            self._index = {"chunks": [], "total_frames": 0, "shape": [1, 96, 96]}
            self._save_index()

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    @property
    def total_frames(self) -> int:
        return self._index["total_frames"] + len(self._buffer)

    def push(self, frame_norm: np.ndarray) -> None:
        if frame_norm.shape != (1, 96, 96):
            raise ValueError(f"expected (1, 96, 96), got {frame_norm.shape}")
        self._buffer.append(frame_norm.astype(np.uint8, copy=False))
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        idx = len(self._index["chunks"])
        path = self.root / f"chunk_{idx:06d}.npz"
        arr = np.stack(self._buffer)
        np.savez_compressed(path, frames=arr)
        self._index["chunks"].append(path.name)
        self._index["total_frames"] += len(self._buffer)
        self._save_index()
        self._buffer = []

    def iter_chunks(self) -> Iterator[np.ndarray]:
        for name in self._index["chunks"]:
            data = np.load(self.root / name)
            yield data["frames"]

    def iter_batches(self, batch_size: int, *, shuffle: bool = True, seed: int = 0) -> Iterator[np.ndarray]:
        chunks = list(self.iter_chunks())
        if not chunks and not self._buffer:
            raise RuntimeError("storage is empty")
        if self._buffer:
            chunks.append(np.stack(self._buffer))
        full = np.concatenate(chunks, axis=0)
        rng = np.random.default_rng(seed)
        n = len(full)
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            sel = order[start : start + batch_size]
            if len(sel) < batch_size:
                break
            yield full[sel]


# --- collector --------------------------------------------------------------
class FrameCollector:
    """Round-robin collector across multiple envs.

    Each `step()` pulls one frame from one env (advanced by a random action),
    normalizes it, and pushes to the target sink (`FrameRing` or `FrameStorage`).
    """

    def __init__(
        self,
        envs: Iterable,
        sink,
        *,
        policy: str = "random",
        seed: int = 0,
        screen_extractor=None,
    ):
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("at least one env required")
        self.sink = sink
        self.policy = policy
        self._rng = random.Random(seed)
        self._cursor = 0
        self._screen_extractor = screen_extractor or self._default_extractor
        self._step_obs_cache = [None] * len(self.envs)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            self._step_obs_cache[i] = obs

    @staticmethod
    def _default_extractor(env, obs) -> np.ndarray:
        """Try env.render() first (full-resolution color frame); fall back to obs[0]."""
        try:
            frame = env.render()
            if frame is not None:
                return frame
        except Exception:
            pass
        return obs[0] if obs.ndim == 3 else obs

    def step(self) -> int:
        """Advance one env by one action; push its frame; return env index."""
        i = self._cursor
        env = self.envs[i]
        action = self._rng.randint(0, env.action_space.n - 1)
        obs, _, term, trunc, _ = env.step(action)
        self._step_obs_cache[i] = obs

        raw = self._screen_extractor(env, obs)
        self.sink.push(normalize_to_96x96(raw))

        if term or trunc:
            obs, _ = env.reset()
            self._step_obs_cache[i] = obs

        self._cursor = (self._cursor + 1) % len(self.envs)
        return i

    def run(self, n_frames: int) -> int:
        for _ in range(n_frames):
            self.step()
        return n_frames
