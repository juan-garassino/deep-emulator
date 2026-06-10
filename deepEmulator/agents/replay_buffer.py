"""Preallocated ring replay buffer with frame-dedup storage and n-step returns.

Replaces the deque-of-tensors buffer:
- frames stored ONCE per step (uint8) instead of full (state, next_state)
  stack pairs — ~5.8x less RAM at the default 100K capacity (3.4 GB -> 0.6 GB)
- O(1) numpy sampling instead of random.sample over a deque (O(n) per index)
- per-env contiguous sub-rings so a future vectorized runner can interleave
  transitions without cross-episode stitching
- n-step aggregation lives HERE: callers feed raw transitions, the buffer
  emits matured windows (R = sum gamma^i r_i, discount gamma^k, partial
  flush at episode end with terminated/truncated distinguished)

Storage layout (per env sub-ring of capacity // n_envs slots):
    slot i holds the NEWEST observation frame after i in-episode steps;
    a frame-stack observation is reconstructed as
        obs[k] = frame[i - min(k, ep_step[i])]
    which is EXACTLY the env's own reset semantics (the reset stack repeats
    one frame), so episode boundaries need no special-casing.
    A transition "starts" at slot i: state = obs(i), next = obs(i + span[i]).

Rank-1 (latent) observations use the same machinery with the full vector
stored per slot and no stacking.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        *,
        n_envs: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        seed: int | None = None,
    ):
        if len(obs_shape) == 3:
            self.frame_mode = True
            self.frame_stack = int(obs_shape[0])
            frame_shape: tuple[int, ...] = tuple(obs_shape[1:])
            frame_dtype = np.uint8
        elif len(obs_shape) == 1:
            self.frame_mode = False
            self.frame_stack = 1
            frame_shape = (int(obs_shape[0]),)
            frame_dtype = np.float32
        else:
            raise ValueError(f"unsupported obs shape {obs_shape}")

        self.n_envs = int(n_envs)
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.cap = max(16, int(capacity) // self.n_envs)
        n_total = self.cap * self.n_envs

        self.frames = np.zeros((n_total, *frame_shape), dtype=frame_dtype)
        self.ep_step = np.zeros(n_total, dtype=np.int32)
        self.action = np.zeros(n_total, dtype=np.int64)
        self.reward = np.zeros(n_total, dtype=np.float32)
        self.done = np.zeros(n_total, dtype=np.float32)  # terminated only — truncation bootstraps
        self.discount = np.zeros(n_total, dtype=np.float32)  # gamma^k for the window starting here
        self.span = np.zeros(n_total, dtype=np.int32)  # k (slots to the window's next_obs)
        self.complete = np.zeros(n_total, dtype=bool)  # transition starting here is emitted

        self.head = np.zeros(self.n_envs, dtype=np.int64)  # monotonic per-env write counts
        # raw (abs_pos_of_state, action, reward) awaiting n-step maturity
        self._queues: list[deque] = [deque() for _ in range(self.n_envs)]
        self._n_complete = 0
        self._rng = np.random.default_rng(seed)

        # frames older than this can be partially overwritten for a still-
        # "complete" transition — exclude them at sample time
        self._margin = self.frame_stack + self.n_step + 1

    # --- write path --------------------------------------------------------
    def _slot(self, env_id: int, pos: np.ndarray | int) -> np.ndarray | int:
        return env_id * self.cap + pos % self.cap

    def _newest_frame(self, obs: np.ndarray) -> np.ndarray:
        return obs[0] if self.frame_mode else obs

    def _write_frame(self, env_id: int, frame: np.ndarray, ep_step: int) -> int:
        pos = int(self.head[env_id])
        slot = self._slot(env_id, pos)
        if self.complete[slot]:
            self.complete[slot] = False
            self._n_complete -= 1
        self.frames[slot] = frame
        self.ep_step[slot] = ep_step
        self.head[env_id] = pos + 1
        return pos

    def begin_episode(self, obs: np.ndarray, env_id: int = 0) -> None:
        """Record the reset observation. Must be called before the episode's
        first add() and after every episode end."""
        self._queues[env_id].clear()
        self._write_frame(env_id, self._newest_frame(obs), ep_step=0)

    def add(
        self,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        env_id: int = 0,
    ) -> None:
        """Record one RAW transition; the buffer matures n-step windows itself."""
        prev_pos = int(self.head[env_id]) - 1
        if prev_pos < 0:
            raise RuntimeError("add() before begin_episode()")
        prev_ep_step = int(self.ep_step[self._slot(env_id, prev_pos)])
        self._write_frame(env_id, self._newest_frame(next_obs), ep_step=prev_ep_step + 1)

        q = self._queues[env_id]
        q.append((prev_pos, int(action), float(reward)))
        if len(q) == self.n_step:
            self._emit(env_id, q, bool(terminated))
            q.popleft()
        if terminated or truncated:
            while q:
                self._emit(env_id, q, bool(terminated))
                q.popleft()

    def _emit(self, env_id: int, q: deque, terminated: bool) -> None:
        start_pos, a0, _ = q[0]
        ret = 0.0
        for i, (_, _, r_i) in enumerate(q):
            ret += (self.gamma**i) * r_i
        k = len(q)
        slot = self._slot(env_id, start_pos)
        # the start slot may already have wrapped out — then the window is lost
        if int(self.head[env_id]) - start_pos > self.cap:
            return
        self.action[slot] = a0
        self.reward[slot] = ret
        self.done[slot] = 1.0 if terminated else 0.0
        self.discount[slot] = self.gamma**k
        self.span[slot] = k
        if not self.complete[slot]:
            self.complete[slot] = True
            self._n_complete += 1

    # --- read path -----------------------------------------------------------
    def __len__(self) -> int:
        return self._n_complete

    def _valid_range(self, env_id: int) -> tuple[int, int]:
        head = int(self.head[env_id])
        lo = max(0, head - self.cap + self._margin)
        return lo, head - 1  # positions [lo, head-1); head-1 itself has no next yet

    def _sample_positions(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        env_ids = np.empty(batch_size, dtype=np.int64)
        positions = np.empty(batch_size, dtype=np.int64)
        filled = 0
        attempts = 0
        max_attempts = 200
        while filled < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"could not sample {batch_size} transitions "
                    f"({self._n_complete} complete in buffer)"
                )
            need = batch_size - filled
            cand_env = self._rng.integers(0, self.n_envs, size=need * 2)
            cand_pos = np.empty(need * 2, dtype=np.int64)
            ok = np.zeros(need * 2, dtype=bool)
            for j in range(need * 2):
                e = int(cand_env[j])
                lo, hi = self._valid_range(e)
                if hi <= lo:
                    continue
                p = int(self._rng.integers(lo, hi))
                cand_pos[j] = p
                ok[j] = bool(self.complete[self._slot(e, p)])
            take = min(need, int(ok.sum()))
            sel = np.flatnonzero(ok)[:take]
            env_ids[filled : filled + take] = cand_env[sel]
            positions[filled : filled + take] = cand_pos[sel]
            filled += take
        return env_ids, positions

    def _obs_at(self, env_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        slots = env_ids * self.cap + positions % self.cap
        if not self.frame_mode:
            return self.frames[slots]
        b = len(positions)
        out = np.empty((b, self.frame_stack, *self.frames.shape[1:]), dtype=self.frames.dtype)
        eps = self.ep_step[slots]
        for k in range(self.frame_stack):
            pk = positions - np.minimum(k, eps)
            out[:, k] = self.frames[env_ids * self.cap + pk % self.cap]
        return out

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        env_ids, pos = self._sample_positions(batch_size)
        slots = env_ids * self.cap + pos % self.cap
        s = self._obs_at(env_ids, pos)
        ns = self._obs_at(env_ids, pos + self.span[slots])
        return (
            s,
            ns,
            self.action[slots].copy(),
            self.reward[slots].copy(),
            self.done[slots].copy(),
            self.discount[slots].copy(),
        )

    def sample_torch(self, batch_size: int, device: str) -> tuple[torch.Tensor, ...]:
        """uint8 H2D transfer, float conversion ON DEVICE. No /255 here — the
        agent's _prep owns normalization (same transform act() uses)."""
        s, ns, a, r, d, disc = self.sample(batch_size)
        return (
            torch.from_numpy(s).to(device).float(),
            torch.from_numpy(ns).to(device).float(),
            torch.from_numpy(a).to(device),
            torch.from_numpy(r).to(device),
            torch.from_numpy(d).to(device),
            torch.from_numpy(disc).to(device),
        )
