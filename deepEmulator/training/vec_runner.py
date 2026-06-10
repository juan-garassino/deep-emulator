"""Vectorized env runner — N worker processes, lock-step synchronous stepping.

Design notes (deliberate, see the audit/plan):
- spawn ALWAYS: macOS SDL safety, no fork-inherited CUDA contexts, and CI
  Linux exercises the same code path as dev machines.
- Only `EnvSpec` (a plain picklable dataclass) crosses the process boundary.
  Workers build their env + a FRESH adapter themselves — adapters are
  stateful (seen_coords, reward baselines); sharing one across envs corrupts
  reward state by construction.
- Pipes with pickled tuples: obs are <= ~17 KB; at PyBoy step cost the
  serialization is noise. Shared-memory rings are future work.
- AUTO-RESET: when an episode ends, the worker resets itself and ships the
  reset observation in the SAME reply — no extra round-trip.
- Actions are computed in the MAIN process (batched forward on the agent's
  device); workers never see the policy.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EnvSpec:
    """Everything a worker needs to build its env. Plain data — picklable."""

    cartridge: str  # registry key, e.g. "POKEMON CORAL" / "SYNTH BLOB"
    rom_path: str | None = None
    init_state: str | None = None
    max_episode_steps: int = 2048
    reward_clip: float | None = 5.0
    include_start: bool = False
    env_kwargs: dict = field(default_factory=dict)


def build_env(spec: EnvSpec):
    """Construct env + FRESH adapter from a spec. Used by workers AND the
    serial path so both run identical construction code."""
    import dataclasses

    from deepEmulator.cartridges import load_all
    from deepEmulator.core import registry

    load_all()
    AdapterCls = registry.get(spec.cartridge)
    kwargs: dict = {}
    if spec.init_state:
        kwargs["init_state"] = Path(spec.init_state)
    if spec.include_start and any(
        f.name == "include_start" for f in dataclasses.fields(AdapterCls)
    ):
        kwargs["include_start"] = True
    adapter = AdapterCls(**kwargs)

    platform = getattr(adapter, "platform", "gameboy")
    if spec.cartridge.upper() == "SYNTH BLOB":
        from deepEmulator.platforms.synthetic import SyntheticEmulatorEnv

        return SyntheticEmulatorEnv(
            adapter,
            episode_length=spec.max_episode_steps,
            **spec.env_kwargs,
        )
    if platform == "gameboy":
        from deepEmulator.platforms.gameboy import PyBoyEnv

        return PyBoyEnv(
            adapter,
            rom_path=spec.rom_path,
            init_state=Path(spec.init_state) if spec.init_state else None,
            headless=True,
            max_steps=spec.max_episode_steps,
            reward_clip=spec.reward_clip,
            **spec.env_kwargs,
        )
    if platform == "atari":
        from deepEmulator.platforms.atari import AtariEnv

        return AtariEnv(adapter, rom_path=spec.rom_path, headless=True, **spec.env_kwargs)
    raise ValueError(f"unsupported platform {platform!r}")


def _worker(conn, spec: EnvSpec, worker_id: int, base_seed: int) -> None:
    """Worker loop. Protocol (tuples over the pipe):
        main -> worker:  ("reset", seed) | ("step", action:int) | ("close",)
        worker -> main:  ("reset_ok", obs, info_lite)
                         ("step_ok", obs, reward, terminated, truncated,
                                     info_lite, reset_obs_or_None)
                         ("error", traceback_str)
    """
    try:
        random.seed(base_seed + worker_id)
        np.random.seed(base_seed + worker_id)
        env = build_env(spec)
    except Exception:
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return

    def _info_lite(info: dict) -> dict:
        out = {}
        if "trajectory" in info:
            x, y, m = info["trajectory"]
            out["trajectory"] = (int(x), int(y), int(m))
        return out

    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "reset":
                obs, info = env.reset(seed=msg[1])
                conn.send(("reset_ok", obs, _info_lite(info)))
            elif cmd == "step":
                obs, r, term, trunc, info = env.step(int(msg[1]))
                reset_obs = None
                if term or trunc:
                    reset_obs, _ = env.reset()
                conn.send(("step_ok", obs, float(r), bool(term), bool(trunc),
                           _info_lite(info), reset_obs))
            elif cmd == "close":
                break
            else:
                conn.send(("error", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


@dataclass
class StepBatch:
    obs: np.ndarray  # (N, *obs_shape) — post-auto-reset for finished envs
    rewards: np.ndarray  # (N,)
    terminated: np.ndarray  # (N,) bool
    truncated: np.ndarray  # (N,) bool
    infos: list  # list[dict]
    pre_reset_obs: list  # the TERMINAL obs for envs that auto-reset, else None


class VecEnvRunner:
    """Lock-step synchronous vector of worker-owned envs."""

    def __init__(self, spec: EnvSpec, num_envs: int, base_seed: int = 0, timeout: float = 120.0):
        self.num_envs = int(num_envs)
        self.timeout = timeout
        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        for i in range(self.num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker, args=(child, spec, i, base_seed), daemon=True
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._closed = False

    def _recv(self, i: int):
        if not self._conns[i].poll(self.timeout):
            self.close()
            raise RuntimeError(f"worker {i} timed out after {self.timeout}s")
        msg = self._conns[i].recv()
        if msg[0] == "error":
            self.close()
            raise RuntimeError(f"worker {i} failed:\n{msg[1]}")
        return msg

    def reset(self, base_seed: int = 0) -> np.ndarray:
        for i, conn in enumerate(self._conns):
            conn.send(("reset", base_seed + i))
        obs = []
        for i in range(self.num_envs):
            msg = self._recv(i)
            obs.append(msg[1])
        return np.stack(obs)

    def step(self, actions: np.ndarray) -> StepBatch:
        for conn, a in zip(self._conns, actions):
            conn.send(("step", int(a)))
        obs_out, rewards, terms, truncs, infos, pre_reset = [], [], [], [], [], []
        for i in range(self.num_envs):
            msg = self._recv(i)
            _, obs, r, term, trunc, info, reset_obs = msg
            if reset_obs is not None:
                pre_reset.append(obs)  # the terminal obs (for the buffer)
                obs_out.append(reset_obs)  # the next episode's first obs
            else:
                pre_reset.append(None)
                obs_out.append(obs)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return StepBatch(
            obs=np.stack(obs_out),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terms, dtype=bool),
            truncated=np.asarray(truncs, dtype=bool),
            infos=infos,
            pre_reset_obs=pre_reset,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
