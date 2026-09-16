"""Double-DQN agent — vendored & refactored from lixado/PyBoy-RL (agent.py, model.py).

Architecture is the classic Mnih-DQN CNN (3 convs + 2 FC), sized for our
(frame_stack, 72, 80) PyBoy observation. Linear input is auto-computed.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from deepEmulator.agents.replay_buffer import ReplayBuffer
from deepEmulator.utils.device import get_device_str, is_cuda


# --- hyperparameters --------------------------------------------------------
@dataclass
class DDQNConfig:
    exploration_rate: float = 1.0
    # multiplicative decay is the fallback when exploration_anneal_steps is None.
    # NOTE: 0.99999975 needs ~12M act() calls to reach the floor — for budgeted
    # runs always set exploration_anneal_steps (train.py does).
    exploration_rate_decay: float = 0.99999975
    exploration_rate_min: float = 0.05
    # when set: linear anneal 1.0 -> min over this many steps, as a pure
    # function of curr_step (resume-correct, batch-stepping-correct)
    exploration_anneal_steps: int | None = None
    deque_size: int = 100_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 0.00025
    burnin: int = 1_000
    learn_every: int = 3
    sync_every: int = 1_000
    grad_clip_norm: float = 10.0
    # divide pixel observations by 255 in act()/sampling. Off by default for
    # backward compat with pre-fix bundles; train.py turns it on for new
    # pixel runs and records it in bundle metadata. Never applies to rank-1
    # latent observations.
    normalize_obs: bool = False
    # dueling V/A head. Off by default so old model.pt state-dict keys keep
    # loading; train.py defaults new runs to on and records it in metadata.
    dueling: bool = False
    # number of parallel envs feeding the buffer (per-env sub-rings)
    n_envs: int = 1
    # mixed-precision learn steps (CUDA only — silently a no-op on CPU so the
    # same config runs everywhere). Off by default; benchmark on a real pod
    # before enabling for long runs.
    amp: bool = False
    # n-step returns. 1 = classic one-step TD (byte-identical to the old
    # behavior). >1 aggregates R = sum(gamma^i * r_i) before insertion and
    # bootstraps with gamma^n; partial windows flush at episode end.
    n_step: int = 1


def config_from_metadata(metadata: dict) -> DDQNConfig:
    """Build an inference-compatible config from a bundle's metadata.

    Bundles written before the `network` block existed get legacy settings
    (no dueling, no normalization, n_step=1) with a warning — their state
    dicts only load into the original architecture.
    """
    net = metadata.get("network")
    if net is None:
        print(
            "[ddqn] pre-fix bundle (no `network` metadata): assuming legacy settings "
            "(no dueling, no normalization, n_step=1)"
        )
        return DDQNConfig(dueling=False, normalize_obs=False, n_step=1)
    return DDQNConfig(
        dueling=bool(net.get("dueling", False)),
        normalize_obs=bool(net.get("normalize_obs", False)),
        n_step=int(net.get("n_step", 1)),
        gamma=float(net.get("gamma", 0.99)),
    )


# --- network ----------------------------------------------------------------
class _DuelingHead(nn.Module):
    """Trunk -> V(s) + A(s, a), combined as Q = V + A - mean(A)."""

    def __init__(self, trunk: nn.Module, trunk_dim: int, n_actions: int):
        super().__init__()
        self.trunk = trunk
        self.value = nn.Linear(trunk_dim, 1)
        self.advantage = nn.Linear(trunk_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)


class DDQNNet(nn.Module):
    """Dual online/target DDQN. Dispatches on obs rank:
        rank-3 (C, H, W) -> Mnih-style CNN
        rank-1 (D,)      -> 2-layer MLP (for latents from a frozen encoder)

    `dueling=False` reproduces the original Sequential layouts exactly, so
    pre-dueling model.pt state dicts keep loading without migration.
    """

    def __init__(self, input_shape: tuple[int, ...], n_actions: int, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        if len(input_shape) == 3:
            self.mode = "cnn"
            self.online = self._build_cnn(input_shape, n_actions, dueling)
        elif len(input_shape) == 1:
            self.mode = "mlp"
            self.online = self._build_mlp(input_shape[0], n_actions, dueling)
        else:
            raise ValueError(f"unsupported obs shape {input_shape}")
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_cnn(input_shape: tuple[int, int, int], n_actions: int, dueling: bool) -> nn.Module:
        c, h, w = input_shape

        def _conv_stack():
            return nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        probe = _conv_stack()
        with torch.no_grad():
            flat = probe(torch.zeros(1, c, h, w)).shape[1]
        if dueling:
            trunk = nn.Sequential(_conv_stack(), nn.Linear(flat, 512), nn.ReLU())
            return _DuelingHead(trunk, 512, n_actions)
        return nn.Sequential(
            _conv_stack(),
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    @staticmethod
    def _build_mlp(in_dim: int, n_actions: int, dueling: bool) -> nn.Module:
        if dueling:
            trunk = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            return _DuelingHead(trunk, 256, n_actions)
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor, mode: str = "online") -> torch.Tensor:
        return (self.online if mode == "online" else self.target)(x)


# --- agent ------------------------------------------------------------------
class DDQNAgent:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: DDQNConfig | None = None,
        device: str | None = None,
    ):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.config = config or DDQNConfig()
        # device-agnostic: mps -> cuda -> cpu (never hardcode .cuda()). An
        # explicit `device` arg still wins for tests / forced-CPU eval.
        self.device = device or get_device_str()

        self.net = DDQNNet(obs_shape, n_actions, dueling=self.config.dueling).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.online.parameters(), lr=self.config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        # `memory` keeps its historical name (and the deque_size config knob)
        # but is now a preallocated ring with frame-dedup storage + n-step
        self.memory = ReplayBuffer(
            self.config.deque_size,
            obs_shape,
            n_envs=self.config.n_envs,
            n_step=self.config.n_step,
            gamma=self.config.gamma,
        )

        self.exploration_rate = self.config.exploration_rate
        self.eval_epsilon = 0.0  # used by act(explore=False); play/eval set this
        self.curr_step = 0
        self._needs_episode_start = True
        # AMP GradScaler + fp16 autocast are CUDA-only; on MPS/CPU they stay
        # off so the same config runs everywhere (the M5 Max path is plain fp32).
        self._amp_active = bool(self.config.amp) and is_cuda(self.device)
        try:
            # torch >= 2.3 unified API
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_active)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_active)

    # --- preprocessing -----------------------------------------------------
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.config.normalize_obs and len(self.obs_shape) == 3:
            x = x / 255.0
        return x

    # --- policy ----------------------------------------------------------
    def _greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
            q = self.net(self._prep(s), mode="online")
            return int(torch.argmax(q, dim=1).item())

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if not explore:
            # eval/play mode: fixed eval_epsilon, no decay, no step counting
            if self.eval_epsilon > 0.0 and random.random() < self.eval_epsilon:
                return random.randint(0, self.n_actions - 1)
            return self._greedy(state)

        if random.random() < self.exploration_rate:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = self._greedy(state)

        self.curr_step += 1
        self._update_epsilon(steps=1)
        return action

    def _update_epsilon(self, steps: int) -> None:
        # SEAM: the single epsilon plug point for serial and batched stepping
        if self.config.exploration_anneal_steps is not None:
            frac = min(1.0, self.curr_step / max(1, self.config.exploration_anneal_steps))
            self.exploration_rate = 1.0 - (1.0 - self.config.exploration_rate_min) * frac
        else:
            self.exploration_rate = max(
                self.config.exploration_rate_min,
                self.exploration_rate * self.config.exploration_rate_decay**steps,
            )

    def act_batch(self, states: np.ndarray) -> np.ndarray:
        """Batched epsilon-greedy for the vectorized runner: ONE forward over
        (N, *obs) on the agent device, per-env Bernoulli(eps) random swap-in.
        Advances curr_step by N."""
        states = np.asarray(states)
        n = states.shape[0]
        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device)
            q = self.net(self._prep(s), mode="online")
            greedy = q.argmax(dim=1).cpu().numpy()
        explore_mask = np.random.random(n) < self.exploration_rate
        randoms = np.random.randint(0, self.n_actions, size=n)
        actions = np.where(explore_mask, randoms, greedy)
        self.curr_step += n
        self._update_epsilon(steps=n)
        return actions

    # --- replay buffer ---------------------------------------------------
    def cache(self, state, next_state, action, reward, done, truncated: bool = False) -> None:
        """Record one RAW transition. `done` means TERMINATED (true env
        terminal — bootstrap is cut). Time-limit ends go in `truncated`: the
        value target still bootstraps, but the n-step window flushes.

        The ring buffer stores each frame once (F9.7 uint8 economy, now
        ~5.8x better) and matures n-step windows internally. Episode starts
        are inferred: the first cache after construction or after any episode
        end registers `state` as the reset observation.
        """
        if self._needs_episode_start:
            self.memory.begin_episode(np.asarray(state))
            self._needs_episode_start = False
        self.memory.add(
            np.asarray(next_state), int(action), float(reward), bool(done), bool(truncated)
        )
        if done or truncated:
            self._needs_episode_start = True

    def _recall(self):
        s, ns, a, r, d, disc = self.memory.sample_torch(self.config.batch_size, self.device)
        # _prep applies the same transform act() uses (optional /255),
        # keeping train/inference aligned
        return self._prep(s), self._prep(ns), a, r, d, disc

    # --- learning --------------------------------------------------------
    def sync_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())

    def train_step(self) -> tuple[float, float]:
        """One gradient step from one sampled batch (no cadence gating)."""
        s, ns, a, r, d, disc = self._recall()
        idx = torch.arange(self.config.batch_size, device=self.device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self._amp_active):
            q_est = self.net(s, mode="online")[idx, a]
            with torch.no_grad():
                next_q_online = self.net(ns, mode="online")
                best_a = torch.argmax(next_q_online, dim=1)
                next_q_target = self.net(ns, mode="target")[idx, best_a]
                # disc = gamma^k for the k-step return ending at ns (k=1 classic)
                q_tgt = r + (1.0 - d) * disc * next_q_target
            loss = self.loss_fn(q_est, q_tgt)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(q_est.mean().item()), float(loss.item())

    def learn(self) -> tuple[float | None, float | None]:
        if self.curr_step % self.config.sync_every == 0:
            self.sync_target()

        # gate on buffer fill, not curr_step: resume restores curr_step but
        # the buffer starts empty — learning from 32 correlated samples
        # right after resume is a Q-collapse recipe
        if len(self.memory) < max(self.config.burnin, self.config.batch_size):
            return None, None
        if self.curr_step % self.config.learn_every != 0:
            return None, None

        return self.train_step()

    # --- persistence -----------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "online": self.net.online.state_dict(),
            "target": self.net.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.net.online.load_state_dict(sd["online"])
        self.net.target.load_state_dict(sd["target"])
        if "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
        if "scaler" in sd:  # tolerant: pre-AMP bundles lack the key
            self.scaler.load_state_dict(sd["scaler"])
        self.exploration_rate = sd.get("exploration_rate", self.exploration_rate)
        self.curr_step = sd.get("curr_step", 0)
