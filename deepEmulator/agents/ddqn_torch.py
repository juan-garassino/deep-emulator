"""Double-DQN agent — vendored & refactored from lixado/PyBoy-RL (agent.py, model.py).

Architecture is the classic Mnih-DQN CNN (3 convs + 2 FC), sized for our
(frame_stack, 72, 80) PyBoy observation. Linear input is auto-computed.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


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
    # n-step returns. 1 = classic one-step TD (byte-identical to the old
    # behavior). >1 aggregates R = sum(gamma^i * r_i) before insertion and
    # bootstraps with gamma^n; partial windows flush at episode end.
    n_step: int = 1


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DDQNNet(obs_shape, n_actions, dueling=self.config.dueling).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.online.parameters(), lr=self.config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory: deque = deque(maxlen=self.config.deque_size)

        self.exploration_rate = self.config.exploration_rate
        self.eval_epsilon = 0.0  # used by act(explore=False); play/eval set this
        self.curr_step = 0
        # raw (s, ns, a, r, terminated) transitions awaiting n-step maturity
        self._nstep_queue: deque = deque()

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
        if self.config.exploration_anneal_steps is not None:
            frac = min(1.0, self.curr_step / max(1, self.config.exploration_anneal_steps))
            self.exploration_rate = 1.0 - (1.0 - self.config.exploration_rate_min) * frac
        else:
            self.exploration_rate = max(
                self.config.exploration_rate_min,
                self.exploration_rate * self.config.exploration_rate_decay,
            )
        return action

    # --- replay buffer ---------------------------------------------------
    def cache(self, state, next_state, action, reward, done, truncated: bool = False) -> None:
        """Record one transition. `done` means TERMINATED (true env terminal —
        bootstrap is cut). Time-limit ends go in `truncated` instead: the value
        target still bootstraps, but the n-step window flushes.
        """
        if self.config.n_step <= 1:
            self._store(state, next_state, action, float(reward), bool(done), self.config.gamma)
            return

        self._nstep_queue.append((state, next_state, action, float(reward), bool(done)))
        if len(self._nstep_queue) == self.config.n_step:
            self._emit_window()
            self._nstep_queue.popleft()
        if done or truncated:
            # flush partial windows: each remaining start gets a shortened
            # return with discount gamma^k
            while self._nstep_queue:
                self._emit_window()
                self._nstep_queue.popleft()

    def _emit_window(self) -> None:
        """Emit the n-step (or shorter, at episode end) transition starting at
        the head of the queue: (s_0, ns_last, a_0, sum_i gamma^i r_i,
        terminated_last, gamma^k)."""
        gamma = self.config.gamma
        q = self._nstep_queue
        s0, _, a0, _, _ = q[0]
        ret = 0.0
        for i, (_, _, _, r_i, _) in enumerate(q):
            ret += (gamma**i) * r_i
        _, ns_last, _, _, term_last = q[-1]
        self._store(s0, ns_last, a0, ret, term_last, gamma ** len(q))

    def _store(self, state, next_state, action, reward: float, done: bool, discount: float) -> None:
        # F9.7: store states as native dtype (typically uint8 from PyBoy).
        # Converting to float32 here would 4x the buffer's memory footprint and
        # blow Colab's ~12 GB host RAM at default deque_size=100K. Conversion +
        # normalization happens in _recall on sampled batches instead.
        state_t = torch.from_numpy(np.ascontiguousarray(state))
        next_state_t = torch.from_numpy(np.ascontiguousarray(next_state))
        self.memory.append(
            (
                state_t,
                next_state_t,
                torch.tensor([action], dtype=torch.long),
                torch.tensor([reward], dtype=torch.float32),
                torch.tensor([done], dtype=torch.float32),
                torch.tensor([discount], dtype=torch.float32),
            )
        )

    def _recall(self):
        batch = random.sample(self.memory, self.config.batch_size)
        s, ns, a, r, d, disc = map(torch.stack, zip(*batch))
        # Float conversion (and optional /255) happens here on the sampled
        # batch, not in cache — the buffer stays uint8 (F9.7). _prep applies
        # the same transform act() uses, keeping train/inference aligned.
        return (
            self._prep(s.to(self.device)),
            self._prep(ns.to(self.device)),
            a.squeeze(1).to(self.device),
            r.squeeze(1).to(self.device),
            d.squeeze(1).to(self.device),
            disc.squeeze(1).to(self.device),
        )

    # --- learning --------------------------------------------------------
    def learn(self) -> tuple[float | None, float | None]:
        if self.curr_step % self.config.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        # gate on buffer fill, not curr_step: resume restores curr_step but
        # the buffer starts empty — learning from 32 correlated samples
        # right after resume is a Q-collapse recipe
        if len(self.memory) < max(self.config.burnin, self.config.batch_size):
            return None, None
        if self.curr_step % self.config.learn_every != 0:
            return None, None

        s, ns, a, r, d, disc = self._recall()
        idx = torch.arange(self.config.batch_size, device=self.device)

        q_est = self.net(s, mode="online")[idx, a]
        with torch.no_grad():
            next_q_online = self.net(ns, mode="online")
            best_a = torch.argmax(next_q_online, dim=1)
            next_q_target = self.net(ns, mode="target")[idx, best_a]
            # disc = gamma^k for the k-step return ending at ns (k=1 classic)
            q_tgt = r + (1.0 - d) * disc * next_q_target

        loss = self.loss_fn(q_est, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return float(q_est.mean().item()), float(loss.item())

    # --- persistence -----------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "online": self.net.online.state_dict(),
            "target": self.net.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.net.online.load_state_dict(sd["online"])
        self.net.target.load_state_dict(sd["target"])
        if "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
        self.exploration_rate = sd.get("exploration_rate", self.exploration_rate)
        self.curr_step = sd.get("curr_step", 0)
