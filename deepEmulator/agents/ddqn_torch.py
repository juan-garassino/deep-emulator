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
    exploration_rate_decay: float = 0.99999975
    exploration_rate_min: float = 0.05
    deque_size: int = 100_000
    batch_size: int = 32
    gamma: float = 0.9
    learning_rate: float = 0.00025
    learning_rate_decay: float = 0.99999985
    burnin: int = 1_000
    learn_every: int = 3
    sync_every: int = 1_000


# --- network ----------------------------------------------------------------
class DDQNNet(nn.Module):
    """Dual online/target DDQN. Dispatches on obs rank:
        rank-3 (C, H, W) -> Mnih-style CNN
        rank-1 (D,)      -> 2-layer MLP (for latents from a frozen encoder)
    """

    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        if len(input_shape) == 3:
            self.mode = "cnn"
            self.online = self._build_cnn(input_shape, n_actions)
        elif len(input_shape) == 1:
            self.mode = "mlp"
            self.online = self._build_mlp(input_shape[0], n_actions)
        else:
            raise ValueError(f"unsupported obs shape {input_shape}")
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_cnn(input_shape: tuple[int, int, int], n_actions: int) -> nn.Module:
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
        return nn.Sequential(
            _conv_stack(),
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    @staticmethod
    def _build_mlp(in_dim: int, n_actions: int) -> nn.Module:
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

        self.net = DDQNNet(obs_shape, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.online.parameters(), lr=self.config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.learning_rate_decay
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory: deque = deque(maxlen=self.config.deque_size)

        self.exploration_rate = self.config.exploration_rate
        self.curr_step = 0

    # --- policy ----------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        if random.random() < self.exploration_rate:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                s = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
                q = self.net(s, mode="online")
                action = int(torch.argmax(q, dim=1).item())

        self.exploration_rate = max(
            self.config.exploration_rate_min,
            self.exploration_rate * self.config.exploration_rate_decay,
        )
        self.curr_step += 1
        return action

    # --- replay buffer ---------------------------------------------------
    def cache(self, state, next_state, action, reward, done) -> None:
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
            )
        )

    def _recall(self):
        batch = random.sample(self.memory, self.config.batch_size)
        s, ns, a, r, d = map(torch.stack, zip(*batch))
        # Float conversion happens here on the sampled batch, not in cache.
        # Preserve the [0, 255] input range that `act()` also uses — keeps
        # training and inference distributions aligned.
        s = s.float()
        ns = ns.float()
        return (
            s.to(self.device),
            ns.to(self.device),
            a.squeeze(1).to(self.device),
            r.squeeze(1).to(self.device),
            d.squeeze(1).to(self.device),
        )

    # --- learning --------------------------------------------------------
    def learn(self) -> tuple[float | None, float | None]:
        if self.curr_step % self.config.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if self.curr_step < self.config.burnin:
            return None, None
        if self.curr_step % self.config.learn_every != 0:
            return None, None
        if len(self.memory) < self.config.batch_size:
            return None, None

        s, ns, a, r, d = self._recall()
        idx = torch.arange(self.config.batch_size, device=self.device)

        q_est = self.net(s, mode="online")[idx, a]
        with torch.no_grad():
            next_q_online = self.net(ns, mode="online")
            best_a = torch.argmax(next_q_online, dim=1)
            next_q_target = self.net(ns, mode="target")[idx, best_a]
            q_tgt = r + (1.0 - d) * self.config.gamma * next_q_target

        loss = self.loss_fn(q_est, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
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
