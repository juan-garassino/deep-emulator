"""DINO objective for self-supervised ViT training on emulator frames.

Components:
- `DINOHead` — 3-layer MLP + L2 norm + weight-normalized last linear (per DINO)
- `DINOLoss` — multi-crop cross-entropy with EMA centering + sharpening
- `update_teacher_ema` — convex combination of student params into teacher
- `DINOTrainer` — wraps student + teacher ViTs + heads + optimizer + loss, exposes `step(batch)`

Designed to be trained from scratch on emulator frames (no pretrained init).
Reference: `facebookresearch/dino` — `main_dino.py` + `dino_loss.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepEmulator.encoders.augmentations import MultiCropAugment, MultiCropConfig
from deepEmulator.encoders.vit import ViTConfig, ViTTiny


# --- projection head --------------------------------------------------------
class DINOHead(nn.Module):
    """3-layer MLP + L2 norm + weight-normalized final linear.

    Output dim K should be large (DINO paper uses 65536; we use 4096 by default
    given much smaller from-scratch corpus).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_linear = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_linear.weight_g.data.fill_(1.0)
        self.last_linear.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_linear(x)


# --- loss -------------------------------------------------------------------
class DINOLoss(nn.Module):
    """Multi-crop cross-entropy with EMA centering on teacher logits."""

    def __init__(
        self,
        out_dim: int,
        n_global: int = 2,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.n_global = n_global
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_outs: list[torch.Tensor],
        teacher_outs: list[torch.Tensor],
    ) -> torch.Tensor:
        """student_outs: list of (B, K), all crops. teacher_outs: list of (B, K), global crops only."""
        s_logits = torch.stack(student_outs, dim=0) / self.student_temp  # (n_total, B, K)
        s_log_probs = F.log_softmax(s_logits, dim=-1)

        t_logits = torch.stack(teacher_outs, dim=0)  # (n_global, B, K)
        with torch.no_grad():
            t_probs = F.softmax((t_logits - self.center) / self.teacher_temp, dim=-1)

        loss = torch.zeros((), device=s_log_probs.device)
        n_pairs = 0
        for ti in range(t_probs.shape[0]):
            for si in range(s_log_probs.shape[0]):
                if si == ti:
                    continue  # skip same-crop pair
                pair = -(t_probs[ti] * s_log_probs[si]).sum(dim=-1).mean()
                loss = loss + pair
                n_pairs += 1
        loss = loss / max(n_pairs, 1)

        # update center (EMA over batch mean of teacher logits)
        with torch.no_grad():
            batch_center = t_logits.mean(dim=(0, 1), keepdim=False).unsqueeze(0)  # (1, K)
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )

        return loss


# --- teacher EMA -----------------------------------------------------------
@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """teacher_param = m * teacher + (1-m) * student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


# --- trainer wrapper -------------------------------------------------------
@dataclass
class DINOConfig:
    out_dim: int = 4096
    hidden_dim: int = 512
    bottleneck_dim: int = 128
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    center_momentum: float = 0.9
    teacher_ema_momentum: float = 0.996
    learning_rate: float = 5e-4
    weight_decay: float = 0.04


class DINOTrainer:
    """Bundles student + teacher ViTs + heads + optimizer + multi-crop + loss.

    Usage:
        trainer = DINOTrainer(ViTConfig(), DINOConfig())
        for step in range(N):
            batch = sample_frames()  # (B, 1, 96, 96) uint8 or float
            loss = trainer.step(batch)
    """

    def __init__(
        self,
        vit_cfg: ViTConfig | None = None,
        dino_cfg: DINOConfig | None = None,
        crop_cfg: MultiCropConfig | None = None,
        device: str | None = None,
    ):
        self.vit_cfg = vit_cfg or ViTConfig()
        self.dino_cfg = dino_cfg or DINOConfig()
        self.crop_cfg = crop_cfg or MultiCropConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Student trunk + head
        self.student = ViTTiny(self.vit_cfg).to(self.device)
        self.student_head = DINOHead(
            self.vit_cfg.out_dim,
            self.dino_cfg.out_dim,
            self.dino_cfg.hidden_dim,
            self.dino_cfg.bottleneck_dim,
        ).to(self.device)

        # Teacher: fresh instance with same config, load student weights.
        # (deepcopy doesn't work with weight_norm modules — known PyTorch quirk.)
        self.teacher = ViTTiny(self.vit_cfg).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head = DINOHead(
            self.vit_cfg.out_dim,
            self.dino_cfg.out_dim,
            self.dino_cfg.hidden_dim,
            self.dino_cfg.bottleneck_dim,
        ).to(self.device)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.augment = MultiCropAugment(self.crop_cfg)
        self.loss_fn = DINOLoss(
            out_dim=self.dino_cfg.out_dim,
            n_global=self.crop_cfg.n_global,
            teacher_temp=self.dino_cfg.teacher_temp,
            student_temp=self.dino_cfg.student_temp,
            center_momentum=self.dino_cfg.center_momentum,
        ).to(self.device)

        params = list(self.student.parameters()) + list(self.student_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.dino_cfg.learning_rate, weight_decay=self.dino_cfg.weight_decay
        )

        self.step_count = 0

    def step(self, batch: torch.Tensor | "np.ndarray") -> float:  # type: ignore[name-defined]
        import numpy as np

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(self.device)

        crops = self.augment(batch)  # list of (B, 1, 96, 96) float

        # Student: forward all crops
        student_outs = []
        for c in crops:
            cls = self.student(c)
            student_outs.append(self.student_head(cls))

        # Teacher: forward only global crops, no grad
        teacher_outs = []
        with torch.no_grad():
            for c in crops[: self.crop_cfg.n_global]:
                cls = self.teacher(c)
                teacher_outs.append(self.teacher_head(cls))

        loss = self.loss_fn(student_outs, teacher_outs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # EMA teacher update
        update_teacher_ema(self.student, self.teacher, self.dino_cfg.teacher_ema_momentum)
        update_teacher_ema(self.student_head, self.teacher_head, self.dino_cfg.teacher_ema_momentum)

        self.step_count += 1
        return float(loss.item())

    # --- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "student": self.student.state_dict(),
            "student_head": self.student_head.state_dict(),
            "teacher": self.teacher.state_dict(),
            "teacher_head": self.teacher_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "center": self.loss_fn.center.detach().cpu(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.student.load_state_dict(sd["student"])
        self.student_head.load_state_dict(sd["student_head"])
        self.teacher.load_state_dict(sd["teacher"])
        self.teacher_head.load_state_dict(sd["teacher_head"])
        if "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
        if "center" in sd:
            self.loss_fn.center.copy_(sd["center"].to(self.device))
        self.step_count = sd.get("step_count", 0)

    def encoder_only_state_dict(self) -> dict:
        """Just the student trunk — what RL inference loads."""
        return self.student.state_dict()
