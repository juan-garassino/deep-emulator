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

import math
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
        # parametrizations.weight_norm (the non-deprecated API). original0 is
        # the magnitude g — fixed at 1 per DINO's norm_last_layer=True.
        self.last_linear = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_linear.parametrizations.weight.original0.data.fill_(1.0)
        self.last_linear.parametrizations.weight.original0.requires_grad = False

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
    # reference-DINO training schedules — active only when DINOTrainer gets
    # total_steps (constants otherwise, preserving library-use behavior):
    warmup_frac: float = 0.1          # linear LR warmup over this fraction
    final_lr_frac: float = 0.01       # cosine LR floor as a fraction of base
    weight_decay_end: float = 0.4     # cosine WD ramp 0.04 -> 0.4
    teacher_momentum_end: float = 1.0  # cosine EMA momentum 0.996 -> 1.0
    grad_clip: float = 3.0            # clip_grad_norm_ on student params
    freeze_last_layer_frac: float = 0.02  # zero last-layer grads early on


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
        total_steps: int | None = None,
    ):
        self.vit_cfg = vit_cfg or ViTConfig()
        self.dino_cfg = dino_cfg or DINOConfig()
        self.crop_cfg = crop_cfg or MultiCropConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # enables the warmup/cosine LR-WD-momentum schedules; None = constants
        self.total_steps = total_steps

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

        # Two param groups: weight decay never applies to biases, norms, or
        # the cls token (reference DINO get_params_groups)
        decay, no_decay = [], []
        for module in (self.student, self.student_head):
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith("cls_token"):
                    no_decay.append(p)
                else:
                    decay.append(p)
        self._trainable_params = decay + no_decay
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.dino_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.dino_cfg.learning_rate,
        )

        self.step_count = 0

    # --- schedules (reference DINO: warmup->cosine LR, cosine WD + EMA m) ----
    def _progress(self) -> float:
        return min(1.0, self.step_count / max(1, self.total_steps or 1))

    def _current_lr(self) -> float:
        base = self.dino_cfg.learning_rate
        if self.total_steps is None:
            return base
        warmup = max(1, int(self.dino_cfg.warmup_frac * self.total_steps))
        if self.step_count < warmup:
            return base * (self.step_count + 1) / warmup
        t = (self.step_count - warmup) / max(1, self.total_steps - warmup)
        floor = base * self.dino_cfg.final_lr_frac
        return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * min(1.0, t)))

    def _current_wd(self) -> float:
        if self.total_steps is None:
            return self.dino_cfg.weight_decay
        t = self._progress()
        start, end = self.dino_cfg.weight_decay, self.dino_cfg.weight_decay_end
        return end - 0.5 * (end - start) * (1 + math.cos(math.pi * t))

    def _current_teacher_momentum(self) -> float:
        if self.total_steps is None:
            return self.dino_cfg.teacher_ema_momentum
        t = self._progress()
        start, end = self.dino_cfg.teacher_ema_momentum, self.dino_cfg.teacher_momentum_end
        return end - 0.5 * (end - start) * (1 + math.cos(math.pi * t))

    def step(self, batch: torch.Tensor | "np.ndarray") -> float:  # type: ignore[name-defined]
        import numpy as np

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(self.device)

        # apply schedules before the gradient step
        lr = self._current_lr()
        wd = self._current_wd()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            if group["weight_decay"] > 0:
                group["weight_decay"] = wd

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
        # early-training stabilizers from reference DINO
        if self.total_steps is not None and self.step_count < int(
            self.dino_cfg.freeze_last_layer_frac * self.total_steps
        ):
            for p in self.student_head.last_linear.parameters():
                p.grad = None
        if self.dino_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._trainable_params, self.dino_cfg.grad_clip)
        self.optimizer.step()

        # EMA teacher update (scheduled momentum)
        m = self._current_teacher_momentum()
        update_teacher_ema(self.student, self.teacher, m)
        update_teacher_ema(self.student_head, self.teacher_head, m)

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
