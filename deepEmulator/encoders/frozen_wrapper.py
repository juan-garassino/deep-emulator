"""FrozenEncoderEnv — wraps any EmulatorEnv to emit (D,) latents instead of pixels.

Each frame in the inner env's frame stack is normalized to 96×96 grayscale,
forwarded through a frozen ViT, and the per-frame latents are concatenated
into a single (frame_stack * latent_dim,) vector. Temporal ordering preserved.

The wrapped env is drop-in compatible with `DDQNAgent` (which now dispatches
on obs rank: rank-1 -> MLP head).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.env import EmulatorEnv
from deepEmulator.core.spaces import Box, Discrete
from deepEmulator.data.frame_corpus import OBS_PREPROCESSING, normalize_to_96x96
from deepEmulator.encoders.vit import ViTConfig, ViTTiny


class FrozenEncoderEnv(EmulatorEnv):
    """Wrap an EmulatorEnv with a frozen ViT encoder.

    Args:
        inner: the underlying EmulatorEnv (e.g. PyBoyEnv, AtariEnv)
        encoder: an instantiated, weight-loaded ViTTiny in eval mode
        device: where the encoder runs (default: encoder's current device)
    """

    def __init__(self, inner: EmulatorEnv, encoder: ViTTiny, device: str | None = None):
        super().__init__(inner.cartridge)
        self.inner = inner
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        if device is not None:
            self.encoder = self.encoder.to(device)
            self.device = device
        else:
            self.device = next(encoder.parameters()).device.type

        c, _, _ = inner.observation_space.shape
        self.frame_stack = c
        self.latent_dim = encoder.cfg.out_dim
        self.action_space = inner.action_space
        self.observation_space = Box(
            low=-1e6,
            high=1e6,
            shape=(c * self.latent_dim,),
            dtype=np.dtype(np.float32),
        )
        # Per-frame latent cache: each step the inner env pushes ONE new frame
        # (index 0, newest-first) — re-encoding the other frame_stack-1 frames
        # would recompute latents we already have. Only valid for per-frame
        # encoders on one-new-frame-per-step inner envs; V-JEPA-style
        # spatiotemporal encoders must bypass it.
        self._cache_ok = isinstance(encoder, ViTTiny)
        self._latents: np.ndarray | None = None  # (frame_stack, latent_dim)

    def _encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """(N, H, W) uint8 -> (N, latent_dim) float32."""
        batch = np.stack([normalize_to_96x96(f)[0] for f in frames])
        x = torch.from_numpy(batch).float().unsqueeze(1).to(self.device) / 255.0
        with torch.no_grad():
            z = self.encoder(x)
        return z.cpu().numpy().astype(np.float32)

    def _encode_full(self, pixel_obs: np.ndarray) -> np.ndarray:
        self._latents = self._encode_frames(pixel_obs[: self.frame_stack])
        return self._latents.flatten()

    def _encode_step(self, pixel_obs: np.ndarray) -> np.ndarray:
        if not self._cache_ok or self._latents is None:
            return self._encode_full(pixel_obs)
        newest = self._encode_frames(pixel_obs[:1])  # (1, latent_dim)
        self._latents = np.concatenate([newest, self._latents[:-1]], axis=0)
        return self._latents.flatten()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self.inner.reset(seed=seed)
        return self._encode_full(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, r, term, trunc, info = self.inner.step(action)
        return self._encode_step(obs), float(r), bool(term), bool(trunc), info

    def render(self) -> Any:
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()


def load_frozen_encoder(
    encoder_dir: Path | str,
    *,
    map_location: str = "cpu",
) -> tuple[ViTTiny, dict]:
    """Load a pretrained encoder bundle and return (model eval-ready, metadata).

    Dispatches on `metadata["algo"]`:
        "dino"  → existing single-frame ViTTiny  (default when key missing,
                  for backward-compat with bundles written before the field
                  was introduced).
        "vjepa" → spatiotemporal ViT (Phase 1, deepEmulator/encoders/vit_spatiotemporal.py).
    """
    import json

    encoder_dir = Path(encoder_dir)
    with open(encoder_dir / "metadata.json") as f:
        metadata = json.load(f)

    algo = metadata.get("algo", "dino")

    # preprocessing contract: FrozenEncoderEnv feeds the env OBSERVATION
    # pipeline; an encoder pretrained on a different pipeline (e.g. a
    # render()-frame corpus) would silently see a shifted distribution.
    # Bundles written before the field existed (None) load with no check.
    recorded = metadata.get("preprocessing")
    if recorded is not None and recorded != OBS_PREPROCESSING:
        raise ValueError(
            f"encoder bundle {encoder_dir} was pretrained on preprocessing="
            f"{recorded!r}, but FrozenEncoderEnv feeds {OBS_PREPROCESSING!r} — "
            "recollect the corpus with the default (obs-path) extractor"
        )

    sd = torch.load(encoder_dir / "encoder_only.pt", map_location=map_location, weights_only=False)

    if algo == "dino":
        vit_kwargs = metadata.get("vit_config", {})
        valid = ViTConfig.__dataclass_fields__.keys()
        cfg = ViTConfig(**{k: v for k, v in vit_kwargs.items() if k in valid})
        model = ViTTiny(cfg)
        model.load_state_dict(sd)
        model.eval()
        return model, metadata

    if algo == "vjepa":
        try:
            from deepEmulator.encoders.vit_spatiotemporal import (
                SpatiotemporalViT,
                SpatiotemporalViTConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                "this bundle records algo='vjepa', but the V-JEPA encoder is "
                "Phase 1 and not implemented yet — retrain with DINO or wait "
                "for encoders/vit_spatiotemporal.py to land"
            ) from e

        st_kwargs = metadata.get("vit_config", {})
        valid = SpatiotemporalViTConfig.__dataclass_fields__.keys()
        cfg = SpatiotemporalViTConfig(**{k: v for k, v in st_kwargs.items() if k in valid})
        model = SpatiotemporalViT(cfg)
        model.load_state_dict(sd)
        model.eval()
        return model, metadata

    raise ValueError(f"unknown encoder algo {algo!r} in {encoder_dir}/metadata.json")
