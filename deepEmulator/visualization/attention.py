"""Attention rollout visualization for the DINO ViT encoder.

Algorithm (Abnar & Zuidema 2020):
1. Per layer, take attention (B, heads, N+1, N+1), mean over heads
2. Add identity (residual connection), renormalize rows
3. Chain layer-by-layer via matmul → cumulative CLS-to-patch attention
4. Reshape (gh × gw), upsample to original frame resolution, colormap, alpha-blend

Pure-numpy/torch. PIL is only used in the optional GIF writer below; if PIL is
not installed, callers can take the numpy frames directly.

Provides a CLI `deepemu-attention` that runs an episode using a frozen encoder
(and optionally a trained DDQN bundle) and writes the attention overlay as a GIF.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch


# --- core rollout ----------------------------------------------------------
def attention_rollout(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    """Compose per-layer attention into CLS-to-patch attention.

    Args:
        attn_maps: list of (B, heads, N+1, N+1) attention tensors (post-softmax),
            one per ViT block, in forward order.
    Returns:
        cls_attn: (B, N) tensor with CLS-token attention to each patch.
    """
    B, _, n_tok, _ = attn_maps[0].shape
    device = attn_maps[0].device
    eye = torch.eye(n_tok, device=device).unsqueeze(0)  # (1, N+1, N+1)
    result = eye.expand(B, -1, -1).clone()
    for layer in attn_maps:
        a = layer.mean(dim=1)  # avg over heads -> (B, N+1, N+1)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    return result[:, 0, 1:]  # CLS row, skip CLS->CLS self-attention


def reshape_to_grid(cls_attn: torch.Tensor, grid_size: int) -> torch.Tensor:
    """(B, N) -> (B, grid_size, grid_size). N must equal grid_size**2."""
    B, N = cls_attn.shape
    assert N == grid_size * grid_size, f"N={N} != grid_size^2={grid_size**2}"
    return cls_attn.reshape(B, grid_size, grid_size)


# --- numpy helpers ---------------------------------------------------------
def _bilinear_up(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(H, W) float -> (out_h, out_w) float via bilinear."""
    src_h, src_w = grid.shape
    y = np.linspace(0, src_h - 1, out_h)
    x = np.linspace(0, src_w - 1, out_w)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    fy = (y - y0).astype(np.float32)
    fx = (x - x0).astype(np.float32)
    g = grid.astype(np.float32)
    a = g[y0[:, None], x0[None, :]]
    b = g[y0[:, None], x1[None, :]]
    c = g[y1[:, None], x0[None, :]]
    d = g[y1[:, None], x1[None, :]]
    return (
        a * (1 - fy[:, None]) * (1 - fx[None, :])
        + b * (1 - fy[:, None]) * fx[None, :]
        + c * fy[:, None] * (1 - fx[None, :])
        + d * fy[:, None] * fx[None, :]
    )


def _heatmap_hot(values: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1] -> (H, W, 3) uint8 hot colormap: black→red→yellow→white."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(v * 3.0, 0, 1)
    g = np.clip(v * 3.0 - 1.0, 0, 1)
    b = np.clip(v * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compose_overlay(
    frame_rgb: np.ndarray,
    attention_grid: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a heatmap of attention over a color frame.

    Args:
        frame_rgb: (H, W, 3) uint8 — original-resolution frame
        attention_grid: (gh, gw) float — CLS attention reshaped to grid
        alpha: heatmap blend weight in [0, 1]
    Returns:
        (H, W, 3) uint8 overlay
    """
    H, W, _ = frame_rgb.shape
    up = _bilinear_up(attention_grid, H, W)
    # Normalize per-frame for max contrast
    mn, mx = up.min(), up.max()
    if mx > mn:
        up = (up - mn) / (mx - mn)
    else:
        up = np.zeros_like(up)
    heat = _heatmap_hot(up)
    blend = (1 - alpha) * frame_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


# --- end-to-end: encode a frame and produce the overlay --------------------
def attention_for_frame(
    encoder: "ViTTiny",  # type: ignore[name-defined]
    frame_rgb: np.ndarray,
    *,
    alpha: float = 0.5,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run encoder forward on a frame, compute rollout, return (overlay, raw_attention_grid).

    `frame_rgb` is normalized to 96×96 grayscale before the encoder, but the
    overlay is composited back onto `frame_rgb` at its original resolution.
    """
    from deepEmulator.data.frame_corpus import normalize_to_96x96

    device = device or next(encoder.parameters()).device.type
    norm = normalize_to_96x96(frame_rgb)  # (1, 96, 96)
    x = torch.from_numpy(norm).float().unsqueeze(0).to(device) / 255.0  # (1, 1, 96, 96)
    encoder.eval()
    with torch.no_grad():
        _ = encoder(x)
        maps = encoder.get_attention_maps()
    cls_attn = attention_rollout(maps)  # (1, N)
    grid = reshape_to_grid(cls_attn, encoder.grid_size)[0].cpu().numpy()
    overlay = compose_overlay(frame_rgb, grid, alpha=alpha)
    return overlay, grid


# --- GIF writer (requires PIL from [viz] extra) ----------------------------
def write_gif(frames: list[np.ndarray], path: Path | str, fps: int = 10) -> None:
    """Write a list of (H, W, 3) uint8 frames as an animated GIF."""
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("PIL not installed — run `pip install -e '.[viz]'`")
    from PIL import Image  # type: ignore

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


# --- CLI -------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-attention")
    p.add_argument("--encoder", type=Path, required=True, help="DINO encoder bundle dir")
    p.add_argument("--cartridge", required=True)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--init-state", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None, help="optional DDQN bundle for greedy policy")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", type=Path, required=True, help="output GIF path")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Late imports keep visualization unused-import-free in P0
    import importlib

    for mod in ("deepEmulator.cartridges.pokemon_red", "deepEmulator.cartridges.atari.pong"):
        try:
            importlib.import_module(mod)
        except Exception:
            pass
    from deepEmulator.core import registry
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder
    from deepEmulator.platforms.gameboy import PyBoyEnv

    encoder, _ = load_frozen_encoder(args.encoder)
    AdapterCls = registry.get(args.cartridge)
    adapter = AdapterCls(init_state=args.init_state)
    env = PyBoyEnv(
        adapter,
        rom_path=args.rom,
        init_state=args.init_state,
        headless=True,
        max_steps=args.steps + 10,
    )

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    for _ in range(args.steps):
        action = int(np.random.randint(0, env.action_space.n))
        obs, _, term, trunc, _ = env.step(action)
        frame_rgb = env.render()[:, :, :3]  # (144, 160, 3 or 4) → take RGB
        overlay, _ = attention_for_frame(encoder, frame_rgb, alpha=args.alpha)
        frames.append(overlay)
        if term or trunc:
            break
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(frames, args.out, fps=args.fps)
    print(f"[deepemu-attention] wrote {len(frames)} frames to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
