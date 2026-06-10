"""Evaluation harness — eval a bundle on N episodes, compare two bundles head-to-head.

Outputs:
- `{bundle}/eval_episodes.tsv` — per-episode reward + length + steps
- `eval_report.html` — side-by-side training curves + eval stats + bundle metadata

CLI:
    deepemu-eval --baseline checkpoints/pokemon_red/<pixel_run> \
                 --treatment checkpoints/pokemon_red/<dino_run> \
                 --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
                 --init-state states/init.state --episodes 20 --out eval_report.html

Single-bundle mode (skip --treatment) runs eval-only and writes a single-column report.
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib
import io
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from deepEmulator.cartridges import load_all as _load_cartridges
from deepEmulator.utils.checkpoints import load_bundle


# --- env construction ------------------------------------------------------
def make_env(
    cartridge: str,
    rom: Path,
    init_state: Path | None,
    encoder_path: Path | None,
    headless: bool = True,
    max_steps: int = 4096,
    action_set: list | None = None,
):
    from deepEmulator.core import registry
    from deepEmulator.platforms.gameboy import PyBoyEnv

    _load_cartridges()
    AdapterCls = registry.get(cartridge)
    adapter = AdapterCls(init_state=init_state)
    if action_set is not None:
        # bundle's recorded action set wins — policy head indices must match
        adapter.action_set = list(action_set)
    env = PyBoyEnv(adapter, rom_path=rom, init_state=init_state, headless=headless, max_steps=max_steps)
    if encoder_path is not None:
        import torch as _torch

        from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder

        device = "cuda" if _torch.cuda.is_available() else "cpu"
        encoder, _ = load_frozen_encoder(encoder_path, map_location=device)
        env = FrozenEncoderEnv(env, encoder, device=device)
    return env


# --- eval one bundle -------------------------------------------------------
def evaluate_bundle(
    bundle_dir: Path,
    *,
    env_factory: Callable[[Path | None], Any],
    n_episodes: int = 20,
    epsilon: float = 0.05,
    capture_frames_per_episode: int = 10,
) -> dict:
    """Load a bundle, run N episodes, write `eval_episodes.tsv` + `eval_frames.npz`,
    return summary.

    Up to `capture_frames_per_episode` uniformly-sampled raw RGB frames are saved
    per episode into `eval_frames.npz` for downstream k-NN retrieval (F5).
    """
    from deepEmulator.agents.ddqn_torch import DDQNAgent, config_from_metadata

    agent_state, metadata = load_bundle(bundle_dir)
    encoder_path = None
    if "encoder" in metadata and metadata["encoder"].get("path"):
        encoder_path = Path(metadata["encoder"]["path"])
        if not encoder_path.is_absolute():
            encoder_path = Path(bundle_dir) / encoder_path  # bundle-relative

    env = env_factory(encoder_path)
    agent = DDQNAgent(
        obs_shape=tuple(metadata["observation_shape"]),
        n_actions=len(metadata["action_set"]),
        config=config_from_metadata(metadata),
        device="cpu",
    )
    agent.load_state_dict(agent_state)
    # fixed eval epsilon — no decay, no curr_step mutation (act(explore=False))
    agent.eval_epsilon = epsilon

    episodes: list[dict] = []
    captured_frames: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    def _grab_frame() -> np.ndarray | None:
        # FrozenEncoderEnv exposes .inner with the original pixel env
        target = getattr(env, "inner", env)
        try:
            f = target.render()
        except Exception:
            return None
        if f is None:
            return None
        return np.asarray(f)[:, :, :3]

    try:
        for ep in range(n_episodes):
            # per-episode reseed: agent.act draws from the module-global RNGs,
            # so baseline and treatment consume IDENTICAL epsilon coin-flips —
            # the whole point of the head-to-head harness
            import random as _random

            _random.seed(ep)
            np.random.seed(ep)
            torch.manual_seed(ep)
            obs, _ = env.reset(seed=ep)
            total_reward = 0.0
            length = 0
            episode_frames: list[np.ndarray] = []
            while True:
                action = agent.act(obs, explore=False)
                obs, r, term, trunc, _ = env.step(action)
                total_reward += r
                length += 1
                if capture_frames_per_episode > 0:
                    f = _grab_frame()
                    if f is not None:
                        episode_frames.append(f)
                if term or trunc:
                    break
            if episode_frames and capture_frames_per_episode > 0:
                # Uniformly sample without replacement
                k = min(capture_frames_per_episode, len(episode_frames))
                idx = rng.choice(len(episode_frames), size=k, replace=False)
                for i in idx:
                    captured_frames.append(episode_frames[int(i)])
            episodes.append({"episode": ep, "reward": total_reward, "length": length})
    finally:
        env.close()

    if captured_frames:
        # All frames should share shape (pad/truncate to common HxW if needed)
        shapes = {f.shape for f in captured_frames}
        if len(shapes) == 1:
            arr = np.stack(captured_frames)
            np.savez_compressed(bundle_dir / "eval_frames.npz", frames=arr)

    # Write per-episode TSV
    out_tsv = bundle_dir / "eval_episodes.tsv"
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["episode", "reward", "length"])
        for ep in episodes:
            w.writerow([ep["episode"], f"{ep['reward']:.4f}", ep["length"]])

    rewards = [ep["reward"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]
    summary = {
        "bundle_dir": str(bundle_dir),
        "cartridge": metadata.get("cartridge_title"),
        "algo": metadata.get("algo"),
        "encoder_path": metadata.get("encoder", {}).get("path") if "encoder" in metadata else None,
        "global_steps": metadata.get("global_steps"),
        "n_episodes": len(episodes),
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "length_mean": statistics.fmean(lengths) if lengths else 0.0,
    }
    return summary


# --- training-curve PNG ----------------------------------------------------
def _training_curve_png(bundles: list[tuple[str, Path]]) -> str | None:
    """Render an embedded PNG of training-loss / reward curves across bundles.

    Reads each bundle's `metrics.tsv` and plots `MeanReward` vs `Step`. Returns
    a base64 data URI, or None if matplotlib is missing or no data is found.
    """
    if importlib.util.find_spec("matplotlib") is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    any_data = False
    for label, bundle in bundles:
        path = bundle / "metrics.tsv"
        if not path.exists():
            continue
        # Read TSV (Step, MeanReward columns)
        steps, rewards = [], []
        with open(path) as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                try:
                    steps.append(int(row["Step"]))
                    rewards.append(float(row["MeanReward"]))
                except (KeyError, ValueError):
                    continue
        if steps:
            ax.plot(steps, rewards, label=label)
            any_data = True
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("training step")
    ax.set_ylabel("rolling mean reward (training)")
    ax.set_title("Training reward curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- k-NN frame retrieval -------------------------------------------------
def _knn_retrieval_grid(bundle_dir: Path, *, n_anchors: int = 4, n_neighbors: int = 3) -> str | None:
    """Encode eval_frames.npz through the bundle's encoder and render a
    (n_anchors × (1 + n_neighbors)) grid of (anchor, top-N similar) frames
    as a base64 PNG data URI. Returns None if no encoder or PIL absent.
    """
    if importlib.util.find_spec("PIL") is None:
        return None
    frames_path = bundle_dir / "eval_frames.npz"
    if not frames_path.exists():
        return None
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    enc_meta = metadata.get("encoder") or {}
    enc_path = enc_meta.get("path")
    if not enc_path:
        return None
    enc_dir = Path(enc_path)
    if not enc_dir.is_absolute():
        enc_dir = bundle_dir / enc_dir  # bundle-relative (portable)
    if not enc_dir.exists():
        return None

    from PIL import Image  # type: ignore

    from deepEmulator.data.frame_corpus import normalize_to_96x96
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder

    frames = np.load(frames_path)["frames"]  # (N, H, W, 3)
    if len(frames) < n_anchors + n_neighbors + 1:
        return None

    encoder, _ = load_frozen_encoder(enc_dir)
    encoder.eval()

    # Encode every frame
    latents: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(len(frames)):
            norm = normalize_to_96x96(frames[i])
            x = torch.from_numpy(norm).float().unsqueeze(0) / 255.0
            z = encoder(x).squeeze(0)
            latents.append(z)
    Z = torch.stack(latents)
    Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)  # L2 normalize for cosine sim
    sims = Z @ Z.t()  # (N, N)

    # Anchor picks: evenly spaced indices
    n = len(frames)
    anchor_idx = np.linspace(0, n - 1, n_anchors, dtype=int).tolist()

    rows: list[list[np.ndarray]] = []
    for ai in anchor_idx:
        # Top-k similar (excluding self)
        s = sims[ai].clone()
        s[ai] = -float("inf")
        top = torch.topk(s, n_neighbors).indices.tolist()
        rows.append([frames[ai]] + [frames[i] for i in top])

    # Compose grid
    H, W, _ = frames[0].shape
    cols = 1 + n_neighbors
    grid = np.zeros((n_anchors * H, cols * W, 3), dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, f in enumerate(row):
            grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = f

    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- HTML report -----------------------------------------------------------
def _format_summary(s: dict) -> str:
    encoder = (
        f"<code>{s['encoder_path']}</code>" if s.get("encoder_path") else "<i>none (pixel CNN)</i>"
    )
    gs = s.get("global_steps")
    gs_str = f"{gs:,}" if isinstance(gs, (int, float)) else "?"
    return f"""
    <table class="kv">
      <tr><th>bundle</th><td><code>{s['bundle_dir']}</code></td></tr>
      <tr><th>cartridge</th><td>{s['cartridge']}</td></tr>
      <tr><th>algo</th><td>{s['algo']}</td></tr>
      <tr><th>encoder</th><td>{encoder}</td></tr>
      <tr><th>training steps</th><td>{gs_str}</td></tr>
      <tr><th>eval episodes</th><td>{s['n_episodes']}</td></tr>
      <tr><th>reward (mean ± std)</th><td><b>{s['reward_mean']:.3f}</b> ± {s['reward_std']:.3f}</td></tr>
      <tr><th>reward [min, max]</th><td>[{s['reward_min']:.3f}, {s['reward_max']:.3f}]</td></tr>
      <tr><th>length (mean)</th><td>{s['length_mean']:.1f}</td></tr>
    </table>
    """


def write_report(
    summaries: list[dict],
    bundle_dirs: list[Path],
    out_path: Path,
    *,
    title: str = "deepEmulator eval report",
) -> Path:
    labels = ["baseline", "treatment"]
    curve = _training_curve_png(list(zip(labels[: len(bundle_dirs)], bundle_dirs)))
    summary_cells = "".join(
        f"<div class='col'><h2>{labels[i]}</h2>{_format_summary(s)}</div>"
        for i, s in enumerate(summaries)
    )
    # k-NN retrieval grids (one per bundle that has an encoder)
    knn_blocks: list[str] = []
    for i, b in enumerate(bundle_dirs):
        try:
            data_uri = _knn_retrieval_grid(b)
        except Exception as e:  # never let viz errors kill the report
            print(f"[eval] k-NN render failed for {b}: {e}")
            data_uri = None
        if data_uri is not None:
            knn_blocks.append(
                f"<div class='col'><h3>{labels[i]} — k-NN retrieval</h3>"
                f"<p class='footer'>each row: anchor + top-3 cosine-similar frames in the encoder latent space</p>"
                f"<img src='{data_uri}' alt='k-NN retrieval grid'></div>"
            )
    knn_section = (
        f"<h2>Encoder retrieval</h2><div class='row'>{''.join(knn_blocks)}</div>"
        if knn_blocks
        else ""
    )
    curve_block = (
        f'<img src="{curve}" alt="training reward curves">'
        if curve
        else "<p><i>matplotlib not installed — install with <code>pip install -e \".[viz]\"</code> for curves</i></p>"
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font: 14px -apple-system, system-ui, sans-serif; padding: 24px; max-width: 980px; margin: auto; }}
h1 {{ font-size: 20px; border-bottom: 1px solid #ccc; padding-bottom: 8px; }}
h2 {{ font-size: 16px; color: #333; }}
.row {{ display: flex; gap: 24px; margin-bottom: 32px; }}
.col {{ flex: 1; min-width: 0; }}
table.kv {{ width: 100%; border-collapse: collapse; }}
table.kv th {{ text-align: left; color: #666; font-weight: normal; width: 40%; padding: 4px 8px; vertical-align: top; }}
table.kv td {{ padding: 4px 8px; }}
table.kv tr:nth-child(odd) {{ background: #fafafa; }}
code {{ font: 12px ui-monospace, Menlo, monospace; background: #f3f3f3; padding: 1px 4px; border-radius: 3px; word-break: break-all; }}
img {{ max-width: 100%; }}
.footer {{ color: #888; font-size: 12px; margin-top: 32px; }}
</style></head><body>
<h1>{title}</h1>
<p class="footer">generated {datetime.now().isoformat(timespec='seconds')}</p>
<h2>Training reward curves</h2>
{curve_block}
<h2>Evaluation results</h2>
<div class="row">{summary_cells}</div>
{knn_section}
<p class="footer">deepEmulator — Phase II eval harness</p>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


# --- CLI -------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-eval")
    p.add_argument("--baseline", type=Path, required=True, help="DDQN bundle dir")
    p.add_argument("--treatment", type=Path, default=None, help="optional second bundle for head-to-head")
    p.add_argument("--cartridge", required=True)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--init-state", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--max-episode-steps", type=int, default=4096)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _read_metadata(bundle: Path) -> dict:
    from deepEmulator.utils.checkpoints import find_latest_run

    md_path = bundle / "metadata.json"
    if not md_path.exists():
        resolved = find_latest_run(bundle)
        if resolved is not None:
            md_path = resolved / "metadata.json"
    return json.loads(md_path.read_text())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    bundles = [args.baseline]
    if args.treatment is not None:
        bundles.append(args.treatment)

    summaries = []
    for b in bundles:
        md = _read_metadata(b)
        bundle_cart = str(md.get("cartridge_title", "")).upper()
        if bundle_cart and bundle_cart != args.cartridge.upper():
            raise ValueError(
                f"bundle {b} was trained on {bundle_cart!r} but --cartridge is {args.cartridge!r}"
            )

        def env_factory(encoder_path: Path | None, _aset=md.get("action_set")):
            return make_env(
                cartridge=args.cartridge,
                rom=args.rom,
                init_state=args.init_state,
                encoder_path=encoder_path,
                max_steps=args.max_episode_steps,
                action_set=_aset,
            )

        print(f"[eval] evaluating {b} on {args.episodes} episodes...")
        summary = evaluate_bundle(
            b, env_factory=env_factory, n_episodes=args.episodes, epsilon=args.epsilon
        )
        print(f"  mean reward: {summary['reward_mean']:.3f} ± {summary['reward_std']:.3f}")
        summaries.append(summary)

    out = write_report(summaries, bundles, args.out)
    print(f"[eval] report at {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
