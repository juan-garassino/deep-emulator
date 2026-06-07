#!/usr/bin/env python3
"""Nano end-to-end simulation of the full pipeline.

No ROM, no Colab, no GPU required. Runs in under 2 minutes on CPU.
Walks the whole train→pretrain→rl→eval→viz loop on a synthetic env so you
can sanity-check the pipeline before plugging in real ROMs.

Stages:
  1. Build a synthetic emulator (bouncing-blob frames)
  2. Collect ~500 frames into FrameStorage
  3. Pretrain a tiny ViT with DINO for ~30 iterations
  4. Train DDQN for ~50 steps on the frozen DINO latents
  5. Evaluate the DDQN bundle and write an HTML report
  6. Render a 10-frame attention overlay GIF

Run:
    python scripts/nano_e2e.py [--out /tmp/nano_e2e]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from dataclasses import dataclass, field

from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig
from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.env import EmulatorEnv
from deepEmulator.core.spaces import Box, Discrete
from deepEmulator.data.frame_corpus import FrameCollector, FrameStorage
from deepEmulator.encoders.augmentations import MultiCropConfig
from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder
from deepEmulator.encoders.vit import ViTConfig
from deepEmulator.training.eval import evaluate_bundle, write_report
from deepEmulator.utils.checkpoints import write_bundle
from deepEmulator.visualization.attention import attention_for_frame, write_gif


# --- synthetic emulator: bouncing blob ----------------------------------------
class SyntheticEmulatorEnv(EmulatorEnv):
    """A pixel-CNN-compatible env with no external deps. The 'game' is a
    bouncing 8×8 blob on a 144×160 canvas; actions nudge its target velocity.
    The reward equals the dx-component of the blob's velocity (so the agent
    learns to move right). Done after `episode_length` steps.
    """

    def __init__(self, cartridge: CartridgeAdapter, *, episode_length: int = 50, seed: int = 0):
        super().__init__(cartridge)
        self.h, self.w = 144, 160
        self.frame_stack = 3
        self.episode_length = episode_length
        self.action_space = Discrete(7)
        self.observation_space = Box(0, 255, (self.frame_stack, self.h // 2, self.w // 2))
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.x = self.w // 2
        self.y = self.h // 2
        self.vx = 0
        self.vy = 0
        self.t = 0
        self._stack = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _render_frame_color(self) -> np.ndarray:
        canvas = np.full((self.h, self.w, 3), 30, dtype=np.uint8)
        # Background gradient (gives the encoder texture)
        canvas[..., 0] += (np.linspace(0, 60, self.h)[:, None]).astype(np.uint8)
        canvas[..., 1] += (np.linspace(0, 60, self.w)[None, :]).astype(np.uint8)
        # Bouncing blob
        x0, y0 = max(0, int(self.x) - 4), max(0, int(self.y) - 4)
        x1, y1 = min(self.w, int(self.x) + 4), min(self.h, int(self.y) + 4)
        canvas[y0:y1, x0:x1] = [240, 220, 80]
        return canvas

    def _downscaled_gray(self) -> np.ndarray:
        rgb = self._render_frame_color()
        gray = rgb.mean(axis=-1).astype(np.uint8)
        return gray.reshape(self.h // 2, 2, self.w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)

    def _push(self, frame: np.ndarray) -> None:
        self._stack = np.roll(self._stack, 1, axis=0)
        self._stack[0] = frame

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        first = self._downscaled_gray()
        for _ in range(self.frame_stack):
            self._push(first)
        return self._stack.copy(), {"game_state": {"x": self.x, "y": self.y, "map_id": 0}}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Action 0=noop, 1=left, 2=right, 3=up, 4=down, 5/6=ignored
        dvx = {1: -1, 2: 1}.get(action, 0)
        dvy = {3: -1, 4: 1}.get(action, 0)
        self.vx = int(np.clip(self.vx + dvx, -3, 3))
        self.vy = int(np.clip(self.vy + dvy, -3, 3))
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x <= 4 or self.x >= self.w - 4:
            self.vx = -self.vx
        if self.y <= 4 or self.y >= self.h - 4:
            self.vy = -self.vy
        self.x = int(np.clip(self.x, 4, self.w - 4))
        self.y = int(np.clip(self.y, 4, self.h - 4))
        self._push(self._downscaled_gray())
        self.t += 1
        reward = float(self.vx) * 0.1
        done = self.t >= self.episode_length
        return (
            self._stack.copy(),
            reward,
            False,
            done,
            {"game_state": {"x": self.x, "y": self.y, "map_id": 0}, "trajectory": (self.x, self.y, 0)},
        )

    def render(self) -> np.ndarray:
        return self._render_frame_color()

    def close(self) -> None:
        pass


@dataclass
class SyntheticCartridge(CartridgeAdapter):
    cartridge_title: str = "SYNTH BLOB"
    platform: str = "gameboy"
    action_set: list = field(default_factory=lambda: list(range(7)))
    observation_shape: tuple = (3, 72, 80)

    def reset_episode(self, emulator) -> None:
        pass

    def read_game_state(self, emulator) -> dict:
        return {"x": 0, "y": 0, "map_id": 0}

    def compute_reward(self, prev_state, curr_state, emulator) -> float:
        return 0.0

    def is_done(self, state: dict) -> bool:
        return False


# --- the pipeline -------------------------------------------------------------
def run_nano_pipeline(
    out_dir: Path,
    *,
    n_frames: int = 500,
    dino_iters: int = 30,
    rl_steps: int = 50,
    eval_episodes: int = 3,
    gif_steps: int = 10,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    torch.manual_seed(0)
    np.random.seed(0)

    cart = SyntheticCartridge()
    env = SyntheticEmulatorEnv(cart)

    # --- Stage 1: collect frames -------------------------------------------
    log(f"[1/6] collecting {n_frames} frames from synthetic env...")
    t0 = time.time()
    corpus_dir = out_dir / "corpus"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    storage = FrameStorage(corpus_dir, chunk_size=128)
    coll = FrameCollector([env], storage, seed=0)
    coll.run(n_frames)
    storage.flush()
    log(f"      done ({storage.total_frames} frames in {time.time() - t0:.1f}s)")

    # --- Stage 2: pretrain DINO --------------------------------------------
    log(f"[2/6] pretraining tiny ViT with DINO for {dino_iters} iterations...")
    t0 = time.time()
    vit_cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    trainer = DINOTrainer(
        vit_cfg=vit_cfg,
        dino_cfg=DINOConfig(out_dim=256, hidden_dim=64, bottleneck_dim=32, learning_rate=1e-3),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    losses = []
    batch_iter = iter(storage.iter_batches(batch_size=16, shuffle=True, seed=0))
    for step in range(dino_iters):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(storage.iter_batches(batch_size=16, shuffle=True, seed=step))
            batch = next(batch_iter)
        losses.append(trainer.step(batch))
    log(f"      done ({time.time() - t0:.1f}s, first loss {losses[0]:.3f}, last loss {losses[-1]:.3f})")

    encoder_dir = out_dir / "encoders" / "dino" / "nano_run"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.encoder_only_state_dict(), encoder_dir / "encoder_only.pt")
    torch.save(trainer.state_dict(), encoder_dir / "encoder.pt")
    (encoder_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": vit_cfg.__dict__, "step_count": dino_iters}, default=str)
    )
    (encoder_dir.parent / "latest.txt").write_text(str(encoder_dir.resolve()) + "\n")
    log(f"      encoder bundle at {encoder_dir}")

    # --- Stage 3: RL on frozen latents -------------------------------------
    log(f"[3/6] training DDQN on frozen DINO latents for {rl_steps} steps...")
    t0 = time.time()
    encoder, _ = load_frozen_encoder(encoder_dir)
    rl_env = FrozenEncoderEnv(env, encoder, device="cpu")
    log(f"      obs shape: {rl_env.observation_space.shape} (was {env.observation_space.shape})")

    agent = DDQNAgent(
        obs_shape=rl_env.observation_space.shape,
        n_actions=rl_env.action_space.n,
        config=DDQNConfig(batch_size=8, deque_size=200, burnin=8, learn_every=2, sync_every=20),
        device="cpu",
    )
    obs, _ = rl_env.reset(seed=42)
    losses_rl = []
    for _ in range(rl_steps):
        action = agent.act(obs)
        next_obs, r, term, trunc, _ = rl_env.step(action)
        agent.cache(obs, next_obs, action, r, term or trunc)
        q, loss = agent.learn()
        if loss is not None:
            losses_rl.append(loss)
        obs = next_obs
        if term or trunc:
            obs, _ = rl_env.reset(seed=int(np.random.randint(0, 1_000_000)))
    log(f"      done ({time.time() - t0:.1f}s, learning steps: {len(losses_rl)})")

    # --- Stage 4: write DDQN bundle ----------------------------------------
    rl_dir = out_dir / "checkpoints" / "synth_blob" / "nano_run"
    write_bundle(
        rl_dir,
        agent_state=agent.state_dict(),
        cartridge_title=cart.cartridge_title,
        cartridge_platform=cart.platform,
        action_set=cart.action_set,
        obs_shape=rl_env.observation_space.shape,
        algo="ddqn",
        extra={
            "global_steps": agent.curr_step,
            "encoder": {
                "path": str(encoder_dir.resolve()),
                "frozen": True,
                "latent_dim": vit_cfg.out_dim,
                "preprocessing": "96x96_grayscale",
            },
        },
    )
    # Drop a minimal metrics.tsv so the eval report has a curve
    metrics_path = rl_dir / "metrics.tsv"
    metrics_path.write_text(
        "Episode\tStep\tEpsilon\tMeanReward\tMeanLength\tMeanLoss\tMeanQValue\tTimeDelta\tTime\n"
        + "\n".join(
            f"{i}\t{i*rl_steps // max(len(losses_rl), 1) or 1}\t0.5\t{losses_rl[i] if i < len(losses_rl) else 0:.3f}\t10\t{losses_rl[i] if i < len(losses_rl) else 0:.3f}\t0.3\t0.1\t{0.1*i:.2f}"
            for i in range(max(len(losses_rl), 1))
        )
    )
    log(f"      DDQN bundle at {rl_dir}")

    # --- Stage 5: eval + HTML report ---------------------------------------
    log(f"[4/6] evaluating bundle on {eval_episodes} episodes...")
    t0 = time.time()

    def env_factory(_encoder_path):
        inner = SyntheticEmulatorEnv(cart, episode_length=20, seed=99)
        if _encoder_path is None:
            return inner
        enc, _ = load_frozen_encoder(_encoder_path)
        return FrozenEncoderEnv(inner, enc, device="cpu")

    summary = evaluate_bundle(rl_dir, env_factory=env_factory, n_episodes=eval_episodes, epsilon=0.1)
    report_path = out_dir / "eval_report.html"
    write_report([summary], [rl_dir], report_path, title="nano e2e — synthetic blob")
    log(f"      mean reward: {summary['reward_mean']:.3f} ± {summary['reward_std']:.3f}")
    log(f"      report at {report_path} ({time.time() - t0:.1f}s)")

    # --- Stage 6: attention overlay GIF ------------------------------------
    log(f"[5/6] rendering attention overlay GIF ({gif_steps} frames)...")
    t0 = time.time()
    env.reset(seed=7)
    frames = []
    for _ in range(gif_steps):
        env.step(np.random.randint(0, env.action_space.n))
        overlay, _ = attention_for_frame(encoder, env.render(), alpha=0.55, device="cpu")
        frames.append(overlay)
    gif_path = out_dir / "attention.gif"
    try:
        write_gif(frames, gif_path, fps=8)
        log(f"      gif at {gif_path} ({time.time() - t0:.1f}s)")
    except RuntimeError as e:
        log(f"      skipped (PIL missing): {e}")
        gif_path = None

    # --- Stage 7: summary --------------------------------------------------
    log("[6/6] done.")
    artifacts = {
        "corpus_dir": corpus_dir,
        "encoder_dir": encoder_dir,
        "rl_bundle": rl_dir,
        "eval_report": report_path,
        "attention_gif": gif_path,
        "dino_first_loss": losses[0],
        "dino_last_loss": losses[-1],
        "rl_learning_steps": len(losses_rl),
        "eval_mean_reward": summary["reward_mean"],
    }
    if verbose:
        print("\nartifacts:")
        for k, v in artifacts.items():
            print(f"  {k}: {v}")
    return artifacts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("/tmp/deepemu_nano_e2e"))
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--dino-iters", type=int, default=30)
    p.add_argument("--rl-steps", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--gif-steps", type=int, default=10)
    args = p.parse_args()
    run_nano_pipeline(
        args.out,
        n_frames=args.frames,
        dino_iters=args.dino_iters,
        rl_steps=args.rl_steps,
        eval_episodes=args.eval_episodes,
        gif_steps=args.gif_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
