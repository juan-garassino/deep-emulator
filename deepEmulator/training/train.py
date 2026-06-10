"""Unified training CLI.

    deepemu-train --cartridge "POKEMON RED" \\
                  --rom roms/PokemonRed.gb \\
                  --init-state states/init.state \\
                  --steps 50000 --headless

Writes a checkpoint bundle to {run_dir}/ with model.pt + metadata.json +
trajectories/ + metrics.tsv every `--save-every` steps and at exit.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
from pathlib import Path

from deepEmulator.cartridges import load_all as _load_cartridges


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-train")
    p.add_argument("--cartridge", required=True, help='e.g. "POKEMON RED"')
    p.add_argument("--rom", required=True, type=Path)
    p.add_argument("--init-state", type=Path, default=None)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--max-episode-steps", type=int, default=2048)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--algo", default="ddqn", choices=["ddqn"])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument(
        "--eps-anneal-frac",
        type=float,
        default=0.10,
        help="Linearly anneal epsilon 1.0 -> min over this fraction of --steps.",
    )
    p.add_argument(
        "--dueling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dueling V/A head (default on for new runs; --no-dueling for the classic net).",
    )
    p.add_argument("--n-step", type=int, default=3, help="n-step returns (1 = classic TD).")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed random/numpy/torch for reproducible runs; recorded in metadata.",
    )
    p.add_argument(
        "--reward-clip",
        type=float,
        default=5.0,
        help="Clamp per-step reward to [-x, x] (GB path). <=0 disables.",
    )
    p.add_argument(
        "--include-start",
        action="store_true",
        help="Re-add the start button to adapters that drop it by default.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest run under checkpoints/<slug>/ (Drive-friendly).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("checkpoints"),
        help="Parent dir holding per-cartridge run histories. Auto-resolves to MyDrive on Colab.",
    )
    p.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="Path to a DINO encoder bundle dir. If set, env is wrapped with FrozenEncoderEnv "
        "and the agent learns on latents instead of pixels.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.seed is not None:
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    _load_cartridges()
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig
    from deepEmulator.core import registry
    from deepEmulator.platforms.gameboy import PyBoyEnv
    from deepEmulator.utils.checkpoints import (
        TrajectoryWriter,
        find_latest_run,
        load_bundle,
        write_bundle,
    )
    from deepEmulator.utils.logger import MetricLogger

    AdapterCls = registry.get(args.cartridge)
    adapter_kwargs: dict = {"init_state": args.init_state}
    if args.include_start and any(
        f.name == "include_start" for f in dataclasses.fields(AdapterCls)
    ):
        adapter_kwargs["include_start"] = True
    adapter = AdapterCls(**adapter_kwargs)

    slug = args.cartridge.lower().replace(" ", "_")
    cartridge_root = args.runs_root / slug

    resume_from: Path | None = None
    if args.resume:
        # an explicit --run-dir that already holds a bundle (e.g. one the
        # RunPod entrypoint just downloaded from GCS) wins over the
        # runs-root marker scan
        if args.run_dir is not None and (args.run_dir / "metadata.json").exists():
            resume_from = args.run_dir
        else:
            resume_from = find_latest_run(cartridge_root)
        if resume_from is None:
            print(f"[deepemu-train] --resume requested but no prior run under {cartridge_root}")

    if args.run_dir is None:
        if resume_from is not None:
            args.run_dir = resume_from
        else:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_dir = cartridge_root / stamp
    args.run_dir.mkdir(parents=True, exist_ok=True)

    env = PyBoyEnv(
        adapter,
        rom_path=args.rom,
        init_state=args.init_state,
        headless=args.headless,
        max_steps=args.max_episode_steps,
        reward_clip=args.reward_clip,
    )
    inner_env = env  # pre-wrap reference for the metadata env block

    encoder_metadata: dict | None = None
    if args.encoder is not None:
        from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder

        encoder, encoder_metadata = load_frozen_encoder(args.encoder)
        env = FrozenEncoderEnv(env, encoder)
        print(
            f"[deepemu-train] using frozen encoder from {args.encoder} "
            f"(latent_dim={encoder.cfg.out_dim}, obs_shape={env.observation_space.shape})"
        )

    obs_shape = env.observation_space.shape
    config = DDQNConfig(
        gamma=args.gamma,
        exploration_anneal_steps=max(1, int(args.eps_anneal_frac * args.steps)),
        normalize_obs=len(obs_shape) == 3,  # pixels only; latents pass through
        dueling=args.dueling,
        n_step=max(1, args.n_step),
    )

    resume_state: dict | None = None
    if resume_from is not None:
        resume_state, resume_md = load_bundle(resume_from, map_location="cpu")
        net_block = resume_md.get("network")
        if net_block is None:
            print(
                "[deepemu-train] resuming a pre-fix bundle: adopting legacy settings "
                "(no dueling, no normalization, n_step=1)"
            )
            config.dueling = False
            config.normalize_obs = False
            config.n_step = 1
        else:
            # the recorded architecture wins over CLI flags — a dueling
            # state dict cannot load into a non-dueling net and vice versa
            if bool(net_block.get("dueling", False)) != config.dueling:
                print(
                    f"[deepemu-train] adopting dueling={net_block.get('dueling')} "
                    "from the resumed bundle (overrides CLI)"
                )
                config.dueling = bool(net_block.get("dueling", False))
            config.normalize_obs = bool(net_block.get("normalize_obs", config.normalize_obs))
            config.n_step = int(net_block.get("n_step", config.n_step))
            config.gamma = float(net_block.get("gamma", config.gamma))

    agent = DDQNAgent(
        obs_shape=obs_shape,
        n_actions=env.action_space.n,
        config=config,
    )

    if resume_state is not None:
        agent.load_state_dict(resume_state)
        print(
            f"[deepemu-train] resumed from {resume_from} "
            f"(curr_step={agent.curr_step}, epsilon={agent.exploration_rate:.4f})"
        )

    logger = MetricLogger(args.run_dir)
    trajectories = TrajectoryWriter(args.run_dir)
    from deepEmulator.utils.wandb_logger import WandbLogger

    wb = WandbLogger(config={
        "cartridge": args.cartridge,
        "algo": args.algo,
        "steps": args.steps,
        "encoder": str(args.encoder) if args.encoder else None,
    })

    def _save_bundle() -> None:
        extra: dict = {
            "global_steps": agent.curr_step,
            "seed": args.seed,
            # everything play/eval need to rebuild the exact same env + agent
            "env": {
                "frame_stack": getattr(inner_env, "frame_stack", None),
                "action_freq": getattr(inner_env, "action_freq", None),
                "press_ticks": getattr(inner_env, "press_ticks", None),
                "max_episode_steps": args.max_episode_steps,
                "reward_clip": getattr(inner_env, "reward_clip", None),
            },
            "network": {
                "dueling": agent.config.dueling,
                "normalize_obs": agent.config.normalize_obs,
                "n_step": agent.config.n_step,
                "gamma": agent.config.gamma,
            },
        }
        if encoder_metadata is not None:
            # copy the encoder INTO the bundle so it stays portable — an
            # absolute Colab/RunPod path is meaningless on the local machine
            import hashlib
            import shutil

            enc_dst = args.run_dir / "encoder"
            enc_dst.mkdir(parents=True, exist_ok=True)
            for name in ("encoder_only.pt", "metadata.json"):
                src = args.encoder / name
                if src.exists():
                    shutil.copy2(src, enc_dst / name)
            sha = hashlib.sha256((enc_dst / "encoder_only.pt").read_bytes()).hexdigest()
            extra["encoder"] = {
                "path": "encoder",  # bundle-relative
                "sha256": sha,
                "frozen": True,
                "latent_dim": encoder_metadata.get("vit_config", {}).get("out_dim"),
                "preprocessing": encoder_metadata.get("preprocessing"),
            }
        write_bundle(
            args.run_dir,
            agent_state=agent.state_dict(),
            cartridge_title=adapter.cartridge_title,
            cartridge_platform=adapter.platform,
            action_set=adapter.action_set,
            obs_shape=env.observation_space.shape,
            algo=args.algo,
            extra=extra,
        )

    episode = 0
    obs, info = env.reset()
    trajectories.start_episode(episode)
    print(f"[deepemu-train] {args.cartridge} | run_dir={args.run_dir} | device={agent.device}")

    try:
        while agent.curr_step < args.steps:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # done drives reset/logging only — the buffer gets terminated, so
            # time-limit truncations keep their bootstrap (Crystal/Coral only
            # ever truncate; caching done here deflated Q at every horizon)
            done = bool(terminated or truncated)

            agent.cache(obs, next_obs, action, reward, bool(terminated), bool(truncated))
            q, loss = agent.learn()
            logger.log_step(reward, loss, q)
            if q is not None and loss is not None:
                wb.log(
                    {"reward": float(reward), "loss": float(loss), "q": float(q),
                     "epsilon": float(agent.exploration_rate)},
                    step=agent.curr_step,
                )

            x, y, m = info["trajectory"]
            trajectories.write(agent.curr_step, x, y, m, action, reward)

            obs = next_obs

            if agent.curr_step > 0 and agent.curr_step % args.save_every == 0:
                _save_bundle()

            if done:
                row = logger.end_episode(
                    episode=episode, step=agent.curr_step, epsilon=agent.exploration_rate
                )
                wb.log(
                    {
                        "episode": episode,
                        "MeanReward": float(row["MeanReward"]),
                        "MeanLength": float(row["MeanLength"]),
                        "epsilon": float(agent.exploration_rate),
                    },
                    step=agent.curr_step,
                )
                print(
                    f"  ep {episode:4d} | step {agent.curr_step:7d} "
                    f"| eps {row['Epsilon']} | meanR {row['MeanReward']}"
                )
                episode += 1
                obs, info = env.reset()
                trajectories.start_episode(episode)
    finally:
        trajectories.close()
        logger.plot()  # populate plots/*.jpg (no-op without matplotlib)
        _save_bundle()
        env.close()
        wb.finish()
    print(f"[deepemu-train] done. bundle at {args.run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
