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

    _load_cartridges()
    from deepEmulator.agents.ddqn_torch import DDQNAgent
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
    adapter = AdapterCls(init_state=args.init_state)

    slug = args.cartridge.lower().replace(" ", "_")
    cartridge_root = args.runs_root / slug

    resume_from: Path | None = None
    if args.resume:
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
    )

    encoder_metadata: dict | None = None
    if args.encoder is not None:
        from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder

        encoder, encoder_metadata = load_frozen_encoder(args.encoder)
        env = FrozenEncoderEnv(env, encoder)
        print(
            f"[deepemu-train] using frozen encoder from {args.encoder} "
            f"(latent_dim={encoder.cfg.out_dim}, obs_shape={env.observation_space.shape})"
        )

    agent = DDQNAgent(
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
    )

    if resume_from is not None:
        agent_state, metadata = load_bundle(resume_from, map_location=agent.device)
        agent.load_state_dict(agent_state)
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
        extra: dict = {"global_steps": agent.curr_step}
        if encoder_metadata is not None:
            extra["encoder"] = {
                "path": str(args.encoder.resolve()),
                "frozen": True,
                "latent_dim": encoder_metadata.get("vit_config", {}).get("out_dim"),
                "preprocessing": "96x96_grayscale",
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
            done = bool(terminated or truncated)

            agent.cache(obs, next_obs, action, reward, done)
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
                print(
                    f"  ep {episode:4d} | step {agent.curr_step:7d} "
                    f"| eps {row['Epsilon']} | meanR {row['MeanReward']}"
                )
                episode += 1
                obs, info = env.reset()
                trajectories.start_episode(episode)
    finally:
        trajectories.close()
        _save_bundle()
        env.close()
        wb.finish()
    print(f"[deepemu-train] done. bundle at {args.run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
