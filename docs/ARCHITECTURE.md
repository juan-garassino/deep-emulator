# Architecture

The load-bearing seams in deepEmulator, with file:line citations so you can read the code while you read this.

## One-sentence summary

**Cartridge-adaptive RL substrate**: one `EmulatorEnv` + `CartridgeAdapter` pair per ROM, one DDQN agent that auto-dispatches on observation rank, one SSL encoder (DINO; V-JEPA planned) that can optionally wrap the env to swap pixels for latents, one portable checkpoint bundle that round-trips between Colab / RunPod / local.

## Data + control flow (training)

```
ROM file ─► PyBoy/ale-py emulator (deepEmulator/platforms/{gameboy,atari}.py)
                  │
                  ▼  pixels
        CartridgeAdapter (deepEmulator/cartridges/<game>.py)
          - reads RAM, computes reward via PhasedReward (boot/tutorial/main)
          - exposes action_set + done predicate
                  │
                  ▼  (obs, reward, term, trunc, info)
        EmulatorEnv (deepEmulator/core/env.py)
          - frame-stacked obs, max_steps cap
                  │
       ┌──────────┴──────────┐
       ▼ pixel path           ▼ encoder path (optional)
  raw (C,H,W) obs       FrozenEncoderEnv (deepEmulator/encoders/frozen_wrapper.py)
                          - normalizes to 96×96 grayscale
                          - forwards through frozen ViT (DINO) or spatiotemporal ViT (V-JEPA, Phase 1)
                          - returns (frame_stack × latent_dim,) rank-1 obs
       │                     │
       └─────────┬───────────┘
                 ▼
        DDQNAgent (deepEmulator/agents/ddqn_torch.py)
          - rank-3 obs → Mnih CNN head
          - rank-1 obs → 3-layer MLP head
          - uint8 replay buffer (F9.7 — 14GB→3.4GB)
                 │
                 ▼
        MetricLogger (deepEmulator/utils/logger.py) → metrics.tsv
        TrajectoryWriter → trajectories/episode_*.csv.gz
        WandbLogger (deepEmulator/utils/wandb_logger.py) → wandb.log (no-op if no key)
                 │
                 ▼  every --save-every steps + at exit
        write_bundle (deepEmulator/utils/checkpoints.py)
          - model.pt + metadata.json + trajectories/ + plots/
          - latest.txt marker one level up
                 │
                 ▼  (RunPod only) trap on EXIT/INT/TERM
        gcs.upload_dir → gs://garassino-ml-artifacts/deepemulator/<prefix>/
```

## The four seams

### 1. CartridgeAdapter

`deepEmulator/core/cartridge.py` — ABC. Each ROM is a subclass implementing:
- `read_game_state(emulator) -> dict` — extract RAM state.
- `compute_reward(prev, curr, emulator) -> float` — reward shape.
- `is_done(state) -> bool` — episode termination.
- `action_set` — list of integer button actions.

Registry: `deepEmulator/core/registry.py` keyed by cartridge title (`POKEMON CORAL`, `POKEMON CRYSTAL`, ...). Lookup at runtime via `registry.get(title)()`.

Cartridge swap = config swap. Same DDQN, same training loop.

### 2. DDQN obs-rank auto-dispatch

`deepEmulator/agents/ddqn_torch.py::DDQNNet.__init__` — inspects `obs_shape`:
- `(C, H, W)` → builds Mnih-style CNN (Karpathy + lixado aligned).
- `(D,)` → builds MLP head with 2 hidden layers (~512 → 512 → n_actions).

This is the seam that lets us A/B test pixel vs. encoder paths without changing agent code. Tests in `tests/test_frozen_encoder.py::test_ddqn_builds_mlp_for_rank1_obs`.

### 3. FrozenEncoderEnv + algo-dispatch

`deepEmulator/encoders/frozen_wrapper.py::FrozenEncoderEnv` wraps any `EmulatorEnv`:
- For each frame in the stack, normalize to 96×96 grayscale.
- Forward through `self.encoder` (frozen, in eval mode).
- Return concatenated rank-1 latents.

`load_frozen_encoder(encoder_dir)` reads `metadata.json`, dispatches on `metadata.get("algo", "dino")`:
- `"dino"` → instantiate `ViTTiny` (single-frame, `vit.py`).
- `"vjepa"` → instantiate spatiotemporal ViT (`vit_spatiotemporal.py`, Phase 1, not yet implemented).
- Missing `algo` field → defaults to `"dino"` for backward-compat with bundles written before the field was added.

Same artifact name (`encoder_only.pt`) across both algos, so the rest of the stack is encoder-agnostic.

### 4. Portable bundle (the Colab ↔ RunPod ↔ local seam)

`deepEmulator/utils/checkpoints.py::write_bundle` writes:

```
{run_dir}/
  model.pt          # torch.save(agent.state_dict())
  metadata.json     # cartridge, platform, action_set, obs_shape, algo,
                    # global_steps, optional encoder: {path, frozen, latent_dim}
  trajectories/episode_*.csv.gz
  metrics.tsv
  plots/{reward,loss,q}.jpg
{run_dir.parent}/latest.txt   # absolute path of most recent run (Drive/GCS-friendly text marker)
```

`run_dir` is fully configurable — no Drive coupling in `train.py` / `pretrain_dino.py` themselves. The Colab seam (`colab_train.py`, `colab_pretrain.py`) and the RunPod seam (`runpod_train.py`, `runpod_pretrain.py`) are thin wrappers that resolve `run_dir` against Drive or GCS respectively, then invoke `train.main(argv)` unchanged.

## Storage abstraction (gs:// + file://)

`deepEmulator/utils/gcs.py` exposes `download`, `upload_dir`, `download_dir`, `latest_run_uri` for two schemes:

- `gs://bucket/key` — real GCS via `google-cloud-storage`.
- `file:///abs/path` — local filesystem (used by tests + `make runpod_run_local` smokes).

Same function signatures for both. The `google-cloud-storage` import is deferred so a `file://`-only test run does not need the cloud package installed. `latest_run_uri` returns `None` (not raises) when no marker is present — first runs are not errors.

## Reward phases

`deepEmulator/core/reward.py` — `RewardPhase` + `PhasedReward` primitives. Each phase declares `is_active(state) -> bool` and `compute(prev, curr) -> float`. The first active phase wins per step.

Canonical 3-phase pattern (Crystal, Coral):

| Phase | Active when | Reward |
| --- | --- | --- |
| **boot** | `party_size == 0` | constant +0.01 + 1.0 bonus when party first appears |
| **tutorial** | `party_size ≥ 1 and badges == 0` | exploration + level + heal + stuck |
| **main** | always (fallback) | full PWhiddy shape: events + heal + badges + explore + levels + stuck |

Solves the "no gradient during title screen" problem. Without phased rewards, DDQN converges to a degenerate constant Q because boot-screen frames yield zero reward variance.

## MODE dispatcher (RunPod entrypoint)

`entrypoint.sh` reads `MODE` env var and dispatches:

| `MODE` | Branch |
| --- | --- |
| `train` (default) | `deepemu-train` with `--rom`, `--init-state`, `--encoder` from staged GCS URIs |
| `collect_frames` | `deepemu-collect-frames` writing to `/runs/<ts>/frames/` |
| `pretrain_dino` | `deepemu-pretrain-dino` against corpus staged from `CORPUS_GCS_URI` |
| `pretrain_vjepa` | `deepemu-pretrain-vjepa` (Phase 1; CLI registered, module unimplemented) |
| `self_improve` | three-flag gate (see `docs/SELF_IMPROVE.md`); spawns Claude Code if all three pass |

`trap sync_runs_out EXIT INT TERM` ensures `/runs/` is rsynced to GCS on any exit path (clean, signal, watchdog kill).

## Three-flag self-improve gate

`MODE=self_improve` AND `CLAUDE_CODE_ENABLED=1` AND `ANTHROPIC_API_KEY` must all be set. Any missing → entrypoint refuses to start. `claude` binary is present in the image (≈ +200 MB) but never invoked unless all three align. See `docs/SELF_IMPROVE.md` for guardrails, signed eval, reviewer flow.

## Deferred (Phase 1+)

- **V-JEPA module** — `deepEmulator/encoders/vjepa.py` + `vit_spatiotemporal.py` + `sequence_augmentations.py` + `training/pretrain_vjepa.py`. The CLI entry point + algo-dispatch are already wired; the module body itself is the next ML lift.
- **Action-conditioned world model** (Phase 2) — Dreamer-V3-style latent dynamics on top of V-JEPA, with RND intrinsic motivation. Sketched in the plan file; not in code yet.
- **Multi-cartridge transfer eval** (Phase 3) — same encoder + world model across Coral/Crystal/Red/Mario/Pong.

## Reference points

- `~/Code/CLAUDE.md` § "GCP architecture" — canonical cloud layout.
- `CLAUDE.md` (this repo) — repo layout, status, Phase V notes.
- `~/.claude/plans/make-the-plan-saving-bubbly-muffin.md` — full plan including deferred phases.
- `/Users/juan-garassino/Code/005-products/020-autoresearch` — upstream pattern for the container + self-improve flow.
