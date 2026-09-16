# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **GCP migration note (2026-06-07):** Cloud target: **`garassino-ml`** / `europe-west1` (show-and-destroy under €25/mo workspace cap). Container image: `ghcr.io/juan-garassino/deepemulator`. Artifact prefix: `gs://garassino-ml-artifacts/deepemulator/`. Terraform state at `gs://garassino-op-tf-state/deepemulator/`; provision with `make tf_apply`, tear down with `make tf_destroy`. **Documented deviation from the workspace 'no SA keys anywhere' rule:** one scoped service-account key exists for RunPod → GCS sync because RunPod cannot federate via WIF — created by the `infra/` Terraform module, exported to `/tmp/gcp-sa-deepemu.json` (gitignored), uploaded once to RunPod as Pod-secret `gcp-sa-deepemu`, mounted in-pod at `/secrets/gcp-sa.json`. Rotated by re-running `make tf_apply`. See workspace root `CLAUDE.md` § "GCP architecture".

## What this is

`deepEmulator` is the **shared emulator/RL substrate** for training deep-RL agents across multiple console families. The target scope is **multi-platform + cartridge-adaptive**:

- **Game Boy / GBC / GBA** via [PyBoy](https://github.com/Baekalfen/PyBoy) — Pokemon Red, Mario, Kirby, etc.
- **Atari 2600** via [ale-py](https://ale.farama.org/) — Pong (shipped), Breakout / Space Invaders adapters easy to add.
- **SEGA Genesis / Master System** via [stable-retro](https://github.com/Farama-Foundation/stable-retro) — Sonic (planned).

The design goal is a **cartridge-adaptive** API: the same agent/training harness works across ROMs by swapping a per-cartridge adapter (RAM map, reward shaping, action set, done condition). The seam mirrors PyBoy's `GameWrapper*` classes and stable-retro's `data.json` / `scenario.json` integration pattern.

The headline workflow is **train on Colab, infer locally**: bundles (model + metadata + trajectories) are written to Google Drive during Colab training, pulled to the user's machine for visible/interactive inference with a live PyBoy SDL2 window, optional pygame arrow overlay, and optional live attention-heatmap overlay (if the bundle uses a DINO encoder).

There is also a **self-supervised vision track**: a ViT-tiny encoder is trained from scratch on a multi-cartridge frame corpus using the DINO objective (multi-crop + EMA teacher + centering/sharpening + reference schedules: linear-warmup→cosine LR, cosine WD 0.04→0.4, cosine teacher momentum 0.996→1.0, grad clip 3.0, early last-layer freeze; no WD on norms/biases/cls). The frozen encoder produces a 256-dim latent that DDQN consumes via a small MLP head instead of the pixel CNN. The DDQN agent auto-dispatches on observation rank (rank-3 pixels → Mnih CNN, rank-1 latent → MLP). **Preprocessing contract**: the corpus is collected from env OBSERVATION frames (`OBS_PREPROCESSING` in `data/frame_corpus.py`) — the exact distribution `FrozenEncoderEnv` feeds at RL time — and `load_frozen_encoder` hard-fails on a mismatch. The encoder is copied INTO each DDQN bundle (`{run}/encoder/`, bundle-relative path + sha256) so Colab-trained bundles replay locally; `deepemu-play --encoder` overrides. Attention maps are stashed only under `.eval()` (training-mode stash held ~6 GB during pretrain).

The active implementation plan lives at `~/.claude/plans/how-you-know-now-golden-rabbit.md`. Phases I (P0–P3) + II (P11–P17) + III (F1–F6) are complete.

The legacy ViZDoom Dueling-DQN that used to live at `deepBoyAdvanced/test.py` has been moved to `notebooks/legacy_vizdoom_dqn.py` for reference only — ViZDoom and TensorFlow are **out of scope**.

**Stack discipline: PyTorch only.** Core runtime deps are `pyboy + torch + numpy`. No gymnasium, no stable-baselines3 — the bits we need (minimal `Env` protocol, `Box` / `Discrete` spaces, DDQN agent, ViT-tiny + DINO loss + attention rollout, arrow viz) are **vendored** as small focused modules ported from upstream references rather than imported as framework deps. `pillow`, `matplotlib`, `seaborn`, `pygame` live in the `[viz]` extra; `stable-retro` in `[sega]`; `ale-py` in `[atari]`. The from-scratch DDQN track (lixado + Karpathy aligned) is the **only** algorithm track — no SB3/PPO track. Frames seen by the SSL encoder are normalized to **96×96 grayscale** regardless of source platform.

## Related RL prototypes (cross-repo)

These external repos under `~/Code/006-research-prototypes/` are the **consumers and reference implementations** this emulator should unify. When refactoring the emulator's API, check them:

| Repo | Platform / Emulator | Role |
|------|---------------------|------|
| `RL-pokemon-red-experiments` | Game Boy / PyBoy ([PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)) | **Primary GB reference.** Gymnasium env + SB3 PPO; canonical PyBoy-RL pipeline with reward shaping over Pokemon Red RAM. |
| `RL-pyboy-rl` | Game Boy / PyBoy ([lixado/PyBoy-RL](https://github.com/lixado/PyBoy-RL) — Mario, Kirby) | **Cartridge-adaptive reference.** `CustomPyBoyGym` + per-game `AISettings` is exactly the abstraction this repo needs to generalize. |
| `RL-sonic-rl` | SEGA Genesis / Gym Retro | **Primary SEGA reference.** Use as the retro/stable-retro integration template. |
| `RL-sega-atari-rl` | SEGA + Atari | Secondary SEGA reference (DQN-on-Breakout-style scripts). |
| `RL-pokemon-experiments` | None — SQL/data only | Not an RL consumer. |
| `RL-maze-reinforcement-learning`, `RL-poker-ai`, `RL-city-rl` | Custom envs | Out of scope (no emulator). |

**Canonical references:**
- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator; supports `pyboy.openai_gym` and per-cartridge `GameWrapper*` classes. Use these as the seam for cartridge adaptation.
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — the reference end-to-end PyBoy-RL pipeline (PPO + reward shaping over RAM state). Mirrored locally at `~/Code/006-research-prototypes/RL-pokemon-red-experiments`.
- [Karpathy — Deep RL: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) — foundational REINFORCE-from-pixels write-up. Use as the conceptual baseline (policy gradients, credit assignment, frame-diff input) when designing the from-scratch training loop before reaching for SB3.
- stable-retro / Gym Retro — for SEGA support; per-ROM integrations declare RAM addresses + done conditions in `data.json` / `scenario.json`.

The cartridge-adaptive layer in this repo should mirror PyBoy's GameWrapper pattern and Gym Retro's integration-file pattern: **one abstract `CartridgeAdapter` interface, concrete subclasses per ROM, swappable via config.**

## Common commands

```bash
make install              # pip install -e . -U
make install_dev          # pip install -e ".[dev,viz]" -U
make install_sega         # pip install -e ".[sega]" -U     (adds stable-retro)
# Or, for the full Phase II+III stack:
pip install -e ".[atari,viz,dev]"
make smoke                # import-smoke check
make test                 # coverage run -m pytest tests/ ; coverage report
make black                # format deepEmulator/ + tests/
make check_code           # flake8 deepEmulator
make clean                # remove build artifacts, __pycache__, .coverage

# Seven CLIs (all wired):
deepemu-train             # train DDQN; --encoder <dir> wraps env with frozen ViT
deepemu-play              # visible local inference; --arrows / --attention live overlays
deepemu-visualize         # render trajectory arrows PNG over the global map
deepemu-pretrain-dino     # SSL pretrain a ViT-tiny on a frame corpus
deepemu-pretrain-vjepa    # (Phase 1) SSL pretrain a spatiotemporal ViT on frame sequences
deepemu-attention         # render attention-rollout GIF for an episode
deepemu-eval              # head-to-head baseline vs treatment; HTML report
deepemu-collect-frames    # build a FrameStorage corpus from N cartridges

# RunPod / GCS:
make runpod_build         # docker build the training image
make runpod_push          # docker push to ghcr.io (needs GITHUB_TOKEN write:packages)
make runpod_run_local     # local docker-run with file:// GCS surrogate (smoke)
make gcs_pull_latest BUCKET=gs://... PREFIX=...   # pull bundle to ./checkpoints
make lock                 # regenerate uv.lock after editing pyproject.toml
```

Build system is `pyproject.toml` (PEP 621). `setup.py` is legacy and will be removed.

## Repo layout

```
deepEmulator/
  core/             EmulatorEnv base, CartridgeAdapter ABC (the AISettings-style
                    per-game plugin: RAM reward + action set + game state),
                    combo_action_set() (multi-button action-space generator),
                    registry, reward.py (PhasedReward), vendored spaces
  platforms/        gameboy.py (PyBoy; single-button step() + step_buttons() combo
                    multi-press with stale-release), atari.py (ale-py), synthetic.py (SYNTH BLOB —
                    ROM-free bouncing-blob env for tests + vec smokes), sega.py (planned)
  cartridges/       pokemon_red, pokemon_crystal, pokemon_coral, generic_gb, atari/pong.
                    __init__.py exposes load_all() — the ONLY way CLIs populate the
                    registry (no per-CLI import lists).
  agents/           ddqn_torch (rank-3 → CNN, rank-1 → MLP — auto-dispatch; dueling head;
                    budget-relative epsilon; n-step returns)
                    replay_buffer.py — preallocated ring, frame-dedup uint8 storage
                    (576 MB vs 3.4 GB at 100K), per-env sub-rings, n-step matured
                    in-buffer, truncation-aware bootstrap
  encoders/         vit.py (ViT-tiny), dino.py (loss + EMA teacher + trainer),
                    augmentations.py (multi-crop), frozen_wrapper.py (FrozenEncoderEnv)
  data/             frame_corpus.py (FrameRing + FrameStorage + FrameCollector)
  training/         train.py + colab_train.py (DDQN; --num-envs N spawns parallel workers)
                    vec_runner.py (EnvSpec + spawn workers + lock-step VecEnvRunner;
                    auto-reset in-reply; actions batched in the main process)
                    pretrain_dino.py + colab_pretrain.py (SSL)
                    collect_frames.py (corpus assembly)
                    eval.py (head-to-head HTML report + k-NN retrieval grid)
  inference/        play.py (visible PyBoy + live arrows + live attention overlay)
  visualization/    arrows.py (offline flow viz), attention.py (rollout + GIF)
  utils/            logger (TSV+plots), checkpoints (portable bundle + latest.txt),
                    device.py — get_device()/get_device_str() (mps → cuda → cpu;
                    the SINGLE source of device selection — never hardcode .cuda())
notebooks/          01_colab_train  02_local_inference  03_visualize_arrows
                    04_pretrain_dino_colab  05_frozen_encoder_ddqn  06_visualize_attention
                    + legacy_vizdoom_dqn.py (legacy reference)
scripts/            nano_e2e.py (synthetic-env full-pipeline simulation, no ROM)
docker/, tests/
roms/               (gitignored) user ROMs — PokemonRed.gb sha1 ea9bcae...
states/             PyBoy .state files (curriculum starting points)
checkpoints/        (gitignored) DDQN run bundles + latest.txt marker
encoders/           (gitignored) DINO encoder bundles + latest.txt marker
data/frames/        (gitignored) SSL frame corpora (FrameStorage)
data/trajectories/  (gitignored) episode CSV.gz for arrow viz

Dockerfile          RunPod GPU training image (autoresearch-style; CUDA 12.8 *base* + uv,
                    python-is-python3, non-root `runner` user)
entrypoint.sh       three-flag MODE dispatcher: train | collect_frames | pretrain_dino |
                    pretrain_vjepa | self_improve. Stages ROMs/encoder from GCS, runs the
                    trainee as a CHILD (never exec — traps must survive), periodic
                    background sync (SYNC_EVERY_SECS) + EXIT-trap final sync, materializes
                    GCP_SA_JSON -> GOOGLE_APPLICATION_CREDENTIALS, downloads the resume
                    bundle. Exit code 4 = run finished but FINAL SYNC FAILED.
.dockerignore       excludes roms/, states/, checkpoints/, etc. from the image.
program.md          Phase 0.5 autonomous-improvement prompt (default OFF, gated by
                    MODE=self_improve + CLAUDE_CODE_ENABLED=1 + ANTHROPIC_API_KEY).
scripts/eval_signed.py        SHA256-signed wrapper around deepemu-eval.
scripts/iteration_watchdog.sh wallclock (hard, pgid-kill) + iteration (advisory) caps.
scripts/lifecycle_smoke.sh    MANDATORY pre-pod gate: host entrypoint + SIGTERM mid-run,
                              asserts bundle + relative marker landed (make smoke_lifecycle).
deepEmulator/utils/gcs.py     gs:// + file:// URI storage (download/upload_dir/latest_run_name);
                              both schemes are additive-merge; retries on per-blob ops.
deepEmulator/utils/sync_runs.py  `python -m` artifact pusher used by the periodic loop +
                              EXIT trap; exits non-zero if any dir fails.
deepEmulator/training/runpod_train.py    GCS-staging wrapper around train.main (mirrors colab_train.py).
deepEmulator/training/runpod_pretrain.py GCS-staging wrapper around pretrain_dino.main / pretrain_vjepa.main.
.github/workflows/ci.yml                      CPU-only pytest (py3.10, container parity) +
                                              smoke-import job (py3.12) on push/PR.
.github/workflows/review_self_improve.yml     auto-runs on claude-self-improve-* branches.
```

## Checkpoint bundle format (Colab→local seam)

Each save writes:

```
checkpoints/{cartridge}/{run_id}/
  model.pt            # torch.save(agent.state_dict())  — online + target + opt + epsilon + curr_step
  metadata.json       # {cartridge, platform, action_set, obs_shape, algo, versions,
                      #  global_steps, optional encoder: {path, frozen, latent_dim, preprocessing}}
  trajectories/episode_*.csv.gz   # (step, x, y, map_id, action, reward)
  metrics.tsv         # MetricLogger TSV (lixado format)
  eval_episodes.tsv   # written by deepemu-eval
  eval_frames.npz     # raw RGB frames sampled during eval (used by k-NN retrieval grid)
  plots/{reward,loss,q}.jpg
checkpoints/{cartridge}/latest.txt   # absolute path of most recent run (Drive-friendly)

encoders/dino/{run_id}/
  encoder.pt          # full trainer state (student + teacher + opt + DINO center)
  encoder_only.pt     # student ViT weights only — what RL inference loads
  metadata.json       # {algo: dino, vit_config, dino_config, crop_config, hyper, versions}
  metrics.tsv         # step, loss, time
encoders/dino/latest.txt

encoders/vjepa/{run_id}/   # Phase 1 — V-JEPA spatiotemporal encoder
  encoder.pt          # full trainer state (student + EMA target + predictor + opt)
  encoder_only.pt     # student spatiotemporal ViT weights — what RL inference loads
  metadata.json       # {algo: vjepa, vit_config, vjepa_config, crop_config, hyper, versions}
  metrics.tsv         # step, loss, time
encoders/vjepa/latest.txt
```

`frozen_wrapper.py::load_frozen_encoder` dispatches on `metadata["algo"]`. **Missing `algo` defaults to `"dino"`** for backward-compat with bundles written before the field was added.

Local inference reads `metadata.json` → instantiates matching env + adapter → (optionally wraps with `FrozenEncoderEnv` if `metadata["encoder"]` is set) → runs `agent.act(obs)`. Bundles are fully portable across Colab/RunPod/local; the `latest.txt` marker lets re-run cells in a fresh Colab session auto-resume.

## RunPod + GCS path (Phase 0)

Mirrors the autoresearch container pattern at `/Users/juan-garassino/Code/005-products/020-autoresearch`. One image, one entrypoint, env-var contract dispatches between modes. Bundles are written under `${RUNS_ROOT:-/runs}` inside the container and pushed to `${GCS_BUCKET}/${GCS_PREFIX}/` by a periodic background sync (every `SYNC_EVERY_SECS`, default 300 — the only protection against SIGKILL/OOM) plus an EXIT-trap final sync (normal exit + SIGTERM). The trainee is a child process — never `exec`'d, which would destroy bash and its traps. Markers live at `${PREFIX}/<mode>/latest.txt` and hold the run **name** (relative); on `RESUME=1` the entrypoint downloads that bundle before launching train.

Storage URIs use a two-scheme convention via `deepEmulator/utils/gcs.py`:
- `gs://bucket/key` — real GCS via `google-cloud-storage` (deferred import).
- `file:///abs/path` — local filesystem, used by tests and `make runpod_run_local` smoke runs.

`latest_run_name(prefix_uri)` returns `None` when no marker is present — first-run resume is **not** an error. Old absolute-path markers degrade to their basename.

## Phase 0.5 — optional Claude-Code self-improvement (OFF by default)

Behind env-var gates (`MODE=self_improve` + `CLAUDE_CODE_ENABLED=1` + `ANTHROPIC_API_KEY` + `GITHUB_TOKEN` + `BASELINE_GCS_URI`), `entrypoint.sh` CLONES the repo (the image carries no repo state), creates `claude-self-improve-${RUN_ID}`, stages the fixed baseline from GCS, and spawns `claude --dangerously-skip-permissions -p "$(cat program.md)"` under `setsid`. The branch is PUSHED on exit. Scoring goes through `scripts/eval_signed.py`, which parses the machine-readable `eval_report.json` (written by `deepemu-eval` next to the HTML) and signs each row with a digest that binds the score to the recorded `eval_episodes.tsv` — tamper-EVIDENCE, not proof. The wallclock watchdog is the hard cap (pgid TERM→KILL); the iteration cap is advisory. `SELF_IMPROVE_DRY=1` exercises the whole git path (clone→branch→commit→push→review workflow) with zero API spend — run it before any real loop. `.github/workflows/review_self_improve.yml` hard-fails branches that touch protected files (`eval_signed.py`, `program.md`, `entrypoint.sh`, `Dockerfile`, `.github/`) or break the `[self-improve N]` commit format. See `docs/SELF_IMPROVE.md` for the reviewer guide and honest risks (Goodhart, GPU cost, no real isolation).

## Phased / dynamic rewards

`deepEmulator/core/reward.py` provides `RewardPhase` + `PhasedReward` primitives. A `PhasedReward` is an ordered list of phases; the **first active phase wins** each step. Each phase declares an `is_active(state, emulator) -> bool` predicate (cheap, reads from the `read_game_state` dict) and a `compute(prev, curr, emulator) -> float` reward function.

Why: real games go through distinct regimes. Boot/title-screen state has `party_size == 0` and no map — none of the PWhiddy-style reward signals (badges, exploration, heal) fire there. Without phased rewards, the agent has **zero gradient signal during the intro** and never learns to escape the title screen. With phased rewards, a tiny per-action reward (`boot_step_reward = 0.01`) plus a bonus when a key transition fires (`boot_acquire_bonus = 1.0` when party_size goes 0 → 1) gives the agent something to optimize.

Canonical 3-phase pattern (used by Pokemon Crystal + Coral):

| Phase | `is_active` predicate | Reward |
|---|---|---|
| **boot** | `party_size == 0` | constant +0.01 per action + 1.0 bonus when party first appears |
| **tutorial** | `party_size >= 1 and badges == 0` | exploration + level + heal + stuck |
| **main** | (always, fallback) | full PWhiddy shape: events + heal + badges + explore + levels + stuck |

Phases are checked top-down so they form a partition. The fallback phase guarantees the agent always gets *some* reward.

`cartridge.current_phase()` (string) is exposed for diagnostics — useful in `dump_state` output to confirm the agent is transitioning out of `boot` as expected.

A subclass-derived cartridge (like `PokemonCoralAdapter(PokemonCrystalAdapter)`) inherits the phase definitions automatically. To customize phase weights per cartridge, override the dataclass fields (`boot_step_reward`, `badge_weight`, etc.) — no need to rewrite the phase list.

Post-audit semantics (2026-06-10 fix round):
- **Stuck penalty is a direct per-step term**, not a totals-delta component (the old form fired once at the 600-visit threshold and *refunded* itself when the agent left the tile).
- **Boot keeps the totals baseline current** so the boot→tutorial transition doesn't pay a ~+20 lump sum of intro event flags in one step.
- **`party_size` refreshes every `_update_heal`** — heal credit survives catches (it was permanently dead after any party change).
- **`PhasedReward(strict=True)`** re-raises predicate/compute exceptions instead of swallowing them — on in `make smoke_rom` and the ROM-gated tests, so a wrong `# VERIFY` RAM address fails loudly before a paid run. Default off in training.
- **`is_done`: party wipe is a true terminal** (`faint_terminal=True` default, party≥1 and hp_fraction 0); everything else is env truncation.
- **GB action set default is 6 buttons** — `start` is opt-in via `include_start=True` / `deepemu-train --include-start` (menu spam burned ~1/7 of exploration). Old 7-action bundles keep working: play/eval rebuild the adapter from the bundle's recorded `action_set`.
- **`PyBoyEnv(reward_clip=5.0)`** clamps per-step reward (CLI `--reward-clip`, ≤0 disables).

## PyBoy-RL plugin surface (AISettings-style) + device selection

Ported from `lixado/PyBoy-RL` (see `003-knowledge-playgrounds/references/analysis/music-rl-games.md`, PyBoy-RL section). The repo already carried the game-agnostic plugin shape (`CartridgeAdapter` = AISettings interface, `PyBoyEnv` = gym wrapper, `DDQNAgent` = DDQN baseline); this port closed the two remaining gaps.

- **Multi-button combo actions.** `deepEmulator/core/cartridge.combo_action_set(buttons, max_buttons=2, include_noop=False)` builds a permutation-generated action space where each action is a *list* of PyBoy button names, with contradictory d-pad pairs (`left+right`, `up+down`) removed — the PyBoy-RL `MarioAISettings.GetActions` pattern. `PyBoyEnv.step_buttons(action)` presses a combo simultaneously (stale-release then multi-press, mirroring `CustomPyBoyGym.step`), so "hold A while running right" registers as one action instead of two serial presses. `action` is either a button-name list or an index into a combo `action_set`. The original single-button `step(int)` is unchanged — combos are strictly additive.
- **Device-agnostic torch.** `deepEmulator/utils/device.get_device()` / `get_device_str()` are the SINGLE source of device selection: **`mps → cuda → cpu`**. Juan trains on an M-series Mac (MPS) locally and RTX/RunPod (CUDA) remotely — **never hardcode `.cuda()`**. `DDQNAgent`, `training/train.py`, and `training/eval.py` all route through it. `is_cuda(device)` guards the CUDA-only fast paths (AMP `GradScaler` + fp16 autocast stay no-ops on MPS/CPU — the M-series path is plain fp32). An explicit `device=` arg (tests, forced-CPU eval) still wins; `prefer=` honors a backend only when it's actually usable.

Tests (logic/shape only, no real emulator training, all CPU-runnable): `tests/test_device.py` (preference chain + DDQN forward pass on the selected device), `tests/test_multibutton.py` (combo generation + stubbed multi-press step contract), `tests/test_reward_shaping_mock_ram.py` (RAM-address reward reads against a mock `pyboy.memory` dict).

> **NEEDS-MPS/GPU-VALIDATION.** These are shape/logic tests on CPU only — they do NOT run a real PyBoy episode, a real MPS/CUDA training step, or a combo action against a live ROM. Before trusting on hardware: (1) run the suite on the M5 Max so `get_device()` actually resolves to `mps` and a DDQN forward/backward step runs on Metal (this x86 CI host has `mps_built=True` but `mps_available=False`); (2) run a short `deepemu-train` on a real ROM using `step_buttons`/`combo_action_set` to confirm combos press together in the emulator; (3) sanity-check AMP stays off on MPS (`_amp_active is False`) and on for a CUDA pod. See the PR's *MPS-or-RunPod validation checklist*.

## Honest project status

Phases I + II + III + IV + F8 + F9 are merged and passing — ~135 tests green, 4 ROM-gated skips (Pokemon Red 100-step, Atari Pong 100-step, Crystal smoke, Crystal init-script).

**F1 closed for real**: Pokemon Coral (a Crystal-based fan romhack) is the first ROM that touched the codebase. The `GenericGameBoyAdapter` ran a 100-step random smoke against PyBoy 2.7 successfully, and `PokemonCoralAdapter` (a 10-line subclass of `PokemonCrystalAdapter`) ran 500 steps of real DDQN training locally producing a clean bundle. PyBoy 2.7 API matches what was written assuming 2.4.

**Phase IV** adds `cartridges/generic_gb.py`, `cartridges/pokemon_crystal.py` (full RAM map ported from pret/pokecrystal — note several `# VERIFY` constants), `cartridges/pokemon_coral.py`, scripted + manual init.state procedures, synthetic-packed Crystal map data for arrow viz.

**F8 phased rewards**: `core/reward.py` provides `RewardPhase` + `PhasedReward`. Crystal/Coral use boot → tutorial → main 3-phase pattern. Solves the "no gradient during title screen" problem.

**F9 Colab-readiness fixes**:
- F9.7 replay buffer uses **uint8** (not float32) — saves 14GB→3.4GB on default 100K-entry buffer. Critical for Colab's ~12GB RAM.
- F9.1 boot phase predicate uses `_prev_party_size` so the +1.0 acquire bonus actually fires on the 0→1 transition step.
- F9.2-6 ship `notebooks/08_colab_train_coral.ipynb` (Python API) + `notebooks/09_colab_train_coral_via_make.ipynb` (make orchestrated).

**The single gating factor for "agent actually learns" vs "pipeline just runs"**: a recorded `states/coral_init.state`. ~5 min of manual play through Coral's intro (procedure in notebook 07's Coral section). Without it the agent is stuck in boot phase with a flat constant reward — DDQN trains but converges to a degenerate constant Q.

**Phase V (in progress) — RunPod + GCS + V-JEPA + optional self-improve**:
- Containerized training: `Dockerfile` + `entrypoint.sh` + `scripts/iteration_watchdog.sh` + `program.md` are wired; `make runpod_build` succeeds locally.
- GCS layer: `deepEmulator/utils/gcs.py` supports both `gs://` and `file://` URIs; round-trip tests green.
- `frozen_wrapper.py` algo-dispatches DINO vs V-JEPA bundles, with backward-compat default to `dino`.
- 8th CLI registered: `deepemu-pretrain-vjepa` (the encoder module itself is Phase 1, not yet implemented — placeholder).
- CI: `.github/workflows/ci.yml` (pytest on push/PR) + `.github/workflows/review_self_improve.yml`.
- Self-improve mode is OPT-IN behind three env-var gates; `claude` binary present in the image but never invoked unless `MODE=self_improve` AND `CLAUDE_CODE_ENABLED=1` AND `ANTHROPIC_API_KEY` are all set.
- **Backlog closure round 2 DONE**: `uv.lock` committed (~542 KB, 100 packages resolved); Dockerfile uses `uv export --frozen` for byte-reproducible image rebuilds. `WandbLogger` helper (`deepEmulator/utils/wandb_logger.py`) wires `train.py` + `pretrain_dino.py` to wandb when `WANDB_API_KEY` is set, degrades to no-op otherwise. 4 new unit tests green.

**2026-06-11 audit fix round (14 commits, 228 tests green)** — a 5-agent audit found the agent effectively couldn't learn and the first pod would lose its artifacts; everything verified-fixable locally was fixed:
- *Learning*: budget-relative epsilon anneal (was ~97% random at end of budget), γ 0.9→0.99, /255 normalization, terminated-vs-truncated bootstrap, heal-reward refresh, real emulator resets, boot-exit lump-sum fix, per-step stuck penalty, reward clip, 6-button default action set, dueling head, n-step returns, grad clip, seeding.
- *Infra*: entrypoint child+traps (exec destroyed the sync trap — artifacts never reached GCS), periodic background sync, working resume chain (verified: fresh container continued from curr_step 3000), GCP_SA_JSON materialization, IAM list fix, vjepa staging guard, slimmer non-root image, pyboy pinned 2.7.0 (2.7.1 is yanked).
- *SSL*: obs-path preprocessing parity (validated at load), DINO reference schedules, attention-stash gate (~6 GB), portable in-bundle encoders, corpus diversity gates.
- *Throughput*: fused frame grab (0.84→0.28 ms), ring replay buffer (3.4 GB→576 MB at 100K), encoder on agent device + latent cache, `--num-envs` parallel workers, optional `--amp`.
- *Self-improve*: clone-on-entry/branch/push wired, JSON-backed signed eval (scores were all NaN before), `SELF_IMPROVE_DRY=1` git-path check, review-workflow protected-files gate.

**Makefile is the canonical user entrypoint** — `make help` prints all verbs grouped by stage. Typical flow:
```
make install_dev → make verify_ram → make smoke_rom → make play (record init.state)
PIXEL:    make train_pixel STEPS=100000 NUM_ENVS=8   (parallel workers; 1 = serial)
DINO:     make collect_frames → make pretrain_dino → make train_encoder
VIZ/EVAL: make visualize / make attention / make eval PIXEL=... TREAT=...
COLAB:    make sync_to_drive DRIVE=...
RUNPOD:   make runpod_build → make smoke_lifecycle → make runpod_push → spin pod
```

Vec throughput measured on the dev Mac (4 workers, Coral ROM): 171 → 359 env-steps/s (2.1x; lock-step sync + 4 cores cap it — expect closer to linear on pod CPUs).

**What's still unverified**:
- Real Colab run hasn't happened. Notebook 08/09 are ready, dry-run cell catches most failures in 30s.
- Real RunPod run hasn't happened. `make smoke_lifecycle` (host entrypoint + SIGTERM, PASSES) and the file:// resume round-trip are the validation so far; the gs://, IAM, and auth legs need the Phase-14 pod ladder (see docs/RUNPOD.md). The post-audit Dockerfile has NOT been built (docker daemon was down) — run `make runpod_build` first.
- DINO has never pretrained on a real corpus (only synthetic frames via `scripts/nano_e2e.py`); the new schedules/per-sample augs are unit-tested but not validated by a real pretrain.
- V-JEPA encoder module is Phase 1; the CLI + entrypoint now stub/guard cleanly (exit 2 BEFORE corpus staging) but the implementation is not shipped.
- Self-improve has not run end-to-end; the git path (clone→branch→commit→push→review workflow) is wired and testable via `SELF_IMPROVE_DRY=1`, which has also not yet been run against the real repo.
- Several Crystal/Coral RAM addresses marked `# VERIFY` (wMapGroup, wMapNumber, wXCoord, wYCoord, wBattleMode) need real-ROM dump_state cross-check — `make smoke_rom` now runs reward_strict=True so a wrong address fails loudly (200-step Coral boot smoke passes).
- No DDQN training run longer than 500 steps has happened on any real ROM yet; `coral_init.state` is still unrecorded (the gating factor for actual learning).
- AMP (`--amp`) is implemented but unbenchmarked — measure on the first GPU pod before enabling for long runs.
- Vec runner: 2.1x at 4 workers on the 4-core dev Mac; pod-CPU scaling unmeasured.
