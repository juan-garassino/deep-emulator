# deepEmulator

Cartridge-adaptive deep-RL substrate for **Game Boy / Game Boy Color** (via [PyBoy](https://github.com/Baekalfen/PyBoy)), **Atari 2600** (via [ale-py](https://ale.farama.org/)), and **SEGA Genesis** (via [stable-retro](https://github.com/Farama-Foundation/stable-retro), planned).

Wired cartridges: **Pokemon Red** (Gen 1, DMG), **Pokemon Crystal** (Gen 2, GBC), **Atari Pong**, plus a **`GENERIC GB`** zero-reward adapter for any GB/GBC ROM (smoke-testing). Mario / Kirby / Sonic adapters on the roadmap.

One harness, many games. Swap Pokemon Red ↔ Crystal ↔ Pong by changing a config — same DDQN training loop, same portable checkpoint bundle, same local-inference path.

The training story is **Colab-train / local-infer**: train on free Colab GPUs, the bundle (model + metadata + trajectories) auto-syncs to Google Drive on every save, then locally `rclone copy` the run and watch the agent play in a visible PyBoy SDL2 window. Optional pygame overlays show the **arrow trajectory** of where the agent has been + a **live attention heatmap** of what its DINO encoder is looking at.

The perception story is **either pixel-CNN or frozen-DINO latents** — the DDQN agent auto-dispatches on observation rank, so swapping a `--encoder` flag re-routes the same agent through a ViT-tiny encoder trained from scratch on a multi-cartridge frame corpus.

## Status

Phases I + II + III complete. Suite: ~75 passing, 2 ROM-gated skips. The two skips run green once you drop a Pokemon Red ROM into `roms/` and an init save-state into `states/`. **No real ROM has been touched yet** by this codebase — the PyBoy 2.4 / ALE API surface is verified only against reference repos. See `~/.claude/plans/how-you-know-now-golden-rabbit.md` for the full plan history.

## Install

```bash
pip install -e ".[atari,play,dev]"   # local laptop install (Phase II + III stack)
# or, à la carte:
pip install -e .                     # core (Game Boy via PyBoy + headless matplotlib/seaborn)
pip install -e ".[atari]"            # + ale-py for Atari
pip install -e ".[sega]"             # + stable-retro for SEGA (planned)
pip install -e ".[play]"             # + pygame (visible local inference window only)
pip install -e ".[cloud]"            # + google-cloud-storage + wandb (RunPod images)
pip install -e ".[dev]"              # + pytest / coverage / black / flake8

# After editing pyproject.toml, regenerate uv.lock for reproducible Docker builds:
make lock
```

ROMs are user-supplied (legal dump of a cart you own). Drop them in `roms/`. PyBoy save-states (curriculum starting points like `init.state`) go in `states/`. Atari ROMs are bundled with `ale-py ≥ 0.11`.

## Lean-stack discipline

**PyTorch only.** Core runtime deps: `pyboy + torch + numpy`. No gymnasium, no stable-baselines3. The minimal `Env` protocol, `Box`/`Discrete` spaces, DDQN agent, ViT-tiny, DINO loss, attention rollout, and arrow flow viz are all vendored as small focused modules ported from upstream references (lixado/PyBoy-RL, PWhiddy/PokemonRedExperiments, facebookresearch/dino, facebookresearch/mae) — not pulled in as frameworks.

## The seven CLIs

```bash
# Train DDQN on pixels (Phase I baseline)
deepemu-train --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
              --init-state states/init.state --steps 100000 --headless

# Train DDQN on frozen DINO latents (Phase II/III)
deepemu-train --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
              --init-state states/init.state --steps 100000 --headless \
              --encoder $(cat encoders/dino/latest.txt)

# Local visible inference with live overlays
deepemu-play --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
             --init-state states/init.state \
             --ckpt $(cat checkpoints/pokemon_red/latest.txt) \
             --visible --arrows --attention

# Offline arrow trajectory viz
deepemu-visualize --trajectories $(cat checkpoints/pokemon_red/latest.txt)/trajectories \
                  --cartridge "POKEMON RED" --out arrows.png

# Assemble a multi-cartridge frame corpus for SSL pretraining
deepemu-collect-frames --cartridges "POKEMON RED" "ATARI PONG" \
                       --rom roms/PokemonRed.gb roms/pong.bin \
                       --init-state states/init.state \
                       --frames 50000 --out data/frames/multi

# Pretrain a ViT-tiny with DINO self-supervised on the corpus
deepemu-pretrain-dino --corpus data/frames/multi --steps 50000 --batch-size 64

# Render the encoder's attention rollout as an animated GIF
deepemu-attention --encoder $(cat encoders/dino/latest.txt) \
                  --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
                  --init-state states/init.state \
                  --steps 300 --out attention.gif

# Head-to-head: baseline pixel-DDQN vs frozen-DINO-DDQN, HTML report
deepemu-eval --baseline checkpoints/pokemon_red/<pixel_run> \
             --treatment checkpoints/pokemon_red/<dino_run> \
             --cartridge "POKEMON RED" --rom roms/PokemonRed.gb \
             --init-state states/init.state \
             --episodes 20 --out eval_report.html
```

## Install on Colab without GitHub

Notebooks pip-install via `git+https://github.com/...` by default, but if you haven't pushed the repo (or it's private), use the Drive-sync path: rsync this repo folder into Drive once, then in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q /content/drive/MyDrive/deepEmulator
```

Re-runs after a session disconnect just re-mount Drive — no re-install needed if the runtime is reused.

## Colab → local workflow

Each Colab notebook uses Drive-mount + a `latest.txt` marker so re-running the cell after a session disconnect auto-resumes.

| Notebook | What it does |
|----------|--------------|
| `notebooks/01_colab_train.ipynb` | Train DDQN on Pokemon Red, headless |
| `notebooks/02_local_inference.ipynb` | Pull a bundle locally, run `deepemu-play --visible --arrows --attention` |
| `notebooks/03_visualize_arrows.ipynb` | Render arrow flow PNG over the Pokemon Red global map |
| `notebooks/04_pretrain_dino_colab.ipynb` | SSL pretrain ViT-tiny on a multi-cartridge corpus |
| `notebooks/05_frozen_encoder_ddqn.ipynb` | DDQN on frozen DINO latents |
| `notebooks/06_visualize_attention.ipynb` | Attention-rollout GIF |
| `notebooks/07_make_crystal_init.ipynb` | Manual + scripted procedures for the Crystal init.state. Coral section appended. |
| `notebooks/08_colab_train_coral.ipynb` | Coral-tuned Colab training with pre-flight dry-run + honest expectations |
| `notebooks/09_colab_train_coral_via_make.ipynb` | Same as 08, but orchestrated through `make` commands instead of the Python API |

Local pull (e.g. via rclone): `rclone copy gdrive:deepEmulator/checkpoints/pokemon_red ./checkpoints/pokemon_red`.

## Nano end-to-end (no ROM, no Colab)

A synthetic-env simulation of the full pipeline — useful for first-time setup verification:

```bash
PYTHONPATH=. python scripts/nano_e2e.py --out /tmp/deepemu_nano_e2e
open /tmp/deepemu_nano_e2e/eval_report.html
open /tmp/deepemu_nano_e2e/attention.gif
```

Runs in ~175 s on CPU. Produces every artifact the real pipeline produces (DINO encoder bundle, DDQN bundle with encoder linkage, eval report, attention GIF).

## Pokemon Crystal

Crystal is **Game Boy Color**, not DMG. The luminance preprocessing in `PyBoyEnv._grab_frame` handles the color → grayscale conversion automatically. A `PokemonCrystalAdapter` reads the Gen 2 RAM layout (party count, badges across two bitmasks for Johto + Kanto, `(map_group, map_number)` two-byte map identifier, event flags, in-battle flag) ported from [pret/pokecrystal](https://github.com/pret/pokecrystal).

**Target ROM**: Pokemon Crystal v1.1 US (sha1 `cd1438c79d6efabd54312c80fb826f4f0eaa3924`). v1.0 RAM layout is functionally identical for the variables we read.

```bash
# 1. Drop your ROM at roms/PokemonCrystal.gbc
# 2. Make an init.state (see notebooks/07_make_crystal_init.ipynb for manual + scripted paths)

# 3. Train
deepemu-train --cartridge "POKEMON CRYSTAL" --rom roms/PokemonCrystal.gbc \
              --init-state states/crystal_init.state --steps 50000 --headless

# 4. Watch it play
deepemu-play  --cartridge "POKEMON CRYSTAL" --rom roms/PokemonCrystal.gbc \
              --init-state states/crystal_init.state \
              --ckpt $(cat checkpoints/pokemon_crystal/latest.txt) \
              --visible --arrows --attention
```

**A few addresses are marked `# VERIFY` in `cartridges/pokemon_crystal.py`** (player position + in-battle flag) — they're widely cited by the Crystal community but weren't authoritatively pinned in this session. At first ROM load, call `from deepEmulator.cartridges.pokemon_crystal import dump_state; print(dump_state(env.pyboy))` and confirm values match in-game state. Wrong addresses surface as obvious nonsense (party_count=42, badges=255, etc.).

## Phased / dynamic rewards

Real games have distinct regimes: title screen, tutorial, main game, post-game. Without phased rewards an RL agent gets **zero signal during the intro** (no party, no badges, no exploration → all reward components are 0) and can't learn to escape the title screen.

`deepEmulator/core/reward.py` provides `PhasedReward` — an ordered list of phases where the first active one provides the reward. Pokemon Crystal + Coral use three phases:

- **boot** (party_size == 0): +0.01 per action + 1.0 bonus when the agent first acquires a starter
- **tutorial** (party_size ≥ 1, badges == 0): exploration + level + heal
- **main** (any state with badges ≥ 1): full PWhiddy-style reward

Phase predicates read from the per-step `read_game_state` dict, not RAM directly, so they're cheap. A subclass-derived cartridge inherits its parent's phases automatically; override weights via dataclass fields. Diagnostic: `adapter.current_phase()` tells you which phase fired on the last step.

## Train on RunPod

Cloud target: **`garassino-ml`** / `europe-west1`. Artifacts land under `gs://garassino-ml-artifacts/deepemulator/`. Container images live at `ghcr.io/juan-garassino/deepemulator`. Show-and-destroy via the `infra/` Terraform module (`make tf_apply` / `make tf_destroy`). See workspace root `CLAUDE.md` § "GCP architecture" for the canonical layout.

Containerized GPU training with bundles persisted to Google Cloud Storage. Mirrors the autoresearch pattern at `/Users/juan-garassino/Code/005-products/020-autoresearch`.

```bash
make runpod_build                      # docker build the image locally
make runpod_run_local MODE=train STEPS=500   # smoke test against a file:// surrogate GCS

# Push to GitHub Container Registry (requires GITHUB_TOKEN with write:packages):
export GITHUB_TOKEN=ghp_...
make runpod_push                       # image -> ghcr.io/juan-garassino/deepemulator:latest
```

**Pod env-var contract:**

| Env var | Required | Default | Notes |
| --- | --- | --- | --- |
| `GCS_BUCKET` | yes | — | `gs://garassino-ml-artifacts` (canonical) or `file:///tmp/...` for local smokes |
| `GCS_PREFIX` | yes | — | e.g. `coral/run-001` — bundles land at `${GCS_BUCKET}/${GCS_PREFIX}/` |
| `MODE` | no | `train` | `train` \| `collect_frames` \| `pretrain_dino` \| `pretrain_vjepa` \| `self_improve` |
| `CARTRIDGE` | no | `POKEMON CORAL` | adapter title |
| `ROM_GCS_URI` | mode-dep | — | `gs://.../PokemonCoral.gbc` — staged to `/data/` on boot |
| `INIT_STATE_GCS_URI` | mode-dep | — | save-state for the cartridge |
| `ENCODER_GCS_URI` | optional | — | frozen DINO/V-JEPA bundle to wrap the env with |
| `CORPUS_GCS_URI` | for pretrain | — | FrameStorage corpus directory |
| `STEPS` | no | `5000` | training step count |
| `SAVE_EVERY` | no | `1000` | checkpoint cadence |
| `RESUME` | no | `1` | `0` to force fresh; missing `latest.txt` is **not** an error |
| `WANDB_API_KEY` | optional | unset | if set, streams metrics to wandb; if unset, `WANDB_MODE=disabled` |

When creating the pod in RunPod's UI, **toggle "Stop pod when container exits"** — RunPod does not auto-stop the pod when `entrypoint.sh` finishes, and an idle GPU pod keeps billing.

**Reproducible image builds.** `uv.lock` is checked into the repo and the Dockerfile installs against it via `uv export --frozen`. After editing `pyproject.toml`, run `make lock` and commit the regenerated `uv.lock` in the same PR — otherwise the lockfile and `pyproject.toml` will drift and `make runpod_build` will fail.

**Live metrics during training.** Set `WANDB_API_KEY` in the pod env and metrics stream to wandb (project `deepemulator` by default, run name `${GCS_PREFIX}-${RUN_ID}`). Without the key, training falls back to `metrics.tsv` only (no wandb traffic, no errors).

Pull the latest bundle locally after a run:

```bash
make gcs_pull_latest BUCKET=gs://garassino-ml-artifacts PREFIX=deepemulator/coral/run-001
```

### Optional: self-improvement via Claude Code

`MODE=self_improve` enables an autonomous loop where Claude Code edits the codebase inside the pod, scores changes against `deepemu-eval`, and pushes a `claude-self-improve-*` branch for human review. **Default OFF.** Three independent env vars must all be set:

```
MODE=self_improve
CLAUDE_CODE_ENABLED=1
ANTHROPIC_API_KEY=sk-ant-...
```

If any is missing, the container runs the deterministic path instead. Guardrails: `MAX_ITERATIONS` (default 10), `MAX_WALLCLOCK_HOURS` (default 12), read-only baseline mount, SHA256-signed eval rows via `scripts/eval_signed.py`, branch isolation (never touches `master`). This mode mirrors `/Users/juan-garassino/Code/005-products/020-autoresearch` — see that repo for the upstream pattern. **Honest risk:** Goodhart-on-eval, GPU cost variance, no real sandbox. Treat self-improve branches as untrusted until reviewed.

**How to turn it off completely:** leave `CLAUDE_CODE_ENABLED` unset (the default). `claude` is never invoked.

## Telegram notifications from CI

Two GitHub Actions workflows push notifications to a Telegram chat:

- `.github/workflows/ci.yml` — fires on **failure only** (no noise on green builds).
- `.github/workflows/review_self_improve.yml` — fires **always** when a `claude-self-improve-*` branch is pushed, because every self-improve run is review-critical.

Notifications are opt-in via repo secrets. Steps:

1. **Create a bot** — message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the prompts, copy the bot token (looks like `1234567890:ABC...`).
2. **Get your chat ID** — send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and find `"chat":{"id":NNN}`. For a group/channel, add the bot to it and use the negative chat ID.
3. **Add GitHub secrets** — repo → Settings → Secrets and variables → Actions:
   - `TELEGRAM_TOKEN` — the bot token
   - `TELEGRAM_TO` — the chat ID
4. **Done.** Both workflows auto-skip the notify step when `TELEGRAM_TOKEN` is unset, so the secrets are truly optional.

## Adding a cartridge

1. Implement `CartridgeAdapter` (see `deepEmulator/core/cartridge.py` + `deepEmulator/cartridges/pokemon_red.py` as a reference).
2. Register: `@register("CARTRIDGE TITLE") class MyAdapter(CartridgeAdapter): ...`.
3. For Game Boy, the registry auto-dispatches by ROM header title (`pyboy.cartridge_title`). For Atari, declare `rom_name` and let `ale_py.roms` resolve it.

## References

- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator (we use `pyboy 2.4`)
- [ale-py](https://ale.farama.org/) — Atari Learning Environment, ROMs bundled in 0.11+
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — reference PyBoy-RL pipeline + arrow viz (`visualization/BetterMapVis_script_version_FLOW.py`)
- [PyBoy-RL](https://github.com/lixado/PyBoy-RL) — reference cartridge-adapter pattern + Colab guide
- [DINO](https://github.com/facebookresearch/dino) — multi-crop self-distillation, EMA teacher, centering/sharpening
- [Karpathy — Deep RL: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) — conceptual baseline for the from-scratch DDQN track
- [Abnar & Zuidema 2020](https://arxiv.org/abs/2005.00928) — attention rollout for ViT visualization
