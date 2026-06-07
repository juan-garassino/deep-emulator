# RunPod operations

Concrete walkthrough for taking the repo from "just cloned" to "RunPod is training and writing bundles to GCS". Each step lists the exact command and what to verify before moving on.

Canonical cloud layout (see `~/Code/CLAUDE.md` § "GCP architecture"):

| Resource | Value |
| --- | --- |
| GCP project | `garassino-ml` |
| Region | `europe-west1` |
| Artifact bucket | `gs://garassino-ml-artifacts` |
| Project prefix | `deepemulator/` |
| Container registry | `ghcr.io/juan-garassino/deepemulator` |
| Terraform state | `gs://garassino-op-tf-state/deepemulator/` |

---

## One-time prereqs

### 1. Local tooling

```bash
# uv (already installed; verify)
uv --version

# gcloud (must be authed to garassino-ml)
gcloud auth application-default login
gcloud config set project garassino-ml

# Verify the shared state bucket exists
gsutil ls gs://garassino-op-tf-state  # should not error

# Terraform (>=1.5)
terraform version

# Docker desktop running locally (for `make runpod_build`)
docker info > /dev/null
```

### 2. Provision the RunPod-to-GCS service account

```bash
make tf_apply             # creates the SA + IAM bindings + SA key (~5s)
make tf_output_sa_key     # writes /tmp/gcp-sa-deepemu.json (gitignored)
```

The `make tf_output_sa_key` output also prints the next manual step:

> RunPod UI → Secrets → Add Secret `gcp-sa-deepemu` = contents of `/tmp/gcp-sa-deepemu.json`

After uploading, the secret is referenced by name in every pod template; the file on local disk can be deleted.

### 3. Build + push the container

```bash
# GHCR PAT with write:packages scope:
export GITHUB_TOKEN=ghp_...

make runpod_build         # ~5 min first time, faster on rebuilds
make runpod_push          # logs in to ghcr.io, pushes ghcr.io/juan-garassino/deepemulator:latest
```

### 4. Upload ROM + init.state to GCS

```bash
gsutil cp roms/PokemonCoral.gbc  gs://garassino-ml-artifacts/deepemulator/inputs/
gsutil cp states/coral_init.state gs://garassino-ml-artifacts/deepemulator/inputs/
```

`coral_init.state` is recorded once via `make play` (~5 min of manual play through Coral's intro). Without it, training stays stuck in the boot phase and the agent converges to a degenerate constant Q.

### 5. (Optional) Wandb + Telegram

```bash
# Wandb — live metrics during the run.
# Put the key in the pod env at creation time (Step 6 below).
WANDB_API_KEY=...

# Telegram — CI failures + self-improve branch pushes.
# GitHub repo → Settings → Secrets → Actions → add:
#   TELEGRAM_TOKEN   (from @BotFather)
#   TELEGRAM_TO      (chat ID — see README "Telegram notifications" section)
```

---

## Spin a pod

Create a pod in the RunPod UI (or `runpodctl create pod ...`):

- **Container image:** `ghcr.io/juan-garassino/deepemulator:latest`
- **GPU:** A4000 (cheap, ~$0.20/hr) for smoke tests; A100 for real training
- **Pod secret:** `gcp-sa-deepemu` mounted at `/secrets/gcp-sa.json`
- **Toggle "Stop pod when container exits"** — critical; otherwise the pod keeps billing after training finishes

Env vars:

```
GCS_BUCKET=gs://garassino-ml-artifacts
GCS_PREFIX=deepemulator/coral/run-001
MODE=train
CARTRIDGE=POKEMON CORAL
ROM_GCS_URI=gs://garassino-ml-artifacts/deepemulator/inputs/PokemonCoral.gbc
INIT_STATE_GCS_URI=gs://garassino-ml-artifacts/deepemulator/inputs/coral_init.state
STEPS=100000
SAVE_EVERY=10000
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-sa.json
WANDB_API_KEY=...                                   # optional
```

`entrypoint.sh` runs automatically on container start. Watch the run via `docker logs` (or RunPod's web log viewer; or wandb if keyed).

---

## After the run

The container exits cleanly when `STEPS` is reached. The `trap sync_runs_out EXIT INT TERM` in `entrypoint.sh` rsyncs `/runs/` → `gs://garassino-ml-artifacts/deepemulator/coral/run-001/`. If you toggled "Stop pod when container exits", the pod auto-stops and billing ends.

Pull the bundle locally:

```bash
make gcs_pull_latest BUCKET=gs://garassino-ml-artifacts PREFIX=deepemulator/coral/run-001
# Lands at ./checkpoints/deepemulator/coral/run-001/
```

Replay it locally with the visible PyBoy window:

```bash
deepemu-play --cartridge "POKEMON CORAL" --rom roms/PokemonCoral.gbc \
             --ckpt ./checkpoints/deepemulator/coral/run-001/run_<ts>/ \
             --arrows --attention
```

---

## Recurring runs (subsequent pods)

After the one-time setup, a new training run is just:

1. New pod with the same image + same secret + bumped `GCS_PREFIX` (e.g. `deepemulator/coral/run-002`).
2. Or same `GCS_PREFIX` with `RESUME=1` (default) — `entrypoint.sh` reads `latest.txt` from the prefix and continues from the last bundle.

---

## Cost guidance

RunPod billing (approx, EU region):

| GPU | Per-hour | Use case |
| --- | --- | --- |
| RTX A4000 | ~$0.20 | Smoke tests, container debugging |
| RTX A5000 | ~$0.40 | Short training, DINO pretrain on small corpus |
| RTX A6000 / A100 | ~$0.80–1.50 | Real training runs, V-JEPA pretraining |

GCS storage at €0.018/GB/month in `europe-west1`. A 1 GB bundle stored for a month ≈ €0.02. Egress within EU is free; egress to US RunPod pods ≈ €0.10/GB.

**Budget alert**: `garassino-ml` has a workspace-wide €25/mo alert at 40%/80%/100%. RunPod billing is separate (PayPal/credit card) and not subject to the GCP cap.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Permission denied` writing to GCS | SA key not mounted | Verify `/secrets/gcp-sa.json` exists in pod; `gsutil` needs `GOOGLE_APPLICATION_CREDENTIALS` set |
| `entrypoint.sh: no prior run found, starting fresh` on what should be a resume | `latest.txt` missing from prefix | Expected on first run for that prefix; not an error |
| `wandb.init` blocks forever | `WANDB_MODE` not set and no API key | `WandbLogger` should skip; if not, set `WANDB_MODE=disabled` explicitly |
| `RuntimeError: CUDA driver mismatch` | Image's torch wheel vs pod's CUDA | Use a pod with CUDA 12.x; the image is built against `nvidia/cuda:12.8.0-devel` |
| Pod keeps billing after training | "Stop pod when container exits" not toggled | Stop manually in RunPod UI; toggle for next run |
| `gsutil rsync` fails with `403` on writes | IAM condition on the SA scopes to `deepemulator/*` only | Check `GCS_PREFIX` starts with `deepemulator/` |

---

## Tearing down

```bash
make tf_destroy           # removes the SA + IAM bindings. Bucket artifacts persist.
```

Images on ghcr.io persist permanently (zero cost). Re-run `make tf_apply` to bring the SA back when needed.
