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

> RunPod UI → Secrets → Add Secret `gcp_sa_deepemu` = contents of `/tmp/gcp-sa-deepemu.json`

**Underscores, not hyphens** — RunPod injects secrets as env vars (`RUNPOD_SECRET_<name>`), and hyphens make invalid shell identifiers. There is no file mount: the pod template sets `GCP_SA_JSON={{ RUNPOD_SECRET_gcp_sa_deepemu }}` and `entrypoint.sh` materializes it to `/secrets/gcp-sa.json` + exports `GOOGLE_APPLICATION_CREDENTIALS` itself. After uploading, the local key file can be deleted.

### 3. Build + push the container

```bash
# GHCR PAT with write:packages scope:
export GITHUB_TOKEN=ghp_...

make runpod_build         # ~5 min first time, faster on rebuilds
make smoke_lifecycle      # MANDATORY pre-pod gate: host entrypoint + SIGTERM
make runpod_smoke_lifecycle  # same gate against the built image (docker kill -s TERM)
make runpod_push          # logs in to ghcr.io, pushes ghcr.io/juan-garassino/deepemulator:latest
```

The lifecycle smokes assert that a SIGTERM'd run still lands its bundle and a
**relative** `train/latest.txt` marker in the (file://) bucket — the failure
classes that silently lose pod money.

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
- **Toggle "Stop pod when container exits"** — critical; otherwise the pod keeps billing after training finishes

Env vars:

```
GCS_BUCKET=gs://garassino-ml-artifacts
GCS_PREFIX=deepemulator/coral/run-001
MODE=train
CARTRIDGE=POKEMON CORAL
ROM_GCS_URI=gs://garassino-ml-artifacts/deepemulator/inputs/PokemonCoral.gbc
INIT_STATE_GCS_URI=gs://garassino-ml-artifacts/deepemulator/inputs/coral_init.state
ENCODER_GCS_URI=gs://.../encoders/dino/<run>      # optional — frozen-encoder training
STEPS=100000
SAVE_EVERY=10000
SYNC_EVERY_SECS=300                                 # periodic background sync interval
GCP_SA_JSON={{ RUNPOD_SECRET_gcp_sa_deepemu }}      # entrypoint materializes the key
WANDB_API_KEY=...                                   # optional
```

`entrypoint.sh` runs automatically on container start. Watch the run via `docker logs` (or RunPod's web log viewer; or wandb if keyed).

---

## After the run

The container exits cleanly when `STEPS` is reached. Artifacts reach GCS three ways, in order of resilience:

1. **Periodic background sync** every `SYNC_EVERY_SECS` (default 300s) — the only protection against SIGKILL/OOM.
2. **EXIT trap final sync** — covers normal exit and SIGTERM (pod stop). The trainee runs as a *child* of bash (never `exec`, which would destroy the traps).
3. Exit code 4 means "training finished but the FINAL SYNC FAILED" — check the pod logs before deleting it.

Marker layout: `${PREFIX}/<mode>/latest.txt` holds the latest run's **name** (relative). On `RESUME=1`, the entrypoint reads the marker, downloads the bundle into `${RUNS_ROOT}/train/<name>`, and training continues from its recorded `curr_step`.

Pull the bundle locally:

```bash
make gcs_pull_latest BUCKET=gs://garassino-ml-artifacts PREFIX=deepemulator/coral/run-001
# Downloads only the latest train bundle to ./checkpoints/deepemulator/coral/run-001/<run>/
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
| `Permission denied` writing to GCS | `GCP_SA_JSON` not in the pod env | Template must set `GCP_SA_JSON={{ RUNPOD_SECRET_gcp_sa_deepemu }}`; the entrypoint logs "GCP credentials materialized" on success |
| `403` on `list_blobs` / corpus staging | Missing bucket-level list permission | `make tf_apply` grants `roles/storage.legacyBucketReader` (conditioned roles can't grant list); re-apply if the module predates this |
| `entrypoint.sh: no prior run found, starting fresh` on what should be a resume | `train/latest.txt` missing from prefix | Expected on first run for that prefix; not an error |
| Container exits with code 4 | Training finished but the final sync failed | Artifacts may be stranded in the pod — check logs / restart before deleting |
| `wandb.init` blocks forever | `WANDB_MODE` not set and no API key | `WandbLogger` should skip; if not, set `WANDB_MODE=disabled` explicitly |
| `RuntimeError: CUDA driver mismatch` | Image's torch wheel vs pod's CUDA | Use a pod with CUDA 12.x; the image base is `nvidia/cuda:12.8.0-base` (torch wheels carry the runtime) |
| Pod keeps billing after training | "Stop pod when container exits" not toggled | Stop manually in RunPod UI; toggle for next run |
| Writes fail `403` under a different prefix | IAM condition scopes object access to `deepemulator/*` | Check `GCS_PREFIX` starts with `deepemulator/` |

---

## Tearing down

```bash
make tf_destroy           # removes the SA + IAM bindings. Bucket artifacts persist.
```

Images on ghcr.io persist permanently (zero cost). Re-run `make tf_apply` to bring the SA back when needed.
