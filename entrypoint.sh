#!/usr/bin/env bash
# deepEmulator — RunPod entrypoint.
#
# Ports /Users/juan-garassino/Code/005-products/020-autoresearch/entrypoint.sh.
# Differences:
#   - Three-flag toggle for the optional self-improve mode.
#   - First-run-aware resume (missing latest.txt is NOT an error).
#   - On-exit rsync of /runs to GCS (or file:// for local smokes).
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Required env contract
# ---------------------------------------------------------------------------
: "${GCS_BUCKET:?GCS_BUCKET must be set (e.g. gs://garassino-ml-artifacts or file:///tmp/deepemu)}"
: "${GCS_PREFIX:?GCS_PREFIX must be set (e.g. deepemulator/coral/run-001)}"

MODE="${MODE:-train}"
RESUME="${RESUME:-1}"
CARTRIDGE="${CARTRIDGE:-POKEMON CORAL}"
STEPS="${STEPS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EPISODE_STEPS="${EPISODE_STEPS:-2048}"

RUNS_ROOT="/runs"
DATA_ROOT="/data"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"

# Strip trailing slashes; the URI module reassembles cleanly.
GCS_BUCKET="${GCS_BUCKET%/}"
GCS_PREFIX="${GCS_PREFIX%/}"
PREFIX_URI="${GCS_BUCKET}/${GCS_PREFIX}"

log "MODE=${MODE}  CARTRIDGE='${CARTRIDGE}'  PREFIX_URI=${PREFIX_URI}  RUN_ID=${RUN_ID}"

# ---------------------------------------------------------------------------
# wandb (optional — quiet when unset)
# ---------------------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    log "wandb: API key present, logging in"
    wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || log "wandb login failed (non-fatal)"
    export WANDB_PROJECT="${WANDB_PROJECT:-deepemulator}"
    export WANDB_NAME="${WANDB_NAME:-${GCS_PREFIX//\//-}-${RUN_ID}}"
else
    log "wandb: WANDB_API_KEY not set, disabling"
    export WANDB_MODE=disabled
fi

# ---------------------------------------------------------------------------
# Stage ROM + init state from GCS (when URIs are provided)
# ---------------------------------------------------------------------------
stage_from_uri() {
    local uri="$1"; local dest="$2"
    log "staging ${uri} -> ${dest}"
    python -c "from deepEmulator.utils import gcs; gcs.download('${uri}', '${dest}')"
}

ROM_LOCAL=""
INIT_STATE_LOCAL=""
if [ -n "${ROM_GCS_URI:-}" ]; then
    ROM_LOCAL="${DATA_ROOT}/$(basename "${ROM_GCS_URI}")"
    stage_from_uri "${ROM_GCS_URI}" "${ROM_LOCAL}"
fi
if [ -n "${INIT_STATE_GCS_URI:-}" ]; then
    INIT_STATE_LOCAL="${DATA_ROOT}/$(basename "${INIT_STATE_GCS_URI}")"
    stage_from_uri "${INIT_STATE_GCS_URI}" "${INIT_STATE_LOCAL}"
fi

# ---------------------------------------------------------------------------
# Resume — missing marker is NOT an error
# ---------------------------------------------------------------------------
RESUME_RUN_URI=""
if [ "${RESUME}" = "1" ]; then
    RESUME_RUN_URI=$(python -c "from deepEmulator.utils import gcs; print(gcs.latest_run_uri('${PREFIX_URI}') or '')")
    if [ -z "${RESUME_RUN_URI}" ]; then
        log "no prior run found at ${PREFIX_URI}/latest.txt — starting fresh"
    else
        log "resuming from ${RESUME_RUN_URI}"
    fi
fi

# ---------------------------------------------------------------------------
# Sync runs back to GCS on exit (success or signal)
# ---------------------------------------------------------------------------
sync_runs_out() {
    local rc=$?
    log "sync_runs_out: exit code ${rc}; pushing ${RUNS_ROOT}/ to ${PREFIX_URI}/"
    python -c "
import sys
from pathlib import Path
from deepEmulator.utils import gcs
root = Path('${RUNS_ROOT}')
prefix = '${PREFIX_URI}'
if not root.exists():
    print('no /runs dir, skip')
    sys.exit(0)
for child in root.iterdir():
    if child.is_dir():
        gcs.upload_dir(child, prefix + '/' + child.name)
        print(f'pushed {child} -> {prefix}/{child.name}')
" || log "sync failed (non-fatal)"
    exit "${rc}"
}
trap sync_runs_out EXIT INT TERM

# ---------------------------------------------------------------------------
# Phase 0.5 — optional self-improvement mode (three-flag toggle)
# ---------------------------------------------------------------------------
if [ "${MODE}" = "self_improve" ]; then
    if [ "${CLAUDE_CODE_ENABLED:-0}" != "1" ]; then
        log "MODE=self_improve but CLAUDE_CODE_ENABLED!=1 — refusing to start"
        exit 2
    fi
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        log "MODE=self_improve but ANTHROPIC_API_KEY unset — refusing to start"
        exit 2
    fi
    export ANTHROPIC_API_KEY
    log "self_improve: launching watchdog + claude code"
    # shellcheck disable=SC1091
    source ./scripts/iteration_watchdog.sh
    : "${MAX_ITERATIONS:=10}"
    : "${MAX_WALLCLOCK_HOURS:=12}"
    export MAX_ITERATIONS MAX_WALLCLOCK_HOURS
    spawn_watchdog
    PROGRAM="${PROGRAM_MD:-./program.md}"
    log "program: ${PROGRAM}"
    exec claude --dangerously-skip-permissions \
        -p "$(cat "${PROGRAM}")" --verbose 2>&1 | tee /app/claude.log
fi

# ---------------------------------------------------------------------------
# Deterministic training modes
# ---------------------------------------------------------------------------
RUN_DIR="${RUNS_ROOT}/${MODE}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
log "RUN_DIR=${RUN_DIR}"

case "${MODE}" in
    train)
        ARGS=(
            --cartridge "${CARTRIDGE}"
            --steps "${STEPS}"
            --max-episode-steps "${EPISODE_STEPS}"
            --save-every "${SAVE_EVERY}"
            --run-dir "${RUN_DIR}"
            --headless
        )
        [ -n "${ROM_LOCAL}" ] && ARGS+=(--rom "${ROM_LOCAL}")
        [ -n "${INIT_STATE_LOCAL}" ] && ARGS+=(--init-state "${INIT_STATE_LOCAL}")
        [ -n "${ENCODER_LOCAL:-}" ] && ARGS+=(--encoder "${ENCODER_LOCAL}")
        [ "${RESUME}" = "1" ] && ARGS+=(--resume)
        exec python -m deepEmulator.training.train "${ARGS[@]}"
        ;;
    collect_frames)
        : "${N_FRAMES:=10000}"
        ARGS=(
            --cartridges "${CARTRIDGE}"
            --rom "${ROM_LOCAL}"
            --frames "${N_FRAMES}"
            --out "${RUN_DIR}/frames"
        )
        exec python -m deepEmulator.training.collect_frames "${ARGS[@]}"
        ;;
    pretrain_dino)
        : "${CORPUS_GCS_URI:?CORPUS_GCS_URI required for pretrain_dino}"
        CORPUS_LOCAL="${DATA_ROOT}/corpus"
        log "staging corpus ${CORPUS_GCS_URI} -> ${CORPUS_LOCAL}"
        python -c "from deepEmulator.utils import gcs; gcs.download_dir('${CORPUS_GCS_URI}', '${CORPUS_LOCAL}')"
        exec python -m deepEmulator.training.pretrain_dino \
            --corpus "${CORPUS_LOCAL}" \
            --steps "${STEPS}" \
            --save-every "${SAVE_EVERY}" \
            --run-dir "${RUN_DIR}"
        ;;
    pretrain_vjepa)
        : "${CORPUS_GCS_URI:?CORPUS_GCS_URI required for pretrain_vjepa}"
        CORPUS_LOCAL="${DATA_ROOT}/corpus"
        log "staging corpus ${CORPUS_GCS_URI} -> ${CORPUS_LOCAL}"
        python -c "from deepEmulator.utils import gcs; gcs.download_dir('${CORPUS_GCS_URI}', '${CORPUS_LOCAL}')"
        exec python -m deepEmulator.training.pretrain_vjepa \
            --corpus "${CORPUS_LOCAL}" \
            --steps "${STEPS}" \
            --save-every "${SAVE_EVERY}" \
            --run-dir "${RUN_DIR}"
        ;;
    *)
        log "unknown MODE='${MODE}' — valid: train | collect_frames | pretrain_dino | pretrain_vjepa | self_improve"
        exit 2
        ;;
esac
