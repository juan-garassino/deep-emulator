#!/usr/bin/env bash
# deepEmulator — RunPod entrypoint.
#
# Ports /Users/juan-garassino/Code/005-products/020-autoresearch/entrypoint.sh.
# Differences:
#   - Three-flag toggle for the optional self-improve mode.
#   - First-run-aware resume (missing latest.txt is NOT an error).
#   - Trainee runs as a CHILD (never exec — exec destroys bash and the
#     EXIT/TERM traps, which is how artifacts get lost).
#   - Periodic background sync to GCS — the only protection against
#     SIGKILL/OOM, which no trap survives.
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
NUM_ENVS="${NUM_ENVS:-1}"
SYNC_EVERY_SECS="${SYNC_EVERY_SECS:-300}"

RUNS_ROOT="${RUNS_ROOT:-/runs}"
DATA_ROOT="${DATA_ROOT:-/data}"
SECRETS_DIR="${SECRETS_DIR:-/secrets}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "${RUNS_ROOT}" "${DATA_ROOT}"

# Strip trailing slashes; the URI module reassembles cleanly.
GCS_BUCKET="${GCS_BUCKET%/}"
GCS_PREFIX="${GCS_PREFIX%/}"
PREFIX_URI="${GCS_BUCKET}/${GCS_PREFIX}"

log "MODE=${MODE}  CARTRIDGE='${CARTRIDGE}'  PREFIX_URI=${PREFIX_URI}  RUN_ID=${RUN_ID}"

# ---------------------------------------------------------------------------
# GCP credentials — RunPod injects secrets as ENV VARS, not file mounts.
# Template contract: GCP_SA_JSON={{ RUNPOD_SECRET_gcp_sa_deepemu }}
# Skipped for file:// buckets (local smokes need no auth).
# ---------------------------------------------------------------------------
if [ "${GCS_BUCKET#gs://}" != "${GCS_BUCKET}" ]; then
    if [ -n "${GCP_SA_JSON:-}" ]; then
        mkdir -p "${SECRETS_DIR}"
        umask 077
        printf '%s' "${GCP_SA_JSON}" > "${SECRETS_DIR}/gcp-sa.json"
        umask 022
        export GOOGLE_APPLICATION_CREDENTIALS="${SECRETS_DIR}/gcp-sa.json"
        log "GCP credentials materialized -> ${GOOGLE_APPLICATION_CREDENTIALS}"
    elif [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
        log "WARNING: gs:// bucket but neither GCP_SA_JSON nor GOOGLE_APPLICATION_CREDENTIALS set — GCS calls will fail"
    fi
fi

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
# Mode-specific preflight — fail in seconds, not after staging gigabytes
# ---------------------------------------------------------------------------
case "${MODE}" in
    collect_frames)
        : "${ROM_GCS_URI:?ROM_GCS_URI required for collect_frames}"
        ;;
    pretrain_vjepa)
        if ! python -c "from deepEmulator.training.pretrain_vjepa import IMPLEMENTED; raise SystemExit(0 if IMPLEMENTED else 2)"; then
            log "MODE=pretrain_vjepa but the V-JEPA module is Phase 1 (not implemented) — refusing before corpus staging"
            exit 2
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# Stage ROM + init state + encoder from GCS (when URIs are provided)
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
if [ -n "${ENCODER_GCS_URI:-}" ]; then
    ENCODER_LOCAL="${DATA_ROOT}/encoder"
    log "staging encoder bundle ${ENCODER_GCS_URI} -> ${ENCODER_LOCAL}"
    python -c "from deepEmulator.utils import gcs; gcs.download_dir('${ENCODER_GCS_URI}', '${ENCODER_LOCAL}')"
fi

# ---------------------------------------------------------------------------
# Resume — marker at ${PREFIX}/<mode>/latest.txt holds the run NAME.
# Download the bundle so train.main can actually continue from it.
# Missing marker is NOT an error (fresh run).
# ---------------------------------------------------------------------------
RESUME_DIR=""
if [ "${RESUME}" = "1" ] && [ "${MODE}" = "train" ]; then
    RESUME_NAME=$(python -c "from deepEmulator.utils import gcs; print(gcs.latest_run_name('${PREFIX_URI}/train') or '')")
    if [ -z "${RESUME_NAME}" ]; then
        log "no prior run found at ${PREFIX_URI}/train/latest.txt — starting fresh"
    else
        RESUME_DIR="${RUNS_ROOT}/train/${RESUME_NAME}"
        log "resuming '${RESUME_NAME}': downloading bundle -> ${RESUME_DIR}"
        python -c "from deepEmulator.utils import gcs; gcs.download_dir('${PREFIX_URI}/train/${RESUME_NAME}', '${RESUME_DIR}')"
    fi
fi

# ---------------------------------------------------------------------------
# Artifact lifeline:
#   - periodic background sync (survives nothing but needs no signal — the
#     only protection against SIGKILL/OOM)
#   - EXIT trap final sync (covers normal exit + SIGTERM/SIGINT)
# NEVER `exec` the trainee — exec replaces bash and destroys both.
# ---------------------------------------------------------------------------
SYNC_PID=""
CHILD_PID=""

run_sync() {
    python -m deepEmulator.utils.sync_runs "${RUNS_ROOT}" "${PREFIX_URI}"
}

start_periodic_sync() {
    (
        while sleep "${SYNC_EVERY_SECS}"; do
            run_sync || log "periodic sync failed — retrying next interval"
        done
    ) &
    SYNC_PID=$!
    log "periodic sync every ${SYNC_EVERY_SECS}s (pid ${SYNC_PID})"
}

sync_runs_out() {
    local rc=$?
    [ -n "${SYNC_PID}" ] && kill "${SYNC_PID}" 2>/dev/null || true
    log "sync_runs_out: exit code ${rc}; final push ${RUNS_ROOT}/ -> ${PREFIX_URI}/"
    if ! run_sync; then
        log "FINAL SYNC FAILED — artifacts under ${RUNS_ROOT} did NOT reach ${PREFIX_URI}"
        # distinct exit code: the run may have succeeded but its artifacts are stranded
        [ "${rc}" = "0" ] && rc=4
    fi
    exit "${rc}"
}

forward_term() {
    log "caught TERM/INT — forwarding to child ${CHILD_PID}"
    [ -n "${CHILD_PID}" ] && kill -TERM "${CHILD_PID}" 2>/dev/null || true
}

trap sync_runs_out EXIT
trap forward_term TERM INT

wait_child() {
    # wait that survives trap interruptions: a trapped signal makes `wait`
    # return 128+N immediately, BEFORE the child has actually exited — keep
    # waiting until it's truly gone, or the final sync races its last writes
    local rc=0
    while true; do
        if wait "${CHILD_PID}"; then
            rc=0
            break
        else
            rc=$?
            kill -0 "${CHILD_PID}" 2>/dev/null || break
        fi
    done
    return "${rc}"
}

run_child() {
    # run the trainee as a foreground child and propagate its exit code
    "$@" &
    CHILD_PID=$!
    wait_child
}

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
    start_periodic_sync
    # shellcheck disable=SC1091
    source ./scripts/iteration_watchdog.sh
    : "${MAX_ITERATIONS:=10}"
    : "${MAX_WALLCLOCK_HOURS:=12}"
    export MAX_ITERATIONS MAX_WALLCLOCK_HOURS
    PROGRAM="${PROGRAM_MD:-./program.md}"
    log "program: ${PROGRAM}"
    # setsid gives claude its own process group so the watchdog can hard-kill
    # the whole tree (kill -TERM 1 was deferred by bash until claude exited)
    setsid claude --dangerously-skip-permissions \
        -p "$(cat "${PROGRAM}")" --verbose > "${RUNS_ROOT}/claude.log" 2>&1 &
    CHILD_PID=$!
    spawn_watchdog "${CHILD_PID}"
    rc=0
    wait_child || rc=$?
    log "self_improve: claude exited rc=${rc}"
    exit "${rc}"
fi

# ---------------------------------------------------------------------------
# Deterministic training modes
# ---------------------------------------------------------------------------
if [ "${MODE}" = "train" ] && [ -n "${RESUME_DIR}" ]; then
    RUN_DIR="${RESUME_DIR}"
else
    RUN_DIR="${RUNS_ROOT}/${MODE}/${RUN_ID}"
fi
mkdir -p "${RUN_DIR}"
log "RUN_DIR=${RUN_DIR}"

start_periodic_sync

case "${MODE}" in
    train)
        ARGS=(
            --cartridge "${CARTRIDGE}"
            --steps "${STEPS}"
            --max-episode-steps "${EPISODE_STEPS}"
            --save-every "${SAVE_EVERY}"
            --num-envs "${NUM_ENVS}"
            --run-dir "${RUN_DIR}"
            --headless
        )
        [ -n "${ROM_LOCAL}" ] && ARGS+=(--rom "${ROM_LOCAL}")
        [ -n "${INIT_STATE_LOCAL}" ] && ARGS+=(--init-state "${INIT_STATE_LOCAL}")
        [ -n "${ENCODER_LOCAL:-}" ] && ARGS+=(--encoder "${ENCODER_LOCAL}")
        [ "${RESUME}" = "1" ] && ARGS+=(--resume)
        run_child python -m deepEmulator.training.train "${ARGS[@]}"
        ;;
    collect_frames)
        : "${N_FRAMES:=10000}"
        ARGS=(
            --cartridges "${CARTRIDGE}"
            --rom "${ROM_LOCAL}"
            --frames "${N_FRAMES}"
            --out "${RUN_DIR}/frames"
        )
        [ -n "${INIT_STATE_LOCAL}" ] && ARGS+=(--init-state "${INIT_STATE_LOCAL}")
        run_child python -m deepEmulator.training.collect_frames "${ARGS[@]}"
        ;;
    pretrain_dino)
        : "${CORPUS_GCS_URI:?CORPUS_GCS_URI required for pretrain_dino}"
        CORPUS_LOCAL="${DATA_ROOT}/corpus"
        log "staging corpus ${CORPUS_GCS_URI} -> ${CORPUS_LOCAL}"
        python -c "from deepEmulator.utils import gcs; gcs.download_dir('${CORPUS_GCS_URI}', '${CORPUS_LOCAL}')"
        run_child python -m deepEmulator.training.pretrain_dino \
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
        run_child python -m deepEmulator.training.pretrain_vjepa \
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
