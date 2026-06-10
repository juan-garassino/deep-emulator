#!/usr/bin/env bash
# Lifecycle smoke — the mandatory pre-pod gate.
#
# Runs entrypoint.sh ON THE HOST against a file:// bucket, SIGTERMs it
# mid-run, and asserts the artifacts landed. Catches the failure classes
# that lose pod money: exec-destroys-trap, broken resume markers, missing
# python alias, sync-on-signal.
#
#   make smoke_lifecycle            # this script
#   make runpod_smoke_lifecycle     # same flow against the docker image
#
# Requires a GB ROM (defaults to roms/test_rom.gb) and a local editable
# install (`make install_dev`).
set -euo pipefail

ROM="${ROM:-roms/test_rom.gb}"
WORK="$(mktemp -d /tmp/deepemu-lifecycle.XXXXXX)"
BUCKET_DIR="${WORK}/bucket"
RUNS="${WORK}/runs"
DATA="${WORK}/data"
PREFIX="deepemulator/lifecycle-smoke"
mkdir -p "${BUCKET_DIR}/${PREFIX}" "${RUNS}" "${DATA}"

log() { printf '[lifecycle-smoke] %s\n' "$*" >&2; }
fail() { log "FAIL: $*"; exit 1; }

[ -f "${ROM}" ] || fail "ROM ${ROM} not found (set ROM=...)"

# seed the fake bucket with the ROM so the staging path is exercised too
cp "${ROM}" "${BUCKET_DIR}/${PREFIX}/$(basename "${ROM}")"

log "workdir=${WORK}"
env -i PATH="${PATH}" HOME="${HOME}" \
    GCS_BUCKET="file://${BUCKET_DIR}" \
    GCS_PREFIX="${PREFIX}" \
    MODE=train \
    CARTRIDGE="GENERIC GB" \
    ROM_GCS_URI="file://${BUCKET_DIR}/${PREFIX}/$(basename "${ROM}")" \
    STEPS=200000 \
    SAVE_EVERY=500 \
    EPISODE_STEPS=400 \
    SYNC_EVERY_SECS=10 \
    RUNS_ROOT="${RUNS}" \
    DATA_ROOT="${DATA}" \
    RUN_ID=smoke-run \
    bash ./entrypoint.sh &
EP_PID=$!

# give it time to boot, save at least one bundle, and run a periodic sync
log "entrypoint pid=${EP_PID}; letting it train for 45s..."
sleep 45

if ! kill -0 "${EP_PID}" 2>/dev/null; then
    wait "${EP_PID}" || true
    fail "entrypoint died before the kill test (check output above)"
fi

log "sending SIGTERM (simulates RunPod pod stop)"
kill -TERM "${EP_PID}"
rc=0
wait "${EP_PID}" || rc=$?
log "entrypoint exited rc=${rc}"

# --- assertions -------------------------------------------------------------
SYNCED="${BUCKET_DIR}/${PREFIX}/train/smoke-run"
[ -f "${SYNCED}/model.pt" ] || fail "model.pt did not reach the bucket (${SYNCED})"
[ -f "${SYNCED}/metadata.json" ] || fail "metadata.json did not reach the bucket"

MARKER="${BUCKET_DIR}/${PREFIX}/train/latest.txt"
[ -f "${MARKER}" ] || fail "train/latest.txt marker missing"
CONTENT="$(tr -d '[:space:]' < "${MARKER}")"
[ "${CONTENT}" = "smoke-run" ] || fail "marker must hold the relative run name, got '${CONTENT}'"

log "PASS — bundle + relative marker landed after SIGTERM"
log "(workdir kept for inspection: ${WORK})"
