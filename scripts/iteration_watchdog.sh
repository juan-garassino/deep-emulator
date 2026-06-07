#!/usr/bin/env bash
# Watchdog companion to entrypoint.sh self_improve mode.
# Enforces hard caps on wall-clock and iteration count. Sourced, not exec'd.
#
# Required env (set by entrypoint.sh before sourcing):
#   MAX_WALLCLOCK_HOURS   integer >= 1
#   MAX_ITERATIONS        integer >= 1
#
# Side effects:
#   - background timer kills the container after MAX_WALLCLOCK_HOURS
#   - touch /runs/.iter_counter; iter_increment() bumps + checks the cap.
set -u

ITER_FILE="/runs/.iter_counter"

spawn_watchdog() {
    local secs=$(( MAX_WALLCLOCK_HOURS * 3600 ))
    (
        sleep "${secs}"
        echo "[watchdog] MAX_WALLCLOCK_HOURS=${MAX_WALLCLOCK_HOURS}h exceeded; killing PID 1" >&2
        kill -TERM 1 2>/dev/null || true
    ) &
    disown
    mkdir -p /runs
    echo 0 > "${ITER_FILE}"
    echo "[watchdog] armed: cap=${MAX_WALLCLOCK_HOURS}h iter_max=${MAX_ITERATIONS}" >&2
}

iter_increment() {
    local n
    n=$(( $(cat "${ITER_FILE}" 2>/dev/null || echo 0) + 1 ))
    echo "${n}" > "${ITER_FILE}"
    if [ "${n}" -gt "${MAX_ITERATIONS}" ]; then
        echo "[watchdog] iteration ${n} > MAX_ITERATIONS=${MAX_ITERATIONS}; halting" >&2
        kill -TERM 1 2>/dev/null || true
        exit 3
    fi
    echo "${n}"
}
