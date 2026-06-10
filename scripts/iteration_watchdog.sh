#!/usr/bin/env bash
# Watchdog companion to entrypoint.sh self_improve mode.
# Sourced, not exec'd.
#
# Caps:
#   MAX_WALLCLOCK_HOURS — HARD cap. The timer TERM-kills the claude PROCESS
#       GROUP (then KILL after a grace period). The old `kill -TERM 1` was
#       soft: bash defers traps until the foreground pipeline exits, so the
#       signal did nothing until claude finished on its own.
#   MAX_ITERATIONS — ADVISORY. iter_increment() lives in this (parent) shell;
#       the agent's own bash calls can't reach it and could rewrite the
#       counter file anyway. Treat it as an honor-system progress marker;
#       the wall clock is the enforcement.
#
# Required env (set by entrypoint.sh before sourcing):
#   MAX_WALLCLOCK_HOURS   integer >= 1
#   MAX_ITERATIONS        integer >= 1
set -u

RUNS_ROOT="${RUNS_ROOT:-/runs}"
ITER_FILE="${RUNS_ROOT}/.iter_counter"

spawn_watchdog() {
    # arg 1: PID of the setsid'd claude process == its process-group id
    local target_pgid="${1:?spawn_watchdog needs the claude pgid}"
    local secs=$(( MAX_WALLCLOCK_HOURS * 3600 ))
    local grace="${WATCHDOG_GRACE_SECS:-60}"
    (
        sleep "${secs}"
        echo "[watchdog] MAX_WALLCLOCK_HOURS=${MAX_WALLCLOCK_HOURS}h exceeded; TERM-killing pgid ${target_pgid}" >&2
        kill -TERM -- "-${target_pgid}" 2>/dev/null || true
        sleep "${grace}"
        if kill -0 -- "-${target_pgid}" 2>/dev/null; then
            echo "[watchdog] still alive after ${grace}s grace; KILL" >&2
            kill -KILL -- "-${target_pgid}" 2>/dev/null || true
        fi
    ) &
    disown
    mkdir -p "${RUNS_ROOT}"
    echo 0 > "${ITER_FILE}"
    echo "[watchdog] armed: wallclock=${MAX_WALLCLOCK_HOURS}h (hard) iter_max=${MAX_ITERATIONS} (advisory)" >&2
}

iter_increment() {
    local n
    n=$(( $(cat "${ITER_FILE}" 2>/dev/null || echo 0) + 1 ))
    echo "${n}" > "${ITER_FILE}"
    if [ "${n}" -gt "${MAX_ITERATIONS}" ]; then
        echo "[watchdog] iteration ${n} > MAX_ITERATIONS=${MAX_ITERATIONS}; halting" >&2
        exit 3
    fi
    echo "${n}"
}
