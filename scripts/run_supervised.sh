#!/usr/bin/env bash
# Keep the evaluation running across anything that stops it.
#
# setsid and nohup detach a process from the terminal, which covers a dropped
# connection or a closed shell. Neither survives the container itself being
# stopped -- and on JupyterHub an idle culler does exactly that, which is what
# ended an earlier run mid-cell with no traceback in the log.
#
# So this does two things. It restarts the pipeline whenever it exits without
# having finished, which covers crashes and out-of-memory kills. And because it
# is safe to launch at any time -- the run lock refuses a second copy, and
# resume never recomputes finished rows -- recovering after a cull is a matter
# of running this one line again.
#
#   setsid nohup bash scripts/run_supervised.sh < /dev/null &
#
# Do not redirect the output yourself. The script writes to a timestamped file
# under logs/, because redirecting to a fixed name truncates it on every launch
# -- which destroyed the evidence of why the previous run died, at the one
# moment it was needed.
#
set -u
cd "$(dirname "$0")/.."

# A launch must never overwrite the record of the launch before it.
mkdir -p logs
SUP_LOG="logs/supervisor-$(date '+%Y%m%d-%H%M%S').log"
exec >>"${SUP_LOG}" 2>&1
echo "supervisor log: ${SUP_LOG}" >&2

MAX_RESTARTS=${MAX_RESTARTS:-50}
PAUSE=${PAUSE:-30}
attempt=0

while :; do
    attempt=$((attempt + 1))
    echo "=== attempt ${attempt} at $(date '+%Y-%m-%d %H:%M:%S') ==="
    free -g 2>/dev/null | head -2 || true
    python3 -m spatial_eval.cli run --all
    status=$?
    # A process killed outright leaves no traceback, so record the state that
    # usually explains it: 137 is SIGKILL, which on this host means the memory
    # killer far more often than anything else.
    if [ "${status}" -ne 0 ]; then
        echo "--- exit ${status} at $(date '+%Y-%m-%d %H:%M:%S') ---"
        [ "${status}" -eq 137 ] && echo "    (137 = SIGKILL, typically out of memory)"
        free -g 2>/dev/null | head -2 || true
        dmesg 2>/dev/null | tail -5 || echo "    dmesg unavailable in this container"
    fi

    if [ "${status}" -eq 0 ]; then
        echo "=== finished cleanly at $(date '+%Y-%m-%d %H:%M:%S') ==="
        break
    fi
    if [ "${status}" -eq 2 ]; then
        # the lock refused us: another copy is already working, leave it alone
        echo "=== another run holds the lock; nothing to do ==="
        break
    fi
    if [ "${attempt}" -ge "${MAX_RESTARTS}" ]; then
        echo "=== giving up after ${attempt} attempts (last exit ${status}) ==="
        break
    fi
    echo "=== exited ${status}; restarting in ${PAUSE}s ==="
    sleep "${PAUSE}"
done
