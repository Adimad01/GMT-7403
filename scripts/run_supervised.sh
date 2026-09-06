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
#   setsid nohup bash scripts/run_supervised.sh > supervisor.log 2>&1 < /dev/null &
#
set -u
cd "$(dirname "$0")/.."

MAX_RESTARTS=${MAX_RESTARTS:-50}
PAUSE=${PAUSE:-30}
attempt=0

while :; do
    attempt=$((attempt + 1))
    echo "=== attempt ${attempt} at $(date '+%Y-%m-%d %H:%M:%S') ==="
    python3 -m spatial_eval.cli run --all
    status=$?

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
