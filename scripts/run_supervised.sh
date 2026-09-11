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

# Inside a container, `free` reports the host: this node shows 2 TB while the
# cgroup that actually kills us may allow a few dozen GB. Report the limit we
# are held to, and the kernel's own count of how often it has enforced it.
mem_state() {
    if [ -r /sys/fs/cgroup/memory.max ]; then
        echo "    cgroup limit=$(cat /sys/fs/cgroup/memory.max) \
current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null) \
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)"
        sed 's/^/    events /' /sys/fs/cgroup/memory.events 2>/dev/null
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        echo "    cgroup limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) \
usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null) \
failcnt=$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)"
    else
        echo "    no cgroup memory accounting readable"
    fi
    # When the container itself is replaced, pid 1 is younger than the run.
    echo "    container pid 1 started $(ps -o lstart= -p 1 2>/dev/null | tr -s ' ')"
}

MAX_RESTARTS=${MAX_RESTARTS:-50}
PAUSE=${PAUSE:-30}

# Cheapest strategies first. The default order walks relations outer and
# strategies alphabetically, which puts two five-hour GoT cells ahead of every
# remaining single-call cell -- and while the server is being culled without
# warning, a finished cell is banked for good whereas a five-hour cell stopped
# at eighty percent is worth nothing until it completes. This order buys a
# comparison across all three relations in a couple of hours, and leaves the
# expensive arms for last. Override with PASSES="..." to change it.
PASSES=${PASSES:-"zero_shot cot few_shot tot got"}
attempt=0

while :; do
    attempt=$((attempt + 1))
    echo "=== attempt ${attempt} at $(date '+%Y-%m-%d %H:%M:%S') ==="
    mem_state
    status=0
    for strategy in ${PASSES}; do
        echo "--- pass: ${strategy} at $(date '+%H:%M:%S') ---"
        python3 -m spatial_eval.cli run --all -s "${strategy}"
        status=$?
        # Stop at the first failure so the retry restarts from a known state.
        # Re-running the passes that already succeeded costs almost nothing:
        # resume never recomputes a finished row.
        [ "${status}" -ne 0 ] && break
    done
    # A process killed outright leaves no traceback, so record the state that
    # usually explains it: 137 is SIGKILL, which on this host means the memory
    # killer far more often than anything else.
    if [ "${status}" -ne 0 ]; then
        echo "--- exit ${status} at $(date '+%Y-%m-%d %H:%M:%S') ---"
        [ "${status}" -eq 137 ] && echo "    (137 = SIGKILL, typically out of memory)"
        mem_state
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
