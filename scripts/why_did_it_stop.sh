#!/usr/bin/env bash
# Work out what ended the last run: the process, or the whole container.
#
# The distinction matters. A process that crashed can be guarded against with a
# supervisor. A container that was culled cannot -- nothing inside it survives,
# and the fix is to stop the culling rather than to restart harder.
set -u
echo "container uptime      : $(uptime -p 2>/dev/null || cat /proc/uptime)"
echo "  If this is far shorter than the gap since your run stopped, the whole"
echo "  server was restarted and the culler is the cause."
echo
echo "runner alive now      : $(pgrep -af 'spatial_eval.cli run' || echo 'no')"
echo
for f in ~/run*.log ~/supervisor.log; do
    [ -f "$f" ] || continue
    echo "--- ${f} : last line, and when it was written ---"
    tail -1 "$f"
    echo "  modified: $(date -r "$f" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || stat -c %y "$f")"
done
echo
echo "kernel OOM kills      :"
dmesg 2>/dev/null | grep -i "killed process" | tail -3 || echo "  (dmesg unavailable in this container)"
