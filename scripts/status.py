"""One command that answers both questions: is it alive, and how far has it got.

A run stopped once and went unnoticed for thirty-five hours, because checking
took three separate commands and nobody ran all three. Neither fact is
sufficient alone: a process can survive while wedged, and a log can still look
fresh minutes after the process died. So this reports both, and says plainly
which of the two it believes.

    python3 scripts/status.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_LOG = REPO / "logs" / "run.log"
RESULTS = REPO / "results"
LOCK = RESULTS / ".run.lock"

# ToT and GoT log every ten rows and spend about a minute on each, so a quiet
# quarter of an hour is ordinary. Past that, something is wrong.
QUIET_OK_MIN = 20

CELLS = 15          # three relations x five strategies, one seed


def _start_ticks(pid: int) -> int | None:
    """When this pid began, in clock ticks since boot.

    Pids are recycled. Without this the check would happily report a run as
    alive because some unrelated process inherited its number.
    """
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[19])          # field 22 overall, 20th after the name
    except Exception:
        return None


def runner_pid() -> tuple[int | None, str]:
    """The pid of the evaluation process, from the lock it holds."""
    if LOCK.exists():
        try:
            rec = json.loads(LOCK.read_text(encoding="utf-8"))
            pid, started = int(rec["pid"]), rec.get("started")
            if Path("/proc").is_dir():
                # Linux: the start time distinguishes the real process from
                # an unrelated one that inherited a recycled pid.
                if _start_ticks(pid) == started:
                    return pid, "verrou"
                return None, "verrou périmé"    # killed without releasing it
            # No /proc, so no start time to compare. Existence alone is a
            # weaker test, but reporting the lock as stale on that basis
            # would be simply wrong.
            try:
                os.kill(pid, 0)
                return pid, "verrou (pid seul)"
            except (ProcessLookupError, PermissionError, OSError):
                return None, "verrou périmé"
        except Exception:
            pass

    # No usable lock: fall back to the process table. Match only real
    # interpreters -- any shell whose command line merely mentions the module
    # would otherwise count as a running experiment.
    try:
        out = subprocess.run(["pgrep", "-af", "spatial_eval"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None, "indéterminé"
    for line in out.strip().splitlines():
        pid_s, _, cmd = line.partition(" ")
        if "python" in cmd and "spatial_eval.cli" in cmd and " run" in cmd:
            return int(pid_s), "table des processus"
    return None, "aucun"


def cell_progress() -> list[dict]:
    """Per-cell counts read from the results on disk, not from the log.

    The log says what was announced; these files say what was kept.
    """
    cells = []
    for rj in sorted(RESULTS.glob("*/*/seed*/run.json")) if RESULTS.exists() else []:
        try:
            meta = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        total = meta.get("n_examples") or 0
        seen: set[int] = set()
        ok = corr = 0
        preds = rj.parent / "predictions.jsonl"
        if preds.exists():
            for line in preds.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(r.get("row_index", -1))
                if r.get("status") == "ok":
                    ok += 1
                    corr += bool(r.get("correct"))
        # A cell is finished when every row has been attempted. Rows that
        # failed still count as attempted -- demanding that they all succeed
        # would leave a finished cell looking permanently unfinished.
        cells.append({
            "id": meta.get("run_id", rj.parent.name),
            "seen": len(seen), "total": total, "ok": ok,
            "acc": corr / ok if ok else 0.0,
            "done": bool(total) and len(seen) >= total,
        })
    return cells


def main() -> int:
    pid, source = runner_pid()
    age_min = (time.time() - RUN_LOG.stat().st_mtime) / 60 if RUN_LOG.exists() else None

    if pid and age_min is not None and age_min <= QUIET_OK_MIN:
        verdict, detail = "EN COURS", f"pid {pid}, journal écrit il y a {age_min:.0f} min"
    elif pid and age_min is not None:
        verdict, detail = "BLOQUÉ", (f"pid {pid} vivant, mais rien d'écrit depuis "
                                     f"{age_min:.0f} min")
    elif pid:
        verdict, detail = "EN COURS", f"pid {pid}, journal absent"
    else:
        verdict, detail = "ARRÊTÉ", ("aucun processus" if age_min is None else
                                     f"aucun processus, dernière écriture il y a "
                                     f"{age_min:.0f} min")

    bar = "─" * 68
    print(bar)
    print(f"  {verdict}    {detail}")
    print(f"  {'':<10}source : {source}")
    print(bar)

    cells = cell_progress()
    done = [c for c in cells if c["done"]]
    running = [c for c in cells if not c["done"] and c["seen"]]

    print(f"\n  cellules terminées : {len(done)}/{CELLS}")
    for c in done or []:
        missed = c["total"] - c["ok"]
        note = f"  ({missed} échecs)" if missed else ""
        print(f"    {c['id']:<32}{c['ok']:>5}/{c['total']:<5}"
              f"{c['acc'] * 100:>7.1f} %{note}")
    if not done:
        print("    (aucune)")

    if running:
        width = 30
        print("\n  cellule en cours :")
        for c in running:
            filled = round(width * c["seen"] / c["total"]) if c["total"] else 0
            print(f"    {c['id']:<32}[{'#' * filled}{'.' * (width - filled)}] "
                  f"{c['seen']}/{c['total']}  ({c['seen'] / c['total'] * 100:.0f} %)"
                  if c["total"] else f"    {c['id']:<32}{c['seen']} lignes")

    remaining = CELLS - len(done)
    if remaining:
        print(f"\n  reste {remaining} cellule(s) sur {CELLS}")

    if RUN_LOG.exists():
        tail = RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n  dernière ligne du journal ({age_min:.0f} min) :")
        print(f"    {tail[-1][:100] if tail else '(vide)'}")

    if verdict != "EN COURS":
        print("\n  Relancer — la reprise conserve tout ce qui est déjà calculé :")
        print("    cd ~ && git pull --rebase origin main && \\")
        print("      setsid nohup bash scripts/run_supervised.sh < /dev/null &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
