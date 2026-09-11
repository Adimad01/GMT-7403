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
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_LOG = REPO / "logs" / "run.log"
RESULTS = REPO / "results"
DATA = REPO / "data"
LOCK = RESULTS / ".run.lock"

# ToT and GoT log every ten rows and spend about a minute on each, so a quiet
# quarter of an hour is ordinary. Past that, something is wrong.
QUIET_OK_MIN = 20

# The grid the runner walks, in its order: relations outer, strategies inner,
# sorted as available() sorts them.
RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("cot", "few_shot", "got", "tot", "zero_shot")
CELLS = len(RELATIONS) * len(STRATEGIES)

# ToT and GoT issue four model calls per row against one for the rest, so a
# single average over all strategies would badly misestimate whichever is
# left. Used only as a fallback when a strategy has no measured rows yet.
MULTI_CALL = {"tot", "got"}


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


def _processes(pattern: str) -> list[tuple[int, str]]:
    """Pids whose command line matches, each with that command line.

    pgrep -a prints the command alongside the pid on Linux but not on BSD or
    macOS, where the same flag yields bare pids -- so a filter reading the
    command from pgrep's own output matches nothing there, silently. The
    command is read from ps instead, which behaves the same everywhere.
    """
    try:
        out = subprocess.run(["pgrep", "-f", pattern],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    found = []
    for pid_s in out.split():
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if pid == os.getpid():
            continue
        try:
            cmd = subprocess.run(["ps", "-p", pid_s, "-o", "command="],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except Exception:
            continue
        if cmd:
            found.append((pid, cmd))
    return found


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
    for pid, cmd in _processes("spatial_eval"):
        if "python" in cmd and "spatial_eval.cli" in cmd and " run" in cmd:
            return pid, "table des processus"
    return None, "aucun"


def expected_rows(relation: str) -> int | None:
    """How many rows a cell of this relation is meant to cover.

    Taken from the pinned eval manifest rather than run.json, because
    run.json is only written once a cell finishes -- so the cell currently
    running, the one you most want to see, has no run.json at all.
    """
    man = DATA / relation / "eval_manifest.json"
    try:
        return len(json.loads(man.read_text(encoding="utf-8"))["rows"])
    except Exception:
        return None


def supervisor_pid() -> int | None:
    """The watchdog that restarts the runner after a crash.

    Worth checking separately. If it has died while the runner lives, every
    reading here still looks healthy -- right up to the next crash, which
    then goes unrestarted. That silent gap is the failure that cost a run
    thirty-five hours.
    """
    for pid, cmd in _processes("run_supervised"):
        if "run_supervised.sh" in cmd:
            return pid
    return None


def cell_progress() -> list[dict]:
    """Per-cell counts read from the results on disk, not from the log.

    The log says what was announced; these files say what was kept.
    """
    cells = []
    # Keyed on predictions.jsonl: it exists from the first row written,
    # whereas run.json appears only at the end.
    for preds in sorted(RESULTS.glob("*/*/seed*/predictions.jsonl")):
        d = preds.parent
        relation, strategy, seed = d.parent.parent.name, d.parent.name, d.name
        rid = f"{relation}__{strategy}__{seed}"

        total = None
        run_json = d / "run.json"
        if run_json.exists():
            try:
                meta = json.loads(run_json.read_text(encoding="utf-8"))
                total = meta.get("n_examples")
                rid = meta.get("run_id", rid)
            except Exception:
                pass
        if not total:
            total = expected_rows(relation)

        seen: set[int] = set()
        ok = corr = 0
        for line in preds.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue        # the final line can be half-written mid-run
            seen.add(r.get("row_index", -1))
            if r.get("status") == "ok":
                ok += 1
                corr += bool(r.get("correct"))

        # A cell is finished when every row has been attempted. Rows that
        # failed were still attempted, and demanding that they all succeed
        # would leave a finished cell looking permanently unfinished.
        cells.append({
            "id": rid, "seen": len(seen), "total": total, "ok": ok,
            "acc": corr / ok if ok else 0.0,
            "done": bool(total) and len(seen) >= total,
        })
    return cells


def per_row_seconds() -> dict[str, float]:
    """Median seconds per row for each strategy, measured from the rows.

    Taken from the rows themselves rather than a cell's total elapsed time,
    which counts only what a session ran and is skewed by every resume.
    """
    samples: dict[str, list[float]] = {}
    for preds in RESULTS.glob("*/*/seed*/predictions.jsonl"):
        strat = preds.parent.parent.name
        for line in preds.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
            if '"seconds"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get("seconds"), (int, float)):
                samples.setdefault(strat, []).append(r["seconds"])
    return {k: statistics.median(v) for k, v in samples.items() if v}


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def remaining(cells: list[dict]) -> None:
    """The grid, and what is left to compute."""
    by_id = {c["id"]: c for c in cells}
    rate = per_row_seconds()
    single = [v for k, v in rate.items() if k not in MULTI_CALL]
    multi = [v for k, v in rate.items() if k in MULTI_CALL]

    def rate_for(strat: str) -> float | None:
        if strat in rate:
            return rate[strat]
        pool = multi if strat in MULTI_CALL else single
        return statistics.median(pool) if pool else None

    print("\n  grille")
    head = "".join(f"{s:>11}" for s in STRATEGIES)
    print(f"    {'':<13}{head}")
    todo, guessed = [], False
    for rel in RELATIONS:
        marks = []
        for strat in STRATEGIES:
            c = by_id.get(f"{rel}__{strat}__seed1")
            total = c["total"] if c else expected_rows(rel)
            seen = c["seen"] if c else 0
            if c and c["done"]:
                marks.append(f"{c['acc'] * 100:>10.1f}%")
            elif seen:
                marks.append(f"{seen / total * 100:>10.0f}%" if total else f"{seen:>11}")
            else:
                marks.append(f"{'·':>11}")
            if not (c and c["done"]) and total:
                left = total - seen
                r = rate_for(strat)
                if r is None:
                    guessed = True
                todo.append((f"{rel}__{strat}", left, (left * r) if r else None,
                             strat not in rate))
        print(f"    {rel:<13}" + "".join(marks))

    if not todo:
        print("\n  tout est calculé.")
        return

    print(f"\n  reste {len(todo)} cellule(s)")
    total_s = 0.0
    any_unknown = False
    for name, left, secs, inferred in todo:
        if secs is None:
            any_unknown = True
            print(f"    {name:<30}{left:>5} lignes          —")
            continue
        total_s += secs
        note = "  (cadence déduite)" if inferred else ""
        print(f"    {name:<30}{left:>5} lignes   {human(secs):>8}{note}")

    if total_s:
        print(f"\n    total estimé   {human(total_s)}"
              + ("  au moins" if any_unknown else ""))
        end = time.localtime(time.time() + total_s)
        print(f"    fin prévue     {time.strftime('%a %d %b %H:%M', end)}")
        print("    (cadence mesurée sur les lignes déjà calculées ; "
              "une reprise ou un arrêt décale d'autant)")


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

    sup = supervisor_pid()

    bar = "─" * 68
    print(bar)
    print(f"  {verdict}    {detail}")
    print(f"  {'':<10}source : {source}")
    print(f"  {'':<10}superviseur : "
          + (f"pid {sup}, relance automatique active" if sup else
             "ABSENT — aucune relance automatique en cas de plantage"))
    print(bar)

    cells = cell_progress()
    done = [c for c in cells if c["done"]]
    running = [c for c in cells if not c["done"] and c["seen"]]

    # The grid below carries each finished cell's accuracy, so listing them
    # again here would only repeat it. Failures do not appear there, and are
    # the one thing worth interrupting for.
    broken = [c for c in done if c["ok"] < c["total"]]
    if broken:
        print("\n  lignes en échec :")
        for c in broken:
            print(f"    {c['id']:<32}{c['total'] - c['ok']} sur {c['total']}")

    if running:
        width = 30
        print("\n  cellule en cours :")
        for c in running:
            if c["total"]:
                filled = round(width * c["seen"] / c["total"])
                print(f"    {c['id']:<32}[{'#' * filled}{'.' * (width - filled)}] "
                      f"{c['seen']}/{c['total']}  ({c['seen'] / c['total'] * 100:.0f} %)")
            else:
                print(f"    {c['id']:<32}{c['seen']} lignes")

    remaining(cells)

    if RUN_LOG.exists():
        tail = RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n  dernière ligne du journal ({age_min:.0f} min) :")
        print(f"    {tail[-1][:100] if tail else '(vide)'}")

    if verdict == "EN COURS" and not sup:
        print("\n  Le calcul avance, mais rien ne le relancera s'il meurt.")
        print("  Démarrer le superviseur (il refusera de doubler le run en cours) :")
        print("    cd ~ && setsid nohup bash scripts/run_supervised.sh < /dev/null &")

    if verdict != "EN COURS":
        print("\n  Relancer — la reprise conserve tout ce qui est déjà calculé :")
        print("    cd ~ && git pull --rebase origin main && \\")
        print("      setsid nohup bash scripts/run_supervised.sh < /dev/null &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
