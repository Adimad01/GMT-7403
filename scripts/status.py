"""One command that answers both questions: is it alive, and how far has it got.

A run stopped once and went unnoticed for thirty-five hours, because checking
took three separate commands and nobody ran all three. Neither fact is
sufficient alone: a process can survive while wedged, and a log can still look
fresh minutes after the process died. So this reports both, and says plainly
which of the two it believes.

    python3 scripts/status.py
"""
from __future__ import annotations

import argparse
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
ADAPTERS = REPO / "adapters"
LOCK = RESULTS / ".run.lock"

# ToT and GoT log every ten rows and spend about a minute on each, so a quiet
# quarter of an hour is ordinary. Past that, something is wrong.
QUIET_OK_MIN = 20

# The grid the runner walks, in its order: relations outer, strategies inner,
# sorted as available() sorts them.
RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("cot", "few_shot", "got", "tot", "zero_shot")
# The arms the design crosses. A cell of any of them that has never been
# started is still work the plan expects, which is not the same as work in
# progress -- and saying TERMINÉ for the second made it sound like the first.
ARMS = ("", "_lora", "_kg", "_lora_kg")
# few-shot on a fine-tuned model draws its demos from the training pool, so
# those cells are out of scope rather than outstanding.
OUT_OF_SCOPE = {("_lora", "few_shot"), ("_lora_kg", "few_shot")}
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


def _etime_seconds(text: str) -> float:
    """Parse ps's elapsed format: [[dd-]hh:]mm:ss."""
    days, _, rest = text.rpartition("-")
    parts = [float(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return (float(days or 0) * 86400) + h * 3600 + m * 60 + sec


def process_age(pid: int) -> float | None:
    """Seconds since this process started, or None if it cannot be read."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        started_ticks = int(fields[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - started_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        pass
    # BSD and macOS have no /proc, and their ps has no etimes either -- only
    # etime, as [[dd-]hh:]mm:ss.
    for spec, parse in (("etimes=", float), ("etime=", _etime_seconds)):
        try:
            out = subprocess.run(["ps", "-o", spec, "-p", str(pid)],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if out:
                return parse(out)
        except Exception:
            continue
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
            except PermissionError:
                # Not allowed to signal it -- which is proof it exists.
                return pid, "verrou (pid seul)"
            except (ProcessLookupError, OSError):
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


def _proc_started_at(pid: int) -> float | None:
    """Wall-clock time this pid began, in seconds since the epoch.

    Needed to tell a cell the current run is writing from one a previous run
    left half-finished. File age alone cannot decide it: a cell killed a minute
    before the relaunch is fresher than anything the new run has had time to
    touch, and was duly announced as "cellule en cours" while sitting
    untouched -- on a run that had been restarted precisely because the
    previous cell was going nowhere.
    """
    ticks = _start_ticks(pid)
    if ticks is None:
        return None
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                return int(line.split()[1]) + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        pass
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
        # "seed1_lora", "seed1_lora-cardinal" -> the arm's variant. The grid
        # showed only seed1, so every fine-tuned result was invisible here
        # even while the audit was reporting it.
        variant = seed[len("seed1"):] if seed.startswith("seed1") else ""

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
            "id": rid, "variant": variant,
            "relation": relation, "strategy": strategy,
            "seen": len(seen), "total": total, "ok": ok,
            "acc": corr / ok if ok else 0.0,
            "done": bool(total) and len(seen) >= total,
            "touched": preds.stat().st_mtime,
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


def remaining(cells: list[dict]) -> bool:
    """The grid and what is left. True when work remains.

    The grid holds the base model and each family's own adapter. A directory
    written by another family's adapter is not part of this comparison and is
    not shown.
    """
    by_id = {c["id"]: c for c in cells}
    rate = per_row_seconds()
    single = [v for k, v in rate.items() if k not in MULTI_CALL]
    multi = [v for k, v in rate.items() if k in MULTI_CALL]

    def rate_for(strat: str) -> float | None:
        if strat in rate:
            return rate[strat]
        pool = multi if strat in MULTI_CALL else single
        return statistics.median(pool) if pool else None

    # Every level counted, level 6 included -- so these differ from
    # audit_results.py, which excludes it. This is a progress view; the audit
    # is where a number is meant to be read.
    print("\n  grille   (tous niveaux ; l'audit exclut le niveau 6)")
    head = "".join(f"{s:>11}" for s in STRATEGIES)
    print(f"    {'':<18}{head}")
    # Every arm of this comparison: the family's own adapter, the knowledge
    # store, and the two together. A variant naming another family's adapter
    # carries a hyphen -- that is the transfer experiment, not this grid.
    # Matching "_lora" alone hid the knowledge-store cells as soon as they
    # existed, which is how a finished run looked like one that never ran.
    variants = [v for v in sorted({c["variant"] for c in cells if c["variant"]})
                if "-" not in v]
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
        print(f"    {rel:<18}" + "".join(marks))
        # Fine-tuned arms sit under their family, so base and adapted scores
        # for the same cell read down a single column.
        for var in variants:
            row = []
            for strat in STRATEGIES:
                c = by_id.get(f"{rel}__{strat}__seed1{var}")
                if c and c["done"]:
                    row.append(f"{c['acc'] * 100:>10.1f}%")
                elif c and c["total"]:
                    row.append(f"{c['seen'] / c['total'] * 100:>10.0f}%")
                else:
                    row.append(f"{'·':>11}")
            if any("·" not in x for x in row):
                print(f"      {var.lstrip('_'):<16}" + "".join(row))

    if not todo:
        print("\n  les 15 cellules de référence sont calculées "
              "(voir coverage.py pour le reste du plan).")
        return False

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
    return True


def adapters() -> None:
    """Fine-tuning state, which the results grid cannot show.

    A half-trained adapter produces results that look like any other, so the
    training state belongs beside the grid rather than in a separate file
    nobody thinks to open.
    """
    states = sorted(ADAPTERS.glob("*/trainer_state.json")) if ADAPTERS.is_dir() else []
    if not states:
        return
    print("\n  adaptateurs")
    for sp in states:
        try:
            st = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            print(f"    {sp.parent.name:<16}état illisible")
            continue
        losses = st.get("losses") or []
        cfg = st.get("config") or {}
        snaps = sorted(q.name for q in sp.parent.glob("epoch*"))
        loss = f"perte {losses[-1]:.4f}" if losses else "perte —"
        state = "terminé" if st.get("finished") else "INACHEVÉ"
        print(f"    {sp.parent.name:<14}{st.get('n_train_rows', '?'):>5} lignes   "
              f"{st.get('epoch', 0)}/{cfg.get('epochs', '?')} époques   "
              f"{loss:<14}{state:<10}"
              f"{round(st.get('elapsed_seconds', 0) / 60):>4} min"
              + (f"   [{len(snaps)} instantanés]" if snaps else ""))


def pending(cells: list[dict]) -> bool:
    """Whether anything is left to compute."""
    # A cell that was started and never finished is outstanding work whatever
    # arm it belongs to. Checking only the fifteen reference cells declared a
    # run finished while a fine-tuned cell sat at 160 of 288 with its process
    # dead -- the one state where saying TERMINÉ costs the most.
    if any(c["seen"] and not c["done"] for c in cells):
        return True
    by_id = {c["id"]: c for c in cells}
    return any(not (c := by_id.get(f"{rel}__{strat}__seed1")) or not c["done"]
               for rel in RELATIONS for strat in STRATEGIES)


def supervisor_log() -> None:
    """How the last supervised run ended, from the supervisor's own journal.

    run.log records what the runner was computing. It says nothing about the
    supervisor's decisions: which arm it had reached, whether a pass exited
    non-zero, whether it gave up. Without that, a run that walked the wrong arm
    and exited cleanly is indistinguishable from one the culler killed -- both
    leave a stale last line and no process. Reading the two journals together
    took two more commands, so in practice nobody read the second one.
    """
    logs = sorted((REPO / "logs").glob("supervisor-*.log"),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        print("\n  journal du superviseur : aucun — run_supervised.sh "
              "n'a jamais été lancé depuis ce conteneur")
        return
    newest = logs[-1]
    lines = [l.rstrip() for l in
             newest.read_text(encoding="utf-8", errors="replace").splitlines()
             if l.strip()]
    age = (time.time() - newest.stat().st_mtime) / 60
    print(f"\n  journal du superviseur ({age:.0f} min)   {newest.name}")

    passes = [l for l in lines if l.startswith("--- pass:")]
    if passes:
        print(f"    {len(passes)} passe(s) parcourue(s), "
              f"dernière : {passes[-1][len('--- pass:'):].strip(' -')}")
    else:
        print("    aucune passe entamée")

    # The supervisor brackets its own decisions in ===, which is the only
    # place an exit code or a give-up is recorded.
    verdicts = [l.strip("= ").strip() for l in lines if l.startswith("===")]
    if verdicts:
        for l in verdicts[-2:]:
            print(f"    {l}")
    else:
        print("    aucun verdict écrit — tué avant de pouvoir en écrire un "
              "(conteneur détruit)")

    arms = [l for l in lines if l.startswith("--- pass:")]
    seen = []
    for l in arms:
        a = l[len("--- pass:"):].strip().split("/")[0].strip()
        if a not in seen:
            seen.append(a)
    if seen:
        print(f"    bras visités : {', '.join(seen)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    pid, source = runner_pid()
    age_min = (time.time() - RUN_LOG.stat().st_mtime) / 60 if RUN_LOG.exists() else None

    age_s = process_age(pid) if pid else None

    if pid and age_min is not None and age_min <= QUIET_OK_MIN:
        verdict, detail = "EN COURS", f"pid {pid}, journal écrit il y a {age_min:.0f} min"
    elif pid and age_s is not None and age_s < age_min * 60:
        # The log is older than the process, so its last line belongs to an
        # earlier run: this one has written nothing yet because it has not
        # been alive long enough to. Reading the log age alone called a
        # process forty seconds old stalled for forty-four minutes.
        verdict, detail = ("EN COURS",
                           f"pid {pid} démarré il y a {age_s / 60:.0f} min "
                           f"(le modèle met ~1 min à charger)")
    elif pid and age_min is not None:
        verdict, detail = "BLOQUÉ", (f"pid {pid} vivant, mais rien d'écrit depuis "
                                     f"{age_min:.0f} min")
    elif pid:
        verdict, detail = "EN COURS", f"pid {pid}, journal absent"
    elif pending(cell_progress()):
        verdict, detail = "ARRÊTÉ", ("aucun processus" if age_min is None else
                                     f"aucun processus, dernière écriture il y a "
                                     f"{age_min:.0f} min")
    else:
        # No process and nothing half-done. That is not the same as the plan
        # being finished, so the verdict says what it means and the count of
        # cells never started is printed beside it.
        planned = sum(1 for a in ARMS for rel in RELATIONS for strat in STRATEGIES
                      if (a, strat) not in OUT_OF_SCOPE
                      and not (RESULTS / rel / strat / f"seed1{a}" / "run.json").exists())
        verdict = "AU REPOS"
        detail = ("rien en cours, rien à reprendre" if not planned else
                  f"rien en cours ; {planned} cellule(s) du plan jamais lancée(s)")

    sup = supervisor_pid()

    bar = "─" * 68
    print(bar)
    print(f"  {verdict}    {detail}")
    print(f"  {'':<10}source : {source}")
    # Only worth saying while something is running: with nothing to restart,
    # a missing watchdog is not a problem to report.
    if verdict != "AU REPOS":
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
        # Only one cell is ever being written. The others are part-finished
        # from an earlier session and wait for their pass to come round --
        # listing them all as "in progress" reads as several at once.
        started = _proc_started_at(pid) if pid else None
        if started is None:
            active = max(running, key=lambda c: c["touched"]) if pid else None
        else:
            # A cell last written before this process existed is not the cell
            # this process is writing, however recently it was touched.
            fresh = [c for c in running if c["touched"] >= started - 5]
            active = max(fresh, key=lambda c: c["touched"]) if fresh else None
        print()
        for c in sorted(running, key=lambda c: -c["touched"]):
            here = c is active
            print("  cellule en cours :" if here else "  reprise en attente :")
            if c["total"]:
                filled = round(width * c["seen"] / c["total"])
                print(f"    {c['id']:<32}[{'#' * filled}{'.' * (width - filled)}] "
                      f"{c['seen']}/{c['total']}  ({c['seen'] / c['total'] * 100:.0f} %)")
            else:
                print(f"    {c['id']:<32}{c['seen']} lignes")

    remaining(cells)

    adapters()

    if RUN_LOG.exists():
        tail = RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n  dernière ligne du journal ({age_min:.0f} min) :")
        print(f"    {tail[-1][:100] if tail else '(vide)'}")

    supervisor_log()

    if verdict == "EN COURS" and not sup:
        print("\n  Le calcul avance, mais rien ne le relancera s'il meurt.")
        print("  Démarrer le superviseur (il refusera de doubler le run en cours) :")
        print("    cd ~ && setsid nohup bash scripts/run_supervised.sh < /dev/null &")

    if verdict == "AU REPOS":
        # Nothing running and nothing half-done. Offering to resume would be
        # wrong -- there is nothing to resume -- and the supervisor only walks
        # the base grid, so it would not start the cells that are missing
        # either. Point at the inventory instead.
        if "jamais lancée" in detail:
            print("\n  Rien à reprendre. Pour voir les cellules qui restent :")
            print("    python3 scripts/coverage.py")
    elif verdict != "EN COURS":
        print("\n  Relancer — la reprise conserve tout ce qui est déjà calculé :")
        print("    cd ~ && git pull --rebase origin main && \\")
        print("      setsid nohup bash scripts/run_supervised.sh < /dev/null &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
