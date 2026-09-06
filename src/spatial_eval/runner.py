"""Run one (relation, strategy, seed) cell and write structured results.

Output layout, one directory per cell:

    results/<relation>/<strategy>/seed<N>/
        predictions.jsonl   one JSON object per evaluation row
        run.json            config, manifest hashes, counts, timing
        traces.jsonl        every prompt and completion  (--save-traces)

Resume semantics: a row already recorded with ``status="ok"`` is never
recomputed. Rows recorded with ``status="error"`` are retried. That is what
makes "rerun the failures without repeating the successes" a single command.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import LOGS_DIR, RunConfig
from .data import load_demos, load_examples, labels_for
from .model import Backend, build_backend
from .strategies import Context, get_strategy

log = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _load_done(path: Path) -> dict[int, dict]:
    """Existing predictions, keyed by row index. Malformed lines are dropped."""
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["row_index"]] = rec
            except (json.JSONDecodeError, KeyError):
                # A truncated final line is expected if a run was killed
                # mid-write. Skip it; the row will simply be recomputed.
                continue
    return done


def run_cell(cfg: RunConfig, backend: Backend | None = None,
             save_traces: bool = False) -> dict:
    """Execute one cell. Returns its summary dict."""
    out_dir = cfg.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    trace_path = out_dir / "traces.jsonl"

    examples, eval_hash = load_examples(cfg.relation, limit=cfg.limit)
    labels = labels_for(cfg.relation)

    strategy_cls = get_strategy(cfg.strategy)
    strategy = strategy_cls()

    demos = demo_hash = None
    if cfg.strategy == "few_shot":
        demos, demo_hash = load_demos(cfg.relation)

    # Resume matches rows by index, which is only meaningful while the data
    # behind those indices is unchanged. After the corpus is regenerated the
    # same index refers to a different item, so predictions kept from before
    # would be scored against text they never saw. run.json records the
    # manifest hash the results were produced under; a mismatch means they
    # describe a corpus that no longer exists.
    prior = out_dir / "run.json"
    if cfg.resume and prior.exists():
        try:
            was = json.loads(prior.read_text(encoding="utf-8"))
            old = was.get("eval_manifest_sha256")
        except Exception:
            old = None
        old_demo = was.get("fewshot_manifest_sha256") if 'was' in dir() else None
        if (cfg.strategy == "few_shot" and old_demo and demo_hash
                and old_demo != demo_hash):
            raise RuntimeError(
                f"{cfg.run_id}: results in {out_dir} used few-shot "
                f"demonstrations {old_demo[:12]}, but the demo map now hashes "
                f"to {demo_hash[:12]}. The prompts have changed, so resuming "
                f"would mix answers given under different demonstrations. "
                f"Delete this directory and run the cell again.")
        if old and old != eval_hash:
            raise RuntimeError(
                f"{cfg.run_id}: results in {out_dir} were produced against eval "
                f"manifest {old[:12]}, but the data now hashes to "
                f"{eval_hash[:12]}. Row indices no longer refer to the same "
                f"items, so resuming would mix answers to different questions. "
                f"Delete this directory (or all of results/) and start the cell "
                f"again.")

    done = _load_done(pred_path) if cfg.resume else {}
    todo = [e for e in examples
            if done.get(e.row_index, {}).get("status") != "ok"]
    n_retry = sum(1 for e in examples
                  if done.get(e.row_index, {}).get("status") == "error")

    log.info("%s | %d rows total, %d already done, %d to run (%d retries)",
             cfg.run_id, len(examples), len(examples) - len(todo), len(todo), n_retry)

    if backend is None:
        backend = build_backend(cfg.model)

    ctx = Context(relation=cfg.relation, labels=labels, seed=cfg.seed,
                  generate=backend.generate, demos=demos)

    started = time.time()
    n_ok = n_err = 0
    with pred_path.open("a", encoding="utf-8") as pf, \
            (trace_path.open("a", encoding="utf-8") if save_traces
             else _NullFile()) as tf:
        for i, ex in enumerate(todo, 1):
            t0 = time.time()
            try:
                res = strategy.run(ex, ctx)
                rec = {
                    "row_index": ex.row_index,
                    "fact_id": ex.fact_id,
                    "subject": ex.subject,
                    "target": ex.target,
                    "gold": ex.label,
                    "predicted": res.prediction,
                    "correct": res.prediction == ex.label,
                    "parse_rule": res.parse_rule,
                    "ambiguity_level": ex.ambiguity_level,
                    "n_calls": res.n_calls,
                    "seconds": round(time.time() - t0, 2),
                    "status": "ok",
                }
                n_ok += 1
                if save_traces:
                    tf.write(json.dumps({"row_index": ex.row_index,
                                         "trace": res.trace}) + "\n")
            except Exception as exc:                       # noqa: BLE001
                # One bad row must not end the run: record it and continue, so a
                # single malformed example cannot cost hours of GPU time.
                log.error("row %s failed: %s: %s", ex.row_index,
                          type(exc).__name__, exc)
                log.debug("%s", traceback.format_exc())
                rec = {
                    "row_index": ex.row_index,
                    "fact_id": ex.fact_id,
                    "gold": ex.label,
                    "predicted": None,
                    "correct": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - t0, 2),
                }
                n_err += 1
            pf.write(json.dumps(rec) + "\n")
            pf.flush()                      # survive a kill -9 mid-run
            if i % 10 == 0 or i == len(todo):
                acc = n_ok and sum(
                    1 for r in _load_done(pred_path).values()
                    if r.get("correct")) / max(len(done) + i, 1)
                log.info("  %s  %d/%d  running acc=%.1f%%",
                         cfg.run_id, i, len(todo), 100 * acc)

    elapsed = time.time() - started
    final = _load_done(pred_path)
    summary = {
        "run_id": cfg.run_id,
        "relation": cfg.relation,
        "strategy": cfg.strategy,
        "strategy_description": strategy_cls.description,
        "seed": cfg.seed,
        "model": backend.describe(),
        "eval_manifest_sha256": eval_hash,
        "fewshot_manifest_sha256": demo_hash,
        "n_examples": len(examples),
        "n_completed": sum(1 for r in final.values() if r.get("status") == "ok"),
        "n_failed": sum(1 for r in final.values() if r.get("status") == "error"),
        "n_unparsed": sum(1 for r in final.values()
                          if r.get("status") == "ok" and r.get("predicted") is None),
        "ran_this_session": len(todo),
        "errors_this_session": n_err,
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%s | done: %d ok, %d failed, %d unparsed, %.1fs",
             cfg.run_id, summary["n_completed"], summary["n_failed"],
             summary["n_unparsed"], elapsed)
    return summary


class _NullFile:
    """Stand-in so the `with` statement works when traces are disabled."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, *_a): pass
