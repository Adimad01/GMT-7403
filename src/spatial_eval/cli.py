"""Command line interface.

    spatial-eval verify                       check data integrity, run nothing
    spatial-eval list                         show relations and strategies
    spatial-eval run --all                    every strategy x every relation
    spatial-eval run -r cardinal -s cot       one cell
    spatial-eval run --all --retry-failed     redo only rows that errored
    spatial-eval evaluate                     recompute metrics from predictions
    spatial-eval report                       comparison across strategies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (LABELS, LOGS_DIR, RELATIONS, RESULTS_DIR, ModelConfig,
                     RunConfig, env_guards)
from .data import ManifestError, load_demos, load_examples
from .metrics import compute, load_predictions
from .model import build_backend
from .report import (collect, render, render_pairwise, render_parse_health,
                     render_per_label, write_csv, write_json)
from .runner import run_cell, setup_logging
from .strategies import available


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model-id", default=ModelConfig.model_id)
    g.add_argument("--backend", default=ModelConfig.backend,
                   help="hf (default) or mock for a dry run without a GPU")
    g.add_argument("--max-new-tokens", type=int, default=ModelConfig.max_new_tokens)
    g.add_argument("--temperature", type=float, default=ModelConfig.temperature)
    g.add_argument("--dtype", default=ModelConfig.dtype)


def _model_from(args) -> ModelConfig:
    return ModelConfig(model_id=args.model_id, backend=args.backend,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, dtype=args.dtype)


def cmd_verify(args) -> int:
    ok = True
    print("=" * 78)
    print("  DATA INTEGRITY")
    print("=" * 78)
    for rel in RELATIONS:
        try:
            examples, ev_hash = load_examples(rel)
            line = (f"  {rel:<13} eval rows={len(examples):>4}  "
                    f"unique facts={len({e.fact_id for e in examples}):>4}  "
                    f"sha={ev_hash[:12]}")
            demos, dm_hash = load_demos(rel)
            missing = [e.row_index for e in examples if e.key not in demos]
            line += f"  demos={len(demos)} sha={dm_hash[:12]}"
            if missing:
                line += f"  ⚠ {len(missing)} rows lack demos"
                ok = False
            print(line + "   OK")
        except ManifestError as exc:
            print(f"  {rel:<13} FAILED\n      {exc}")
            ok = False
    print("\n  " + ("all manifests verified — every strategy will see identical "
                    "examples" if ok else "PROBLEMS FOUND — fix before running"))
    return 0 if ok else 1


def cmd_list(args) -> int:
    print("\n  spatial relations:")
    for r in RELATIONS:
        print(f"    {r:<14} {len(LABELS[r])} labels: {', '.join(LABELS[r])}")
    print("\n  prompting strategies:")
    from .strategies import get_strategy
    for s in available():
        print(f"    {s:<14} {get_strategy(s).description}")
    print()
    return 0


def cmd_run(args) -> int:
    env_guards()
    # An explicit -r/-s always narrows the grid, including alongside --all:
    # `--all -s got` reads as "every relation, this one strategy", and used to
    # silently queue all five instead. Resume made that harmless but confusing.
    relations = [args.relation] if args.relation else RELATIONS
    strategies = [args.strategy] if args.strategy else available()
    seeds = args.seeds

    setup_logging(args.verbose, LOGS_DIR / "run.log")
    model_cfg = _model_from(args)

    # Build the backend once: loading a 20B model takes minutes, and every cell
    # in this process can share it.
    backend = build_backend(model_cfg)

    cells = [(r, s, sd) for r in relations for s in strategies for sd in seeds]
    print(f"\n  {len(cells)} cells: {len(relations)} relations x "
          f"{len(strategies)} strategies x {len(seeds)} seeds\n")

    failures = []
    for i, (rel, strat, seed) in enumerate(cells, 1):
        print(f"  [{i}/{len(cells)}] {rel} / {strat} / seed {seed}")
        cfg = RunConfig(relation=rel, strategy=strat, seed=seed, model=model_cfg,
                        limit=args.limit, resume=not args.no_resume)
        try:
            summary = run_cell(cfg, backend=backend, save_traces=args.save_traces)
            if summary["n_failed"]:
                failures.append(f"{cfg.run_id} ({summary['n_failed']} rows)")
        except Exception as exc:                        # noqa: BLE001
            # A whole cell failing (bad config, missing manifest) must not stop
            # the remaining cells.
            print(f"      CELL FAILED: {type(exc).__name__}: {exc}")
            failures.append(f"{cfg.run_id} (cell error)")

    print("\n" + "=" * 78)
    if failures:
        print(f"  {len(failures)} cell(s) with failures:")
        for f in failures:
            print(f"    - {f}")
        print("\n  Rerun only the failed rows:")
        print("    python3 -m spatial_eval.cli run --all --seeds <same seeds>")
    else:
        print("  all cells completed")
    print("=" * 78 + "\n")
    return 1 if failures else 0


def cmd_evaluate(args) -> int:
    n = 0
    short: list[str] = []
    # Row counts per relation, so a cell still being written is visible rather
    # than reported as a finished small-n result.
    expected: dict[str, int] = {}
    for rel in RELATIONS:
        try:
            expected[rel] = len(load_examples(rel)[0])
        except Exception:
            expected[rel] = 0
    for rel in RELATIONS:
        for strat in available():
            base = RESULTS_DIR / rel / strat
            if not base.exists():
                continue
            for seed_dir in sorted(base.glob("seed*")):
                recs = load_predictions(seed_dir / "predictions.jsonl")
                if not recs:
                    continue
                m = compute(recs, LABELS[rel])
                (seed_dir / "metrics.json").write_text(
                    json.dumps(m, indent=2), encoding="utf-8")
                flag = ""
                if expected.get(rel) and m.get("n", 0) < expected[rel]:
                    flag = f"   << INCOMPLETE ({m.get('n', 0)}/{expected[rel]})"
                    short.append(f"{rel}/{strat}/{seed_dir.name}")
                print(f"  {rel:<13} {strat:<11} {seed_dir.name:<7} "
                      f"acc={m.get('accuracy', 0):>6.2f}%  "
                      f"macroF1={m.get('macro_f1', 0):.3f}  n={m.get('n', 0)}{flag}")
                n += 1
    print(f"\n  wrote metrics.json for {n} cell(s)")
    if short:
        print(f"\n  WARNING: {len(short)} cell(s) are incomplete — a run is still "
              "in progress or died early.")
        print("  Metrics for these are partial and must not be reported:")
        for c in short[:8]:
            print(f"      {c}")
        if len(short) > 8:
            print(f"      ... and {len(short) - 8} more")
    return 0


def cmd_report(args) -> int:
    cells = collect(seeds=set(args.seeds) if args.seeds else None)
    print(render(cells, metric=args.metric))
    if args.per_label:
        print(render_per_label(cells))
    if args.pairwise:
        print(render_pairwise(seeds=set(args.seeds) if args.seeds else None,
                              alpha=args.alpha))
    if args.parse_health:
        print(render_parse_health())
    if args.csv:
        write_csv(cells, Path(args.csv)); print(f"  wrote {args.csv}")
    if args.json:
        write_json(cells, Path(args.json)); print(f"  wrote {args.json}")
    return 0


def cmd_prompts(args) -> int:
    """Print the exact prompt every strategy sends, for one evaluation row.

    Prompts are the experimental manipulation -- the only thing that differs
    between arms -- so they belong in the writeup verbatim rather than being
    reconstructed from the source.
    """
    from .data import load_demos, load_examples
    from .strategies import Context, get_strategy

    examples, _ = load_examples(args.relation)
    ex = next((e for e in examples if e.row_index == args.row), examples[0])
    demos = None
    try:
        demos, _ = load_demos(args.relation)
    except Exception:
        pass

    captured: list[tuple[str, str]] = []

    def recording_generate(prompt, seed, max_new_tokens=None):
        captured.append((f"call {len(captured) + 1}", prompt))
        # A plausible reply keeps multi-step strategies walking their full path.
        return f"Reasoning placeholder.\nANSWER: {ex.label}"

    ctx = Context(relation=args.relation, labels=LABELS[args.relation],
                  seed=1, generate=recording_generate, demos=demos)

    print("=" * 78)
    print(f"  PROMPTS — {args.relation}, eval row {ex.row_index}")
    print("=" * 78)
    print(f"  subject : {ex.subject}")
    print(f"  object  : {ex.target}")
    print(f"  gold    : {ex.label}   ({ex.ambiguity_level})")
    print(f"  text    : {ex.text}")

    for name in (available() if not args.strategy else [args.strategy]):
        captured.clear()
        strat = get_strategy(name)()
        strat.run(ex, ctx)
        print("\n" + "=" * 78)
        print(f"  STRATEGY: {name}   ({len(captured)} model call"
              f"{'s' if len(captured) != 1 else ''})")
        print(f"  {strat.description}")
        print("=" * 78)
        for label, prompt in captured:
            print(f"\n--- {label} " + "-" * (70 - len(label)))
            print(prompt.rstrip())
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spatial-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="check data integrity").set_defaults(func=cmd_verify)
    sub.add_parser("list", help="list relations and strategies").set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="run experiments")
    r.add_argument("-r", "--relation", choices=RELATIONS)
    r.add_argument("-s", "--strategy", choices=available())
    r.add_argument("--all", action="store_true",
                   help="every relation x strategy (the default; -r/-s narrow it)")
    r.add_argument("--seeds", type=int, nargs="+", default=[1])
    r.add_argument("--limit", type=int, help="evaluate only the first N rows (debug)")
    r.add_argument("--no-resume", action="store_true",
                   help="recompute rows that already succeeded")
    r.add_argument("--retry-failed", action="store_true",
                   help="(default behaviour) rerun error rows, keep successes")
    r.add_argument("--save-traces", action="store_true",
                   help="write every prompt and completion to traces.jsonl")
    r.add_argument("-v", "--verbose", action="store_true")
    _add_model_args(r)
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("prompts", help="print the exact prompts each strategy sends")
    pr.add_argument("-r", "--relation", choices=RELATIONS, default="relative")
    pr.add_argument("-s", "--strategy", choices=available())
    pr.add_argument("--row", type=int, help="eval row_index (default: the first)")
    pr.set_defaults(func=cmd_prompts)

    e = sub.add_parser("evaluate", help="recompute metrics from saved predictions")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("report", help="compare strategies")
    c.add_argument("--metric", default="accuracy",
                   choices=["accuracy", "accuracy_by_fact", "macro_f1"])
    c.add_argument("--per-label", action="store_true")
    c.add_argument("--pairwise", action="store_true",
                   help="paired McNemar tests between every strategy pair")
    c.add_argument("--parse-health", action="store_true",
                   help="accuracy with vs without unparseable completions")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--seeds", type=int, nargs="+")
    c.add_argument("--csv", help="also write a CSV here")
    c.add_argument("--json", help="also write JSON here")
    c.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
