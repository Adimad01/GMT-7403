"""Offline test suite. No GPU, no model download: the mock backend stands in.

Runs with plain python (``python3 tests/test_pipeline.py``) as well as under
pytest, so it works on a cluster where pytest may not be installed.

These cover the properties the experiment design depends on -- identical
examples across strategies, determinism, resume, correct metrics -- because
those are the failures that produce plausible-looking wrong numbers rather than
crashes.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import spatial_eval.config as C                                   # noqa: E402
from spatial_eval.config import LABELS, RELATIONS, ModelConfig, RunConfig  # noqa: E402
from spatial_eval.data import load_demos, load_examples           # noqa: E402
from spatial_eval.metrics import compute, wilson                  # noqa: E402
from spatial_eval.model import build_backend, prompt_seed         # noqa: E402
from spatial_eval.parsing import parse_label                      # noqa: E402
from spatial_eval.runner import run_cell                          # noqa: E402
from spatial_eval.strategies import Context, available, get_strategy  # noqa: E402

MOCK = ModelConfig(backend="mock")


def test_manifests_load_and_verify():
    for rel in RELATIONS:
        examples, h = load_examples(rel)
        assert examples, rel
        assert len(h) == 64
        assert all(e.label in LABELS[rel] for e in examples)


def test_every_strategy_sees_identical_examples():
    """The central fairness guarantee of the whole project."""
    for rel in RELATIONS:
        base = [(e.row_index, e.label) for e in load_examples(rel)[0]]
        for _ in available():
            again = [(e.row_index, e.label) for e in load_examples(rel)[0]]
            assert again == base, rel


def test_demos_are_pinned_stable_and_label_matched():
    for rel in RELATIONS:
        d1, h1 = load_demos(rel)
        d2, h2 = load_demos(rel)
        assert h1 == h2
        assert all(len(v) == 5 for v in d1.values())
        ex = {e.key: e for e in load_examples(rel)[0]}
        for key, demos in d1.items():
            assert all(dm.label == ex[key].label for dm in demos), rel


def test_prompt_seed_deterministic_and_order_independent():
    a = prompt_seed(1, "hello")
    assert a == prompt_seed(1, "hello")
    assert a != prompt_seed(2, "hello")
    assert a != prompt_seed(1, "hello world")


def test_parsing_rules():
    labs = LABELS["cardinal"]
    assert parse_label("ANSWER: north_of", labs, "cardinal")[0] == "north_of"
    assert parse_label("maybe south_of\nANSWER: east_of", labs, "cardinal")[0] == "east_of"
    assert parse_label("it lies to the north", labs, "cardinal")[0] == "north_of"
    assert parse_label("", labs, "cardinal")[0] is None
    assert parse_label("no idea whatsoever", labs, "cardinal")[0] is None
    rel = LABELS["relative"]
    assert parse_label("towards your port arm", rel, "relative")[0] == "left_of"


def test_wilson_reference_values():
    lo, hi = wilson(50, 100)
    assert abs(lo - 40.38) < 0.05 and abs(hi - 59.62) < 0.05
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 27.75) < 0.05


def test_all_strategies_produce_a_valid_prediction():
    backend = build_backend(MOCK)
    for rel in RELATIONS:
        examples, _ = load_examples(rel, limit=2)
        demos, _ = load_demos(rel)
        ctx = Context(relation=rel, labels=LABELS[rel], seed=1,
                      generate=backend.generate, demos=demos)
        for name in available():
            res = get_strategy(name)().run(examples[0], ctx)
            assert res.prediction in LABELS[rel], f"{name}/{rel} -> {res.prediction}"
            assert res.n_calls >= 1 and res.trace


def test_multistep_strategies_make_multiple_calls():
    backend = build_backend(MOCK)
    examples, _ = load_examples("relative", limit=1)
    ctx = Context(relation="relative", labels=LABELS["relative"], seed=1,
                  generate=backend.generate, demos=None)
    assert get_strategy("zero_shot")().run(examples[0], ctx).n_calls == 1
    assert get_strategy("tot")().run(examples[0], ctx).n_calls == 4
    assert get_strategy("got")().run(examples[0], ctx).n_calls == 4


def test_run_writes_results_and_resume_skips_completed():
    tmp = Path(tempfile.mkdtemp())
    original = C.RESULTS_DIR
    try:
        C.RESULTS_DIR = tmp
        cfg = RunConfig(relation="relative", strategy="zero_shot", seed=1,
                        model=MOCK, limit=5)
        s1 = run_cell(cfg, backend=build_backend(MOCK))
        assert s1["n_completed"] == 5
        assert s1["ran_this_session"] == 5
        assert (cfg.result_dir / "predictions.jsonl").exists()
        assert (cfg.result_dir / "run.json").exists()

        # A second run must recompute nothing.
        s2 = run_cell(cfg, backend=build_backend(MOCK))
        assert s2["ran_this_session"] == 0
        assert s2["n_completed"] == 5
    finally:
        C.RESULTS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_records_row_errors_without_aborting():
    tmp = Path(tempfile.mkdtemp())
    original = C.RESULTS_DIR
    try:
        C.RESULTS_DIR = tmp

        class Exploding:
            calls = 0
            cfg = MOCK

            def generate(self, prompt, seed):
                Exploding.calls += 1
                if Exploding.calls == 2:
                    raise RuntimeError("simulated generation failure")
                return "ANSWER: left_of"

            def describe(self):
                return {"backend": "exploding"}

        cfg = RunConfig(relation="relative", strategy="zero_shot", seed=7,
                        model=MOCK, limit=4)
        s = run_cell(cfg, backend=Exploding())
        # one row failed, the rest still ran
        assert s["n_failed"] == 1, s
        assert s["n_completed"] == 3, s
    finally:
        C.RESULTS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_metrics_on_synthetic_records():
    labs = LABELS["relative"]
    recs = [{"row_index": i, "fact_id": f"f{i // 2}", "gold": labs[i % len(labs)],
             "predicted": labs[i % len(labs)] if i % 3 else None,
             "correct": bool(i % 3), "status": "ok", "parse_rule": "answer_tag",
             "ambiguity_level": f"Level {i % 5 + 1}"} for i in range(30)]
    m = compute(recs, labs)
    assert m["n"] == 30
    assert m["unparsed"] == 10
    assert m["n_unique_facts"] == 15          # fact_id clustering
    assert set(m["per_label"]) == set(labs)
    assert 0 <= m["accuracy"] <= 100
    assert 0 <= m["macro_f1"] <= 1


def test_mcnemar_reference_values():
    from spatial_eval.metrics import mcnemar_exact
    assert abs(mcnemar_exact(10, 2) - 0.03857) < 1e-3
    assert abs(mcnemar_exact(8, 0) - 0.0078125) < 1e-6
    assert mcnemar_exact(5, 5) == 1.0          # symmetric -> no evidence
    assert mcnemar_exact(0, 0) == 1.0          # no discordant rows


def test_holm_is_monotone_and_bounded():
    from spatial_eval.metrics import holm_bonferroni
    adj = holm_bonferroni([0.01, 0.02, 0.03])
    assert [round(a, 4) for a in adj] == [0.03, 0.04, 0.04]
    assert holm_bonferroni([]) == []
    assert all(0 <= a <= 1 for a in holm_bonferroni([0.5, 0.9, 0.99]))


def test_parse_health_separates_formatting_from_reasoning():
    from spatial_eval.metrics import accuracy_excluding_unparsed
    # 10 unparseable rows, all of which would have been wrong anyway
    recs = [{"row_index": i, "status": "ok", "correct": i % 2 == 0,
             "predicted": None if i < 10 else "x"} for i in range(50)]
    r = accuracy_excluding_unparsed(recs)
    assert r["n_unparsed"] == 10
    assert r["n_parsed"] == 40
    # parsed-only accuracy must be at least the all-rows figure
    assert r["accuracy_parsed_only"] >= 100 * sum(1 for x in recs if x["correct"]) / 50


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n  {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
