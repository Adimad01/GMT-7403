"""Dataset access, pinned by manifest.

The central guarantee of this project: every strategy evaluates the *same* rows
in the same order, with the *same* few-shot demonstrations. That is enforced
here, not left to convention.

Two manifests per relation:

  eval_manifest.json     which rows are evaluated, in order, with a sha256 over
                         their content
  fewshot_manifest.json  eval row -> the exact training rows used as demos

Both are data files under version control. Loading verifies the hash, so a run
against altered data fails immediately instead of producing numbers that are
quietly incomparable with everything else.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import COLUMNS, DATA_DIR, LABELS


class ManifestError(RuntimeError):
    """Raised when the data on disk does not match its manifest."""


@dataclass(frozen=True)
class Example:
    """One evaluation item."""
    row_index: int
    fact_id: str          # rows sharing this assert the same fact; not independent
    subject: str
    target: str
    label: str            # gold
    ambiguity_level: str
    text: str             # the natural-language description shown to the model

    @property
    def key(self) -> str:
        return str(self.row_index)


@dataclass(frozen=True)
class Demo:
    subject: str
    target: str
    label: str
    ambiguity_level: str
    text: str


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relation_dir(relation: str) -> Path:
    return DATA_DIR / relation


def load_eval_manifest(relation: str) -> dict:
    path = relation_dir(relation) / "eval_manifest.json"
    if not path.exists():
        raise ManifestError(f"missing eval manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        "".join(r["row_sha256"] for r in manifest["rows"]).encode()).hexdigest()
    if digest != manifest["manifest_sha256"]:
        raise ManifestError(
            f"{relation}: eval manifest is internally inconsistent "
            f"(recomputed {digest[:12]}, recorded {manifest['manifest_sha256'][:12]}). "
            "The manifest file has been edited by hand or corrupted.")
    return manifest


def load_examples(relation: str, limit: int | None = None,
                  row_indices: tuple[int, ...] | None = None,
                  ) -> tuple[list[Example], str]:
    """Return the pinned evaluation examples and the manifest hash.

    No filtering happens here, deliberately. An earlier version of this project
    dropped rows whose entities failed to geocode, reading a mutable cache at
    run time -- so arms run before and after a cache refresh silently evaluated
    different rows. The manifest is the single source of truth.
    """
    manifest = load_eval_manifest(relation)
    cols = COLUMNS[relation]

    src = relation_dir(relation) / ("eval.csv" if (relation_dir(relation) / "eval.csv").exists()
                                    else "corpus.csv")
    rows = _read_csv(src)

    examples: list[Example] = []
    for entry in manifest["rows"]:
        i = entry["row_index"]
        if i >= len(rows):
            raise ManifestError(
                f"{relation}: manifest row_index {i} out of range for {src.name} "
                f"({len(rows)} rows). Data and manifest are out of sync.")
        row = rows[i]
        ex = Example(
            row_index=i,
            fact_id=entry["fact_id"],
            subject=row[cols["subject"]].strip(),
            target=row[cols["object"]].strip(),
            label=row[cols["label"]].strip().lower(),
            ambiguity_level=row.get("ambiguity_level", "").strip(),
            text=row.get(cols["text"], "").strip(),
        )
        if ex.label != entry["label"]:
            raise ManifestError(
                f"{relation}: row {i} label is '{ex.label}' but the manifest "
                f"recorded '{entry['label']}'. The CSV has changed since the "
                "manifest was frozen; regenerate it and rerun every arm.")
        examples.append(ex)

    if row_indices is not None:
        # Named rows, for inspecting particular answers. Order follows the
        # manifest, not the order they were asked for, so the result reads
        # the same however the request was written. Not called `rows`: that
        # name already holds the CSV lines in this function, and shadowing it
        # turned the selection into a set of dicts.
        wanted = set(row_indices)
        found = {e.row_index for e in examples} & wanted
        if missing := wanted - found:
            raise ManifestError(
                f"{relation}: row_index {sorted(missing)} not in the eval set "
                f"(it holds {len(examples)} rows)")
        examples = [e for e in examples if e.row_index in wanted]
    if limit is not None:
        examples = examples[:limit]
    return examples, manifest["manifest_sha256"]


def load_demos(relation: str) -> tuple[dict[str, list[Demo]], str]:
    """Return eval-row-key -> demonstrations, and the demo map hash.

    Few-shot demos are pinned for the same reason the eval set is: sampling them
    at run time would give different arms different demonstrations for the same
    question, and the comparison would no longer be about the strategy.
    """
    path = relation_dir(relation) / "fewshot_manifest.json"
    if not path.exists():
        raise ManifestError(f"missing few-shot manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    eval_manifest = load_eval_manifest(relation)
    if manifest["eval_manifest_sha256"] != eval_manifest["manifest_sha256"]:
        raise ManifestError(
            f"{relation}: few-shot manifest was built against a different eval "
            "manifest. Regenerate it (scripts/build_manifests.py) and rerun all "
            "few-shot arms.")

    cols = COLUMNS[relation]
    train = _read_csv(relation_dir(relation) / "train.csv")

    demos: dict[str, list[Demo]] = {}
    for key, idxs in manifest["demos"].items():
        items = []
        for i in idxs:
            if i >= len(train):
                raise ManifestError(
                    f"{relation}: demo index {i} out of range for train.csv "
                    f"({len(train)} rows).")
            r = train[i]
            items.append(Demo(
                subject=r[cols["subject"]].strip(),
                target=r[cols["object"]].strip(),
                label=r[cols["label"]].strip().lower(),
                ambiguity_level=r.get("ambiguity_level", "").strip(),
                text=r.get(cols["text"], "").strip(),
            ))
        demos[key] = items
    return demos, manifest["demo_map_sha256"]


def labels_for(relation: str) -> list[str]:
    return LABELS[relation]


def load_train(relation: str, exclude_levels: tuple[str, ...] = ()) -> list[Example]:
    """The fine-tuning pool for one relation.

    No manifest pins these: train.csv is not scored, so there is nothing to
    keep comparable across arms. The eval manifest stays the only frozen
    thing, which is what lets a fine-tuned model be compared to the base one
    on identical rows.

    row_index here is the position in train.csv and is unrelated to the
    eval row_index of the same name -- they index different files.
    """
    cols = COLUMNS[relation]
    src = relation_dir(relation) / "train.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist; run scripts/build_splits.py")

    out: list[Example] = []
    for i, row in enumerate(_read_csv(src)):
        level = row.get("ambiguity_level", "").strip()
        if level in exclude_levels:
            continue
        out.append(Example(
            row_index=i,
            fact_id=f"train{i:05d}",
            subject=row[cols["subject"]].strip(),
            target=row[cols["object"]].strip(),
            label=row[cols["label"]].strip().lower(),
            ambiguity_level=level,
            text=row[cols["text"]].strip(),
        ))
    unknown = {e.label for e in out} - set(LABELS[relation])
    if unknown:
        raise ManifestError(
            f"{relation}: train.csv holds labels outside the label set: "
            f"{sorted(unknown)}")
    return out
