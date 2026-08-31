"""Experiment configuration.

Everything that can change a result lives here and is written into every result
file, so any number can be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"

# The three spatial relation families. "relation" is the user-facing word;
# each has its own label vocabulary and its own frozen eval manifest.
RELATIONS = ("topological", "cardinal", "relative")

LABELS = {
    "topological": ["contains", "within", "touches", "crosses",
                    "disjoint", "overlaps", "equals"],
    "cardinal": ["north_of", "south_of", "east_of", "west_of",
                 "northeast_of", "northwest_of", "southeast_of", "southwest_of"],
    "relative": ["left_of", "right_of", "in_front_of", "behind", "next_to"],
}

# Column names differ between the topological files and the other two; the rest
# of the code reads through these maps rather than hard-coding names.
COLUMNS = {
    "topological": {"label": "spatial_relation", "subject": "place_name_subject",
                    "object": "place_name_object", "text": "relation_predicate"},
    "cardinal": {"label": "relation_label", "subject": "source_entity",
                 "object": "target_entity", "text": "corpus"},
    "relative": {"label": "relation_label", "subject": "source_entity",
                 "object": "target_entity", "text": "corpus"},
}

TRAIN_COLUMNS = dict(COLUMNS)


@dataclass(frozen=True)
class ModelConfig:
    """Generation settings. Recorded verbatim in every result file."""
    model_id: str = "openai/gpt-oss-20b"
    backend: str = "hf"                 # "hf" | "mock" (tests)
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = True
    dtype: str = "bfloat16"
    # gpt-oss ships MXFP4 weights; dequantising to bf16 is what the MIG A100
    # needs. Harmless for models that are not quantised this way.
    mxfp4_dequantize: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    relation: str
    strategy: str
    seed: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    limit: int | None = None            # debugging: evaluate only the first N rows
    resume: bool = True

    @property
    def run_id(self) -> str:
        return f"{self.relation}__{self.strategy}__seed{self.seed}"

    @property
    def result_dir(self) -> Path:
        return RESULTS_DIR / self.relation / self.strategy / f"seed{self.seed}"


def env_guards() -> None:
    """Environment that must be set before transformers is imported.

    transformers imports TensorFlow through image_transforms whenever TF looks
    importable. On the target cluster TF's generated protobuf code is rejected
    by the installed protobuf, which takes the whole import chain down. Nothing
    here uses TF, so switch it off rather than repair it.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
