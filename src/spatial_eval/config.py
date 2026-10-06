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
# Which evaluation set a run reads. "eval" is the frozen evaluation every result in
# results/ is computed on. Another name (e.g. "probe_crosses") reads
# data/<relation>/<name>.csv with its own manifest and knowledge store, and writes
# under results_<name>/, so a probe can never mix with the main grid:
#     SPATIAL_EVAL_SET=probe_crosses python3 -m spatial_eval.cli run -r topological -s cot
EVAL_SET = os.environ.get("SPATIAL_EVAL_SET", "eval").strip() or "eval"
RESULTS_DIR = REPO_ROOT / ("results" if EVAL_SET == "eval" else f"results_{EVAL_SET}")
# A rerun that must not touch the main grid (e.g. cut answers under a larger token limit)
# names its own results directory, relative to the repository root.
if os.environ.get("SPATIAL_RESULTS_DIR"):
    RESULTS_DIR = REPO_ROOT / os.environ["SPATIAL_RESULTS_DIR"]
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

# All three relations now share one schema. Topological used to carry its own
# column names, left over from a corpus that was replaced; the map is kept
# because reading through it costs nothing and the next schema change will not
# need to touch every call site.
COLUMNS = {
    "topological": {"label": "relation_label", "subject": "source_entity",
                    "object": "target_entity", "text": "corpus"},
    "cardinal": {"label": "relation_label", "subject": "source_entity",
                 "object": "target_entity", "text": "corpus"},
    "relative": {"label": "relation_label", "subject": "source_entity",
                 "object": "target_entity", "text": "corpus",
                 "observer": "observer_entity"},
}

TRAIN_COLUMNS = dict(COLUMNS)


@dataclass(frozen=True)
class ModelConfig:
    """Generation settings. Recorded verbatim in every result file."""
    # Path to a LoRA adapter, or None for the base model. It belongs here so
    # it is written into run.json: a fine-tuned score that cannot be told from
    # a base one by reading the result file is a score waiting to be misread.
    adapter: str | None = None
    # "none" leaves the prompt as it was; "input" puts the stored facts about
    # the entities the question names in front of the description.
    kg_mode: str = "none"
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
    rows: tuple[int, ...] | None = None  # inspect these eval rows and no others
    resume: bool = True

    @property
    def variant(self) -> str:
        """What distinguishes this run's weights from the base model's.

        A fine-tuned arm must not land in the base arm's directory: the two
        are the comparison, and resume would otherwise treat one as a partial
        copy of the other and skip the work.
        """
        kg = "_kg" if self.model.kg_mode not in ("none", "", None) else ""
        if not self.model.adapter:
            return kg
        # Which adapter answered has to be in the path, or a cross-family run
        # -- the cardinal adapter on the relative eval set, say -- would land
        # on the relative adapter's results and resume would call the work
        # already done. The plain "_lora" is kept when the adapter matches its
        # own family, so the paths already written stay where they are.
        name = Path(self.model.adapter).name
        if name == self.relation:
            lora = "_lora"
        elif name == f"{self.relation}_kg":
            # Fine-tuned on prompts that already carried the facts. Kept free
            # of a hyphen so status.py lists it as an arm of the plan rather
            # than as a cross-family probe.
            lora = "_lorakg"
        else:
            lora = f"_lora-{name}"
        return lora + kg

    @property
    def run_id(self) -> str:
        return f"{self.relation}__{self.strategy}__seed{self.seed}{self.variant}"

    @property
    def result_dir(self) -> Path:
        return (RESULTS_DIR / self.relation / self.strategy
                / f"seed{self.seed}{self.variant}")


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
