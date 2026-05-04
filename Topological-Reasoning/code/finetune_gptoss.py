import argparse
import os
import torch

# ===========================================================================
# DEPENDENCY HELL MONKEY PATCH
# Transformers' new MXFP4 quantizer expects PyTorch >= 2.5.
# Since our server drivers restrict us to PyTorch 2.4.0, we spoof the
# missing modules/functions here BEFORE importing transformers.
# ===========================================================================

# Patch 1: Missing torch.accelerator (PyTorch < 2.4/2.5)
if not hasattr(torch, "accelerator"):
    class DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.accelerator = DummyAccelerator()

# Patch 2: Missing torch.nn.Module.set_submodule (PyTorch < 2.5)
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        """Polyfill for torch.nn.Module.set_submodule missing in PyTorch < 2.5."""
        if target == "":
            raise ValueError("target cannot be empty")
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f"'{type(mod).__name__}' has no attribute '{item}'")
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError(f"'{item}' is not an nn.Module")
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = _set_submodule
# ===========================================================================

import inspect
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
# NOTE: prepare_model_for_kbit_training is intentionally NOT imported.
# It is a bitsandbytes helper for 4-bit/8-bit bnb models only.
# Our model is dequantized to plain bf16, so calling it causes a
# CUDA allocator crash when it tries to upcast params to float32.
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
import trl
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Detect where max_seq_length lives in this TRL version (moves between releases)
# ---------------------------------------------------------------------------
_SFTCONFIG_PARAMS  = set(inspect.signature(SFTConfig.__init__).parameters)
_SFTTRAINER_PARAMS = set(inspect.signature(SFTTrainer.__init__).parameters)

MAX_SEQ_LENGTH = 512

print(f"[INFO] TRL version: {trl.__version__}")
if "max_seq_length" in _SFTCONFIG_PARAMS:
    _SEQ_LEN_IN = "SFTConfig"
    print("[INFO] max_seq_length -> SFTConfig")
elif "max_seq_length" in _SFTTRAINER_PARAMS:
    _SEQ_LEN_IN = "SFTTrainer"
    print("[INFO] max_seq_length -> SFTTrainer")
else:
    _SEQ_LEN_IN = "neither"
    print("[WARN] max_seq_length not found in either class -- omitting (TRL uses its own default).")

# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------
def build_training_prompt(row, tokenizer):
    """Build a complete instruction-following training example string from a dataset row."""
    place_type_subject = str(row.get("placetype_subject", "place"))
    place_type_object  = str(row.get("placetype_object",  "place"))
    geom_type_subject  = str(row.get("geometry_type_subject", "unknown"))
    geom_type_object   = str(row.get("geometry_type_object",  "unknown"))

    relation_predicate = str(row.get("relation_predicate", row.get("Sentence", "")))
    expected_truth     = str(row.get("spatial_relation", "")).lower().strip()

    prompt_text = (
        "### SYSTEM PROMPT ###\n"
        "You are an expert in geospatial topological reasoning using the DE-9IM model.\n\n"
        "### TASK ###\n"
        "Given a vernacular spatial relation between two geographic entities A and B, "
        "determine the correct DE-9IM topological predicate.\n\n"
        "### VALID PREDICATES ###\n"
        "contains, within, touches, crosses, disjoint, overlaps, equals\n\n"
        "### RULES ###\n"
        "1. The relation is DIRECTED from entity A to entity B.\n"
        "2. Consider the geometry types (Point, LineString, Polygon, MultiPolygon) "
        "when reasoning about topological validity.\n"
        "3. You must pick EXACTLY ONE predicate from the list above.\n\n"
        "### OUTPUT FORMAT ###\n"
        "Reasoning: <brief one-sentence justification>\n"
        "Answer: A [<PREDICATE>] B\n\n"
        "### INPUT ###\n"
        f"Relation: A {relation_predicate} B\n"
        f"Entity A: {place_type_subject} (geometry: {geom_type_subject})\n"
        f"Entity B: {place_type_object} (geometry: {geom_type_object})\n\n"
        "### OUTPUT ###\n"
        "Reasoning:"
    )

    target_output = (
        f" The relation '{relation_predicate}' between a {place_type_subject} "
        f"({geom_type_subject}) and a {place_type_object} ({geom_type_object}) "
        f"maps to the topological predicate '{expected_truth}'.\n"
        f"Answer: A [{expected_truth}] B"
    )

    return prompt_text + target_output + tokenizer.eos_token


# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------
def main():
    """Load data, build training set, attach LoRA adapters, and fine-tune the GPT-OSS model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="../dataset/triplet_update_v3_70.csv")
    parser.add_argument("--model-id",   default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="finetuned_gptoss_topological")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load Data & Tokenizer
    # ------------------------------------------------------------------
    print("[1/5] Loading dataset and tokenizer...")
    df = pd.read_csv(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Build Dataset
    # ------------------------------------------------------------------
    print("[2/5] Building training dataset...")
    records = [
        {"text": build_training_prompt(row, tokenizer)}
        for _, row in df.iterrows()
        if not pd.isna(row.get("spatial_relation"))
    ]
    train_dataset = Dataset.from_list(records)
    print(f"      -> {len(records)} training examples loaded.")

    # ------------------------------------------------------------------
    # 3. Load Model — dequantize MXFP4 to bf16 so the model is trainable.
    #
    # gpt-oss-20b ships as a native MXFP4 checkpoint (inference-only).
    # Mxfp4Config(dequantize=True) decompresses weights to bf16 on load.
    # Using `dtype=` instead of the deprecated `torch_dtype=`.
    # ------------------------------------------------------------------
    print(f"[3/5] Loading model {args.model_id} (dequantizing MXFP4 -> bf16 for training)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,                          # replaces deprecated torch_dtype
        quantization_config=Mxfp4Config(dequantize=True),
    )
    print(f"      -> Model dtype: {next(model.parameters()).dtype}")

    # ------------------------------------------------------------------
    # 4. Attach LoRA adapters directly on the bf16 model.
    #
    # prepare_model_for_kbit_training() is intentionally skipped here.
    # That helper is only for bitsandbytes 4-bit/8-bit quantized models.
    # Calling it on a plain bf16 model triggers a CUDA allocator crash
    # because it tries to upcast every parameter to float32 in-place.
    # ------------------------------------------------------------------
    print("[4/5] Attaching LoRA adapters...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("[5/5] Starting training...")

    sftconfig_kwargs  = {}
    sfttrainer_kwargs = {}
    if _SEQ_LEN_IN == "SFTConfig":
        sftconfig_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    elif _SEQ_LEN_IN == "SFTTrainer":
        sfttrainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        report_to="none",
        **sftconfig_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        **sfttrainer_kwargs,
    )

    trainer.train()

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------
    save_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete. Adapter saved to: {save_path}")


if __name__ == "__main__":
    main()