"""LoRA fine-tuning, one adapter per relation family.

One adapter per family, never one across all three. The families do not share
a label set, and a single adapter would let the model learn which family a
question came from -- turning part of the task into format recognition. Kept
apart, each adapter answers a closed question and the comparison against the
base model is about that question alone.

The target string is the same one the evaluator parses, produced by the same
strategy object that builds the evaluation prompt. Training and testing cannot
drift apart through a copied format string, because there is no copy.

Loss is taken on the answer only. With the prompt included, most of the
gradient would come from reproducing a task header that is identical on every
row, and the model would be learning to recite the instructions.

    python3 -m spatial_eval.cli finetune -r topological
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import REPO_ROOT, ModelConfig, env_guards
from .data import Example, labels_for, load_train
from .strategies import Context, get_strategy

log = logging.getLogger(__name__)

ADAPTERS = REPO_ROOT / "adapters"


@dataclass
class FinetuneConfig:
    relation: str
    base_model: str = ModelConfig.model_id
    # The evaluation drops level 6, so training drops it too: a model trained
    # on a level the report never scores is being shaped by data outside the
    # experiment.
    exclude_levels: tuple[str, ...] = ("Level 6",)
    strategy: str = "zero_shot"     # the format the adapter is trained to answer in
    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_len: int = 1024
    seed: int = 1
    save_every: int = 50            # optimiser steps between checkpoints

    @property
    def out_dir(self) -> Path:
        return ADAPTERS / self.relation


def _render(ex: Example, relation: str, strategy_name: str) -> tuple[str, str]:
    """The prompt the model will see, and the answer it should produce."""
    strategy = get_strategy(strategy_name)()
    ctx = Context(relation=relation, labels=labels_for(relation),
                  generate=lambda *a, **k: "", seed=0, demos=None)
    return strategy.build_prompt(ex, ctx), f"ANSWER: {ex.label}"


def train(cfg: FinetuneConfig) -> dict:
    env_guards()
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError as exc:                                # pragma: no cover
        raise RuntimeError(
            "peft is required for fine-tuning: pip install 'peft>=0.11,<1'") from exc

    torch.manual_seed(cfg.seed)
    examples = load_train(cfg.relation, cfg.exclude_levels)
    log.info("%s | %d training rows (levels excluded: %s)",
             cfg.relation, len(examples), ", ".join(cfg.exclude_levels) or "none")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex: Example):
        prompt, answer = _render(ex, cfg.relation, cfg.strategy)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
        # Truncate the prompt, never the answer: a clipped answer would teach
        # the model to stop mid-label.
        room = cfg.max_len - len(a_ids)
        if room < 1:
            raise ValueError(f"max_len={cfg.max_len} too small for the answer")
        p_ids = p_ids[-room:]
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids[:]      # loss on the answer only
        return {"input_ids": ids, "labels": labels}

    encoded = [encode(e) for e in examples]

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad] * (width - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"]))
                 for b in batch]),
            "labels": torch.tensor(
                [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch]),
        }

    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    try:
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        # Trades compute for memory. On a MIG slice that is the difference
        # between training and an allocator failure.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.out_dir / "trainer_state.json"
    resume = state_path.exists() and (cfg.out_dir / "adapter_config.json").exists()
    if resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # The server is stopped roughly hourly, so a run that cannot resume
        # would never finish. The adapter on disk is the checkpoint.
        model = PeftModel.from_pretrained(model, cfg.out_dir, is_trainable=True)
        log.info("resuming %s at epoch %d, step %d",
                 cfg.relation, state["epoch"], state["global_step"])
    else:
        state = {"epoch": 0, "global_step": 0, "seen_in_epoch": 0, "losses": []}
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    def epoch_loader(epoch: int) -> DataLoader:
        """This epoch's batches, in an order that depends only on the epoch.

        Seeding one generator for the whole run makes each epoch's order
        depend on how many epochs preceded it. After an interruption the
        loader is rebuilt from scratch, so epoch 2 would be replayed in
        epoch 1's order -- and skipping the batches already done would skip
        the wrong ones, training twice on some rows and never on others.
        """
        return DataLoader(
            encoded, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch))

    n_batches_total = math.ceil(len(encoded) / cfg.batch_size)
    steps_per_epoch = math.ceil(n_batches_total / cfg.grad_accum)

    opt_path = cfg.out_dir / "optimizer.pt"
    if resume and opt_path.exists():
        # Without this the moments restart at zero on every resume, and on a
        # server interrupted hourly the optimiser would spend the whole run
        # warming up. Only the adapter's parameters are in it: a few tens of
        # megabytes, not the base model's.
        opt.load_state_dict(torch.load(opt_path, map_location=model.device))
        log.info("restored optimiser state from %s", opt_path)

    def checkpoint(st):
        model.save_pretrained(cfg.out_dir)
        torch.save(opt.state_dict(), opt_path)
        st["config"] = {**asdict(cfg), "exclude_levels": list(cfg.exclude_levels)}
        st["n_train_rows"] = len(examples)
        st["base_model"] = cfg.base_model
        state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")

    model.train()
    started = time.time()
    resume_epoch, resume_at = state["epoch"], state["seen_in_epoch"]
    for epoch in range(resume_epoch, cfg.epochs):
        loader = epoch_loader(epoch)
        running, n_batches = 0.0, 0
        for i, batch in enumerate(loader):
            # Batches already done in this epoch before an interruption.
            # Compared against the values read at startup, not the running
            # state, which this loop rewrites as each epoch completes.
            if epoch == resume_epoch and i < resume_at:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            n_batches += 1
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if state["global_step"] % 10 == 0:
                    log.info("%s | epoch %d  step %d/%d  loss %.4f",
                             cfg.relation, epoch + 1, state["global_step"],
                             steps_per_epoch * cfg.epochs, running / max(n_batches, 1))
                if state["global_step"] % cfg.save_every == 0:
                    state["seen_in_epoch"] = i + 1
                    checkpoint(state)
        state["losses"].append(running / max(n_batches, 1))
        state["epoch"] = epoch + 1
        state["seen_in_epoch"] = 0
        checkpoint(state)
        log.info("%s | epoch %d done, mean loss %.4f",
                 cfg.relation, epoch + 1, state["losses"][-1])

    state["elapsed_seconds"] = round(time.time() - started, 1)
    state["finished"] = True
    checkpoint(state)
    log.info("%s | adapter written to %s", cfg.relation, cfg.out_dir)
    return state
