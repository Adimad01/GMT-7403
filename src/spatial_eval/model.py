"""Model backends.

`Backend` is the seam for adding models later: implement `generate(prompt)` and
register the class. Nothing above this module knows what a transformer is.
"""
from __future__ import annotations

import logging
import zlib
from abc import ABC, abstractmethod

from .config import ModelConfig, env_guards

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type["Backend"]] = {}


def register_backend(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def build_backend(cfg: ModelConfig) -> "Backend":
    if cfg.backend not in _REGISTRY:
        raise KeyError(f"unknown backend '{cfg.backend}'. "
                       f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.backend](cfg)


class Backend(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @abstractmethod
    def generate(self, prompt: str, seed: int,
                 max_new_tokens: int | None = None) -> str:
        """Return the model's completion for `prompt`.

        `seed` must make the output reproducible for a given (prompt, seed).
        `max_new_tokens` overrides the configured budget for this call only --
        a short extraction call needs a handful of tokens, not the full budget.
        """

    def describe(self) -> dict:
        return {"backend": self.cfg.backend, **self.cfg.as_dict()}


def prompt_seed(base_seed: int, prompt: str) -> int:
    """Derive a per-prompt seed.

    Seeding once at startup would make a row's output depend on how many rows
    ran before it, so resuming a partial run would produce different answers
    than a clean one. Deriving the seed from the prompt makes each row
    reproducible on its own.
    """
    return (base_seed * 1_000_003 + zlib.crc32(prompt.encode("utf-8"))) % (2 ** 31 - 1)


@register_backend("mock")
class MockBackend(Backend):
    """Deterministic stand-in used by the test suite.

    Echoes a fixed pattern so parsing, resume, metrics and reporting can all be
    exercised without a GPU. It is intentionally imperfect: it answers from a
    rotation so accuracy lands between 0 and 1 and the metrics code sees a
    realistic confusion matrix.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.calls = 0

    def generate(self, prompt: str, seed: int,
                 max_new_tokens: int | None = None) -> str:
        self.calls += 1
        # Read the allowed labels out of the prompt itself, so the mock stays
        # agnostic about which relation it is answering and the Backend
        # signature is identical to the real one.
        labels: list[str] = []
        for line in prompt.splitlines():
            if line.startswith("Allowed answers:"):
                labels = [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
                break
        if not labels:
            return "No label list was provided."
        s = prompt_seed(seed, prompt)
        pick = labels[s % len(labels)]
        return f"Considering the description and the alternatives.\nANSWER: {pick}"


@register_backend("hf")
class HFBackend(Backend):
    """Hugging Face causal LM.

    Carries the workarounds the MIG-partitioned A100 needs for gpt-oss-20b.
    They are no-ops on other hardware and other models.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        env_guards()
        self._patch_transformers()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        log.info("loading %s (dtype=%s, mxfp4_dequantize=%s)",
                 cfg.model_id, cfg.dtype, cfg.mxfp4_dequantize)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        kwargs = {"dtype": getattr(torch, cfg.dtype), "device_map": "auto"}
        if cfg.mxfp4_dequantize:
            try:
                from transformers import Mxfp4Config
                kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            except ImportError:
                log.warning("Mxfp4Config unavailable; loading without dequantize")
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()

        self.device = next(self.model.parameters()).device
        max_pos = getattr(self.model.config, "max_position_embeddings", None) \
            or getattr(self.tokenizer, "model_max_length", 2048)
        self.max_input = max(64, max_pos - cfg.max_new_tokens - 32)
        log.info("model ready on %s (max input %d tokens)", self.device, self.max_input)

    @staticmethod
    def _patch_transformers() -> None:
        """MIG A100 workarounds, applied before the model loads."""
        import transformers as tf

        major = int(str(tf.__version__).split(".")[0])
        if major >= 5:
            raise RuntimeError(
                f"transformers {tf.__version__} is not supported. The MXFP4 "
                "dequantisation patch targets the 4.5x loader; 5.x replaced it "
                "and model loading fails on MIG with an NVML assert in the CUDA "
                "caching allocator.\n"
                "  Fix: pip install 'transformers>=4.55,<5'")

        # Run MoE dequantisation on CPU. Doing it on a MIG device trips
        # NVML_SUCCESS == r INTERNAL ASSERT FAILED in the caching allocator.
        try:
            import torch
            import transformers.integrations.mxfp4 as mx

            def wrap(orig):
                def cpu_convert(blocks, scales, *a, **kw):
                    target = blocks.device
                    b = blocks.cpu() if blocks.device.type != "cpu" else blocks
                    s = scales.cpu() if scales.device.type != "cpu" else scales
                    return orig(b, s, *a, **kw).to(target)
                return cpu_convert

            # The function was renamed between releases. Patch whichever exists
            # and say so loudly if neither does: a silently skipped patch looks
            # like success and then fails minutes later during weight loading.
            patched = []
            for name in ("_convert_moe_packed_tensors", "convert_moe_packed_tensors"):
                fn = getattr(mx, name, None)
                if callable(fn):
                    setattr(mx, name, wrap(fn))
                    patched.append(name)
            if patched:
                log.info("patched mxfp4 dequantisation -> CPU (%s)", ", ".join(patched))
            else:
                log.warning("no mxfp4 conversion function found to patch; "
                            "model loading may fail on MIG hardware")
        except ImportError:
            log.debug("transformers.integrations.mxfp4 unavailable; skipping patch")

        try:
            import transformers.modeling_utils as mu
            mu.caching_allocator_warmup = lambda *a, **k: None
            log.info("disabled caching_allocator_warmup (MIG OOM workaround)")
        except Exception as exc:                      # pragma: no cover
            log.debug("warmup patch skipped: %s", exc)

    def generate(self, prompt: str, seed: int,
                 max_new_tokens: int | None = None) -> str:
        from transformers import set_seed
        set_seed(prompt_seed(seed, prompt))

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_input).to(self.device)
        n_in = inputs["input_ids"].shape[-1]
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
