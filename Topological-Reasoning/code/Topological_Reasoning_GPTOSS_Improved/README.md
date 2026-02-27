---
base_model: unsloth/gpt-oss-20b-bnb-4bit
library_name: transformers
model_name: Topological_Reasoning_GPTOSS_Improved
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for Topological_Reasoning_GPTOSS_Improved

This model is a fine-tuned version of [unsloth/gpt-oss-20b-bnb-4bit](https://huggingface.co/unsloth/gpt-oss-20b-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.28.0
- Transformers: 5.2.0
- Pytorch: 2.11.0.dev20260216+cu128
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and GallouÃ©dec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```