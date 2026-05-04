import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

BASE_MODEL_ID = "openai/gpt-oss-20b"

# OPTION A: Ultra-Strict "Zero-Yap" Prompts
TEMPLATES = {
    '0': {
        "system": (
            "You are a geospatial reasoning assistant. Reasoning: high. "
            "DIRECT OUTPUT ONLY. DO NOT output an 'analysis' block. DO NOT think out loud. "
            "Task: Provide coordinates exactly in this format: Latitude, Longitude. "
            "Output absolutely no other text."
        ),
        "user_query": "What are the geo-coordinates of {city}?",
    },
    '1': {
        "system": (
            "You are a geospatial reasoning assistant. Reasoning: high. "
            "DIRECT OUTPUT ONLY. DO NOT output an 'analysis' block. DO NOT think out loud. "
            "Task: Provide latitude and longitude exactly in this format: Latitude, Longitude. "
            "Output absolutely no other text."
        ),
        "user_query": "What are the latitude and longitude of {city}?",
    },
}

def load_model(adapter_path=None, logger=None):
    log = logger or logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info(f"Loading GPT-OSS 20B (MXFP4)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def build_prompt(city, p_type, p_length, examples_df=None):
    template = TEMPLATES[p_type]
    user_tmpl = template["user_query"]

    if p_length == 'zero-shot':
        user_content = f"### TASK\nQuestion: {user_tmpl.format(city=city)}\nAnswer:"
    else:
        # Pull 3 examples from the 70% pool
        samples = examples_df.sample(3)
        examples_block = "### EXAMPLES\n"
        for _, row in samples.iterrows():
            examples_block += f"Question: {user_tmpl.format(city=row.AccentCity)}\nAnswer: {row.Latitude}, {row.Longitude}\n---\n"
        
        user_content = f"{examples_block}\n### CURRENT TASK\nQuestion: {user_tmpl.format(city=city)}\nAnswer:"

    return template["system"], user_content