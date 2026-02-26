#!/usr/bin/env python3
"""
run_all_inference.py
Python wrapper converted from run_all_inference.ps1 / run_all_inference.sh
Usage:
  python run_all_inference.py
Optional args override defaults (see --help).
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
import shutil


def detect_gpu():
    # Prefer nvidia-smi
    try:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            out = subprocess.check_output([nvsmi, "--query-gpu=name", "--format=csv,noheader"], stderr=subprocess.DEVNULL)
            name = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
            return name
    except Exception:
        pass
    # Fallback to python torch
    try:
        import importlib
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            import torch
            if torch.cuda.is_available():
                try:
                    return torch.cuda.get_device_name(0)
                except Exception:
                    return "GPU (unknown)"
    except Exception:
        pass
    return "CPU"


def run_command(cmd_args, env=None):
    print("+ ", " ".join(cmd_args))
    res = subprocess.run(cmd_args, env=env)
    if res.returncode != 0:
        raise SystemExit(f"Command failed (exit {res.returncode}): {' '.join(cmd_args)}")


def main():
    parser = argparse.ArgumentParser(description="Run zero- and few-shot inference using run_gptoss_inference.py")
    parser.add_argument("--hf-token", default="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", help="HF API token")
    parser.add_argument("--base-model", default="unsloth/gpt-oss-20b-bnb-4bit")
    parser.add_argument("--adapter-dir", default="Topological_Reasoning_GPTOSS_Standard/final_adapter")
    parser.add_argument("--data", default="../dataset/triplet_update_v3_30.csv")
    parser.add_argument("--shot-file", default="../dataset/triplet_update_v3_70.csv")
    parser.add_argument("--system-prompt-file", default="results/system_prompt.txt")
    parser.add_argument("--user-prompt-file", default="results/user_prompt_template.txt")
    parser.add_argument("--checkpoint-interval", default="5")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-subdir", default="../results/Results Gptoss20b FT")
    parser.add_argument("--n-shots", default="4", type=int)
    parser.add_argument("--no-zero", action="store_true", help="Skip zero-shot run")
    parser.add_argument("--no-few", action="store_true", help="Skip few-shot run")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable to run inference script")

    args = parser.parse_args()

    # Move to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Prepare env
    env = os.environ.copy()
    if args.hf_token:
        env["HF_API_TOKEN"] = args.hf_token
        print("Using provided HF_API_TOKEN (from arg or environment)")
    elif "HF_API_TOKEN" in env and env["HF_API_TOKEN"]:
        print("Using HF_API_TOKEN from environment")
    else:
        print("Warning: HF_API_TOKEN not set; calls requiring auth may fail")

    # Ensure results folder exists
    os.makedirs(args.results_subdir, exist_ok=True)
    out_zero = os.path.join(args.results_subdir, "gptoss_preds_30_zero.jsonl")
    out_few = os.path.join(args.results_subdir, "gptoss_preds_30_few.jsonl")

    # Save run info
    gpu_name = detect_gpu()
    run_info_path = os.path.join(args.results_subdir, "run_info.txt")
    with open(run_info_path, "w", encoding="utf8") as f:
        f.write(f"Run started: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Compute device: {gpu_name}\n")
    print(f"Compute device detected: {gpu_name}")

    # Build common arg list for run_gptoss_inference.py
    base_cmd = [args.python_exec, "run_gptoss_inference.py",
                "--base-model", args.base_model,
                "--adapter-dir", args.adapter_dir,
                "--data", args.data,
                "--system-prompt-file", args.system_prompt_file,
                "--user-prompt-file", args.user_prompt_file,
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--llm-offload",
                "--checkpoint-interval", str(args.checkpoint_interval),
                "--temperature", "0"]

    # Zero-shot
    if not args.no_zero:
        print("Running ZERO-SHOT inference...")
        cmd = list(base_cmd)
        cmd += ["--output", out_zero, "--mode", "zero"]
        run_command(cmd, env=env)
    else:
        print("Skipping zero-shot run")

    # Few-shot
    if not args.no_few:
        print(f"Running FEW-SHOT inference (n_shots={args.n_shots})...")
        cmd = list(base_cmd)
        cmd += ["--output", out_few, "--mode", "few", "--n-shots", str(args.n_shots), "--shot-file", args.shot_file]
        run_command(cmd, env=env)
    else:
        print("Skipping few-shot run")

    print(f"All runs completed. Outputs:\n - Zero-shot: {out_zero}\n - Few-shot: {out_few}")


if __name__ == "__main__":
    main()
