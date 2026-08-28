"""
diagnose_gpu.py
================================================================================
Isolate the `NVML_SUCCESS == r INTERNAL ASSERT FAILED at
CUDACachingAllocator.cpp:995` failure seen when loading gpt-oss-20b on the
MIG-partitioned A100.

The model load fails inside the mxfp4 dequantization patch, but the traceback
also shows a plain `torch.empty_like(..., device=cuda)` failing the same way.
So the question is not "is the patch right" but "can this process allocate GPU
memory at all, and under which settings".

This script answers that WITHOUT loading a 20B model, so it takes seconds
instead of four minutes. It walks up from the smallest possible allocation to
the operation that actually fails, then tries each candidate workaround in a
fresh subprocess (allocator config is read once at init, so it cannot be
changed in-process).

Usage
  python diagnose_gpu.py            # full report
  python diagnose_gpu.py --probe    # internal: single probe, do not call directly
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROBE_ENVS = [
    ("baseline", {}),
    ("expandable_segments:False", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}),
    ("expandable_segments:True", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ("no caching allocator", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}),
    ("NVML CUDA check off", {"PYTORCH_NVML_BASED_CUDA_CHECK": "0"}),
]


def probe() -> int:
    """Escalating allocation test. Prints one line per step; exits non-zero on
    the first failure so the caller learns exactly how far it got."""
    import torch
    steps = []

    def step(name, fn):
        try:
            fn()
            steps.append((name, "ok", ""))
            return True
        except Exception as e:
            steps.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))
            return False

    ok = True
    ok &= step("cuda.is_available", lambda: torch.cuda.is_available() or (_ for _ in ()).throw(RuntimeError("False")))
    ok &= step("device_properties", lambda: torch.cuda.get_device_properties(0))
    ok &= step("tiny alloc  .cuda()", lambda: torch.zeros(8).cuda())
    ok &= step("cpu->gpu  .to('cuda')", lambda: torch.zeros(8).to("cuda"))
    ok &= step("empty_like on gpu", lambda: torch.empty_like(torch.zeros(8, device="cuda")))
    ok &= step("100MB alloc", lambda: torch.empty(25_000_000, dtype=torch.float32, device="cuda"))
    ok &= step("2GB alloc", lambda: torch.empty(500_000_000, dtype=torch.float32, device="cuda"))
    ok &= step("bf16 matmul", lambda: (torch.randn(512, 512, device="cuda", dtype=torch.bfloat16) @
                                       torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)).sum().item())
    ok &= step("mem_get_info", lambda: torch.cuda.mem_get_info())

    for name, status, err in steps:
        line = f"    {status:<5} {name}"
        if err:
            line += f"\n          {err}"
        print(line)
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.probe:
        sys.exit(probe())

    print("=" * 78)
    print("  GPU / MIG DIAGNOSTIC")
    print("=" * 78)

    # ---- environment --------------------------------------------------------
    print("\n-- versions -------------------------------------------------------------------")
    try:
        import torch
        print(f"    torch         {torch.__version__}   (cuda {torch.version.cuda})")
    except Exception as e:
        print(f"    torch         IMPORT FAILED: {e}")
        sys.exit(1)
    for mod in ("transformers", "accelerate", "peft", "triton", "kernels"):
        try:
            m = __import__(mod)
            print(f"    {mod:<13} {getattr(m, '__version__', '?')}")
        except ImportError as e:
            # "not installed" and "installed but fails to import here" are very
            # different diagnoses: peft imports only AFTER the engines install
            # their torch shims (torch.float8_e8m0fnu, Module.set_submodule),
            # so a bare probe can report it absent when it is merely unshimmed.
            print(f"    {mod:<13} NOT INSTALLED ({e})")
        except Exception as e:
            print(f"    {mod:<13} present but import FAILED: "
                  f"{type(e).__name__}: {str(e)[:90]}")
            print(f"    {'':<13} (may still import inside the engine, after its torch shims)")

    try:
        import transformers as _tf
        if int(str(_tf.__version__).split(".")[0]) >= 5:
            print(f"\n    *** transformers {_tf.__version__} is INCOMPATIBLE with these engines.")
            print("        They patch the 4.5x MXFP4 loader, which 5.x replaced.")
            print("        Fix: pip install 'transformers>=4.55,<5'")
    except Exception:
        pass

    print("\n-- environment ----------------------------------------------------------------")
    for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
              "PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NVML_BASED_CUDA_CHECK",
              "NVIDIA_VISIBLE_DEVICES"):
        print(f"    {k:<32} {os.environ.get(k, '(unset)')}")

    # ---- MIG topology -------------------------------------------------------
    # A MIG instance must usually be addressed by its UUID; an integer index can
    # resolve to the parent GPU, whose NVML queries then fail for the instance.
    print("\n-- MIG topology ---------------------------------------------------------------")
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("\n".join("    " + l for l in out.splitlines()) or "    (no output)")
        if "MIG" in out and not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
            print("\n    NOTE: MIG instances present but CUDA_VISIBLE_DEVICES is not a MIG- UUID.")
            print("          Addressing a MIG device by index is a known source of NVML asserts.")
    except Exception as e:
        print(f"    nvidia-smi unavailable: {e}")

    # ---- escalating allocation, per candidate setting -----------------------
    print("\n-- allocation probes ----------------------------------------------------------")
    print("   Each runs in a fresh process: the CUDA allocator reads its config once at")
    print("   init, so these cannot be tested in-process.\n")

    working = []
    for label, env in PROBE_ENVS:
        print(f"  [{label}]")
        e = dict(os.environ); e.update(env)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe"],
                           capture_output=True, text=True, env=e, timeout=300)
        print(r.stdout.rstrip() or "    (no output)")
        if r.returncode != 0 and r.stderr.strip():
            print("          " + r.stderr.strip().splitlines()[-1][:200])
        if r.returncode == 0:
            working.append((label, env))
        print()

    # ---- verdict ------------------------------------------------------------
    print("=" * 78)
    if working:
        label, env = working[0]
        print(f"  WORKING CONFIGURATION: {label}")
        if env:
            print("\n  Export this before running experiments:")
            for k, v in env.items():
                print(f"    export {k}={v}")
        else:
            print("\n  Plain allocation works. The failure is then specific to the model")
            print("  load path (mxfp4 dequantization), not to CUDA allocation itself.")
            print("  Next step: pin transformers to the version this engine was written")
            print("  against -- the traceback shows the new core_model_loading API.")
    else:
        print("  NO CONFIGURATION ALLOWED GPU ALLOCATION.")
        print("  This is an environment/driver problem, not a code problem:")
        print("    - the MIG instance may need addressing by UUID (see topology above)")
        print("    - the driver and the torch CUDA build may be mismatched")
        print("    - another process may hold the MIG instance exclusively")
    print("=" * 78)


if __name__ == "__main__":
    main()
