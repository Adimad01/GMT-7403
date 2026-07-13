#!/usr/bin/env bash
# =============================================================================
# setup_and_run.sh — one-shot server env setup + launch the Relative pipeline
#
# Repairs the JupyterHub environment after a container reset, then starts
# run_experiments.sh (which resumes from checkpoints by default).
#
# What it fixes (learned the hard way):
#   • torchaudio/torchvision 0.13/0.14 pin torch==1.13.1 and silently DOWNGRADE
#     torch during any 'pip install transformers/peft/trl' → uninstall them
#     (the eval engine stubs both, they are never used)
#   • torch must be 2.5.1+cu121 → installed explicitly, never as a dependency
#   • NVIDIA cu12 runtime libs (cupti, cublas, …) must live in ~/.local so
#     they survive resets and sit on one predictable path
#   • LD_LIBRARY_PATH must include the nvidia */lib dirs (both ~/.local and
#     /opt/conda) or torch fails with 'libcupti.so.12: cannot open'
#
# Usage (on the server):
#   bash setup_and_run.sh                 # setup, verify, then resume pipeline
#   bash setup_and_run.sh --skip-train    # setup, then eval-only resume
#   bash setup_and_run.sh --fresh         # setup, then wipe results + retrain
#   bash setup_and_run.sh --setup-only    # setup + verify, don't launch
# =============================================================================

set -uo pipefail

line() { echo "──────────────────────────────────────────────────────────────"; }

# ── parse args: keep pipeline flags, strip --setup-only ─────────────────────
SETUP_ONLY=0
PIPELINE_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--setup-only" ]]; then
        SETUP_ONLY=1
    else
        PIPELINE_ARGS+=("$arg")
    fi
done

# ── library path (must be set before any torch import) ──────────────────────
set_lib_path() {
    local dirs
    dirs=$(ls -d "$HOME"/.local/lib/python3.10/site-packages/nvidia/*/lib \
                 /opt/conda/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${dirs}${LD_LIBRARY_PATH:-}"
}
set_lib_path

env_ok() {
    python - <<'PY' >/dev/null 2>&1
import torch, transformers, peft
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert torch.cuda.is_available()
PY
}

echo ""
line
echo "  STEP 1 — Check environment (torch 2.5.1 + CUDA + transformers + peft)"
line
if env_ok; then
    echo "  [OK] environment already healthy — skipping installs"
else
    echo "  [FIX] environment broken — repairing..."

    echo ""
    echo "  1a. Remove torchaudio + ALL stale torchvision copies (~/.local AND /opt/conda)"
    # pip uninstall removes one copy per call; container resets restore an old
    # cu116 torchvision in /opt/conda that poisons transformers — purge both.
    for _ in 1 2; do pip uninstall -y torchaudio torchvision 2>/dev/null || true; done

    echo ""
    echo "  1b. Install torch 2.5.1 + matching torchvision (never as deps)"
    pip install --user torch==2.5.1
    # torchvision 0.20.1 pairs with torch 2.5.1; transformers 5.x needs a real
    # torchvision (InterpolationMode.NEAREST_EXACT) or peft fails to import.
    # --no-deps so it can never touch torch; PyPI index avoids the broken
    # download-r2.pytorch.org host.
    pip install --user --no-deps --index-url https://pypi.org/simple torchvision==0.20.1

    echo ""
    echo "  1c. Install the ML stack"
    pip install --user transformers peft trl

    echo ""
    echo "  1d. Force NVIDIA cu12 runtime libs into ~/.local (survives resets)"
    pip install --user --force-reinstall --no-deps \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nvtx-cu12==12.1.105

    set_lib_path   # re-glob: picks up freshly created lib dirs
fi

# persist LD_LIBRARY_PATH for future shells (guarded, added once)
if ! grep -q "nvidia/\*/lib" ~/.bashrc 2>/dev/null; then
    # shellcheck disable=SC2016
    echo 'export LD_LIBRARY_PATH="$(ls -d $HOME/.local/lib/python3.10/site-packages/nvidia/*/lib /opt/conda/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo "  [OK] LD_LIBRARY_PATH persisted to ~/.bashrc"
fi

echo ""
line
echo "  STEP 2 — Verify"
line
if python -c "import torch, transformers, peft; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__)"; then
    if ! env_ok; then
        echo "  [ERROR] imports work but torch/CUDA check failed (wrong version or no GPU)."
        echo "          Run: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
        exit 1
    fi
    echo "  [OK] environment verified"
else
    echo "  [ERROR] environment still broken after repair — see import error above."
    exit 1
fi

if [[ $SETUP_ONLY -eq 1 ]]; then
    echo ""
    echo "  --setup-only: done. Launch later with: bash run_experiments.sh"
    exit 0
fi

echo ""
line
echo "  STEP 3 — Launch pipeline (resumes from checkpoints by default)"
line
exec bash run_experiments.sh "${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}"
