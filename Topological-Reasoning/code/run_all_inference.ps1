                                                                                                                                                                                                                                                                                                                                                                     # run_all_inference.ps1
# Wrapper to run zero-shot and few-shot inference with the finetuned GPT-OSS adapter.
# Edit the HF token and paths below before running.

# --- CONFIGURE ---
$HF_API_TOKEN = "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"   # <-- paste your HF API token here (or leave blank to use existing env var)
$BaseModel = "unsloth/gpt-oss-20b-bnb-4bit"
$AdapterDir = "Topological_Reasoning_GPTOSS_Standard/final_adapter"
$DataFile = "..\dataset\triplet_update_v3_30.csv"     # data to run inference on
$ShotFile = "..\dataset\triplet_update_v3_70.csv"     # few-shot exemplars source (CSV with spatial_relation)
$SystemPromptFile = "results\system_prompt.txt"
$UserPromptFile = "results\user_prompt_template.txt"
$CheckpointInterval = 5
$BatchSize = 1
$Device = "cuda"

# Output files
$OutZero = "..\results\gptoss_preds_30_zero.jsonl"
$OutFew = "..\results\gptoss_preds_30_few.jsonl"

# --- RUN ---
if ($HF_API_TOKEN -and $HF_API_TOKEN -ne "") {
    Write-Host "Setting HF_API_TOKEN from script variable."
    $env:HF_API_TOKEN = $HF_API_TOKEN
} elseif (-not $env:HF_API_TOKEN) {
    Write-Host "Warning: no HF_API_TOKEN provided; continuing without explicit token."
}

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Create results subfolder and set output paths
$ResultsSubdir = "..\results\Results Gptoss20b FT"
if (-not (Test-Path $ResultsSubdir)) {
    New-Item -ItemType Directory -Path $ResultsSubdir | Out-Null
}

# Override output paths to save into the subfolder
$OutZero = Join-Path $ResultsSubdir "gptoss_preds_30_zero.jsonl"
$OutFew = Join-Path $ResultsSubdir "gptoss_preds_30_few.jsonl"

# Detect compute device and print GPU info (prefer nvidia-smi, fallback to torch)
$gpuName = "CPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $g = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($g) { $gpuName = $g.Trim() }
    } catch { }
} else {
    # Try Python torch check
    try {
        $pyCheck = 'import sys,torch; print(torch.cuda.is_available())'
        $isCuda = & python -c $pyCheck 2>$null
        if ($LASTEXITCODE -eq 0 -and $isCuda -match 'True') {
            $gpuName = (& python -c "import torch; print(torch.cuda.get_device_name(0))") -join ''
        }
    } catch { }
}
Write-Host "Compute device detected: $gpuName"

# Save run info
$runInfo = Join-Path $ResultsSubdir "run_info.txt"
"Run started: $(Get-Date -Format o)" | Out-File -FilePath $runInfo -Encoding utf8
"Compute device: $gpuName" | Out-File -FilePath $runInfo -Append -Encoding utf8

Write-Host "Running ZERO-SHOT inference..."
$q = { param($s) '"' + $s + '"' }

$cmdZero = @(
    "python",
    "run_gptoss_inference.py",
    "--base-model", $BaseModel,
    "--adapter-dir", (& $q $AdapterDir),
    "--data", (& $q $DataFile),
    "--system-prompt-file", (& $q $SystemPromptFile),
    "--user-prompt-file", (& $q $UserPromptFile),
    "--output", (& $q $OutZero),
    "--batch-size", $BatchSize,
    "--mode", "zero",
    "--device", $Device,
    "--llm-offload",
    "--checkpoint-interval", $CheckpointInterval,
    "--temperature", "0"
) -join ' '
Write-Host $cmdZero
Invoke-Expression $cmdZero
if ($LASTEXITCODE -ne 0) { Write-Host "Zero-shot run failed with exit code $LASTEXITCODE"; Pop-Location; exit $LASTEXITCODE }

Write-Host "Running FEW-SHOT inference (n_shots=4)..."
$cmdFew = @(
    "python",
    "run_gptoss_inference.py",
    "--base-model", $BaseModel,
    "--adapter-dir", (& $q $AdapterDir),
    "--data", (& $q $DataFile),
    "--shot-file", (& $q $ShotFile),
    "--system-prompt-file", (& $q $SystemPromptFile),
    "--user-prompt-file", (& $q $UserPromptFile),
    "--output", (& $q $OutFew),
    "--batch-size", $BatchSize,
    "--mode", "few",
    "--n-shots", "4",
    "--device", $Device,
    "--llm-offload",
    "--checkpoint-interval", $CheckpointInterval,
    "--temperature", "0"
) -join ' '
Write-Host $cmdFew
Invoke-Expression $cmdFew
if ($LASTEXITCODE -ne 0) { Write-Host "Few-shot run failed with exit code $LASTEXITCODE"; Pop-Location; exit $LASTEXITCODE }

Write-Host "All runs completed. Outputs:\n - Zero-shot: $OutZero\n - Few-shot: $OutFew"
Pop-Location
