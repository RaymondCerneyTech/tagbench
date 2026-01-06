param(
    [string]$ModelPath = $env:GOLDEVIDENCEBENCH_MODEL,
    [string]$OutDir = "runs\\release_gates\\instruction_override_gate"
)

if (-not $ModelPath) {
    Write-Error "Set -ModelPath or GOLDEVIDENCEBENCH_MODEL before running."
    exit 1
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Remove-Item Env:\GOLDEVIDENCEBENCH_* -ErrorAction SilentlyContinue
$env:GOLDEVIDENCEBENCH_MODEL = $ModelPath
$env:GOLDEVIDENCEBENCH_RETRIEVAL_DETERMINISTIC_ANSWER = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_COPY_CLAMP = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = "prefer_update_latest"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_K = "4"

goldevidencebench sweep --out $OutDir --seeds 4 --episodes 1 --steps 80 --queries 16 `
  --state-modes kv,set --distractor-profiles standard,instruction_suite `
  --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
  --no-twins --require-citations --results-json "$OutDir\combined.json" `
  --max-book-tokens 400
python .\scripts\summarize_results.py --in "$OutDir\combined.json" --out-json "$OutDir\summary.json"

Write-Host "Instruction override gate: $OutDir\\summary.json"
