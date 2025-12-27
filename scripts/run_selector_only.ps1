param(
    [ValidateSet("quick", "standard")]
    [string]$Preset = "quick",
    [string]$ModelPath = $env:GOLDEVIDENCEBENCH_MODEL,
    [ValidateSet("none", "latest_step", "last_occurrence", "prefer_set_latest")]
    [string]$Rerank = "none",
    [float]$NoteRate = 0.12
)

if (-not $ModelPath) {
    Write-Error "Set -ModelPath or GOLDEVIDENCEBENCH_MODEL before running."
    exit 1
}

$env:GOLDEVIDENCEBENCH_MODEL = $ModelPath
$env:GOLDEVIDENCEBENCH_REQUIRE_CITATIONS = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_SELECTOR_ONLY = "1"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_WRONG_TYPE = "same_key"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER = "shuffle"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_ORDER_SEED = "0"
$env:GOLDEVIDENCEBENCH_RETRIEVAL_RERANK = $Rerank

if ($Preset -eq "quick") {
    $seeds = 2
    $steps = 160
    $queries = 12
} else {
    $seeds = 5
    $steps = 200
    $queries = 24
}

$ks = @("2","4","8")

foreach ($k in $ks) {
    $env:GOLDEVIDENCEBENCH_RETRIEVAL_K = $k
    $outDir = "runs\selector_only_${Preset}_${Rerank}_k${k}"
    goldevidencebench sweep --out $outDir --seeds $seeds --episodes 1 --steps $steps --queries $queries `
      --state-modes kv --distractor-profiles standard `
      --note-rate $NoteRate `
      --adapter goldevidencebench.adapters.retrieval_llama_cpp_adapter:create_adapter --no-derived-queries `
      --no-twins --require-citations --results-json "$outDir\combined.json" `
      --max-book-tokens 400 --distractor-rate 0.7 --clear-rate 0.01 --tail-distractor-steps 80
    python .\scripts\summarize_results.py --in "$outDir\combined.json" --out-json "$outDir\summary.json"
}

python .\scripts\collect_runs.py --runs-dir .\runs --out-csv .\runs\summary_all.csv --latest-only

exit $LASTEXITCODE
