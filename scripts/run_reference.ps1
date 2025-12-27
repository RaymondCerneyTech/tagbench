param(
    [string]$ModelPath = $env:GOLDEVIDENCEBENCH_MODEL,
    [ValidateSet("quick", "standard")]
    [string]$Preset = "quick"
,
    [float]$NoteRate = 0.12
)

if (-not $ModelPath) {
    Write-Error "Set -ModelPath or GOLDEVIDENCEBENCH_MODEL before running."
    exit 1
}

$env:GOLDEVIDENCEBENCH_MODEL = $ModelPath

Write-Host "Running baseline (no rerank)..."
.\scripts
un_selector_bench.ps1 -Preset $Preset -ModelPath $ModelPath

Write-Host "Running reranker (latest_step)..."
.\scripts
un_selector_bench.ps1 -Preset $Preset -ModelPath $ModelPath -UseRerank

Write-Host "Collecting summaries..."
python .\scripts\collect_runs.py --runs-dir .\runs --out-csv .\runs\summary_all.csv --latest-only

Write-Host "Done. See runs\summary_all.csv for comparison."