param(
    [string]$OutRoot = "",
    [int[]]$Duplicates = @(1, 2, 3, 4, 5),
    [int]$Steps = 5,
    [int]$Seed = 0,
    [string]$Labels = "Next,Continue,Save",
    [string]$Adapter = "goldevidencebench.adapters.ui_fixture_adapter:create_adapter",
    [string]$SelectionMode = "",
    [string]$SelectionSeed = "0"
)

if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "runs\\ui_same_label_wall_$stamp"
}

New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null

if ($SelectionMode) {
    $env:GOLDEVIDENCEBENCH_UI_SELECTION_MODE = $SelectionMode
    $env:GOLDEVIDENCEBENCH_UI_SELECTION_SEED = $SelectionSeed
}

Write-Host "UI same_label wall sweep"
Write-Host "OutRoot: $OutRoot"
Write-Host "Duplicates: $($Duplicates -join ',')"
Write-Host "Steps: $Steps Seed: $Seed Labels: $Labels"
Write-Host "Adapter: $Adapter"
if ($SelectionMode) {
    Write-Host "Selection mode: $SelectionMode (seed $SelectionSeed)"
}

foreach ($dup in $Duplicates) {
    $runDir = Join-Path $OutRoot "dups$dup"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $fixturePath = Join-Path $runDir "fixture.jsonl"
    $scoreOut = Join-Path $runDir "score.json"
    $summaryOut = Join-Path $runDir "summary.json"
    $configOut = Join-Path $runDir "config.json"

    $config = @{
        duplicates = $dup
        steps = $Steps
        seed = $Seed
        labels = $Labels
        adapter = $Adapter
        selection_mode = $SelectionMode
        selection_seed = $SelectionSeed
    }
    $config | ConvertTo-Json -Depth 4 | Set-Content -Path $configOut -Encoding UTF8

    goldevidencebench ui-generate --out $fixturePath --steps $Steps --duplicates $dup --labels $Labels --seed $Seed
    goldevidencebench ui-score --fixture $fixturePath --adapter $Adapter --out $scoreOut
    goldevidencebench ui-summary --fixture $fixturePath --out $summaryOut
}

if ($SelectionMode) {
    Remove-Item Env:\GOLDEVIDENCEBENCH_UI_SELECTION_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:\GOLDEVIDENCEBENCH_UI_SELECTION_SEED -ErrorAction SilentlyContinue
}

$wallOut = Join-Path $OutRoot "wall_summary.json"
$wallCsv = Join-Path $OutRoot "wall_summary.csv"
python .\scripts\summarize_ui_wall.py --runs-dir $OutRoot --out $wallOut --out-csv $wallCsv
Write-Host "Wall summary: $wallOut"
Write-Host "Wall CSV: $wallCsv"

$latestDir = "runs\\ui_same_label_wall_latest"
New-Item -ItemType Directory -Path $latestDir -Force | Out-Null
$maxDup = ($Duplicates | Measure-Object -Maximum).Maximum
$latestRun = Join-Path $OutRoot "dups$maxDup"
$latestScore = Join-Path $latestRun "score.json"
$latestSummary = Join-Path $latestRun "summary.json"
$latestConfig = Join-Path $latestRun "config.json"
if (Test-Path $latestScore) {
    Copy-Item $latestScore -Destination (Join-Path $latestDir "score.json") -Force
}
if (Test-Path $latestSummary) {
    Copy-Item $latestSummary -Destination (Join-Path $latestDir "summary.json") -Force
}
if (Test-Path $latestConfig) {
    Copy-Item $latestConfig -Destination (Join-Path $latestDir "config.json") -Force
}
Write-Host "Latest UI wall snapshot: $latestDir"
