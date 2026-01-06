param(
    [string]$FixturePath = "data\\ui_popup_overlay_fixture.jsonl",
    [string]$ConfigPath = "configs\\ui_popup_overlay.json"
)

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config not found: $ConfigPath"
    exit 1
}

if (-not (Test-Path $FixturePath)) {
    Write-Error "Fixture not found: $FixturePath"
    exit 1
}

$lineCount = (Get-Content $FixturePath).Count
$runsDir = "runs"
$gateOut = Join-Path $runsDir "ui_popup_overlay_gate.json"
$summaryOut = Join-Path $runsDir "ui_popup_overlay_summary.json"
New-Item -ItemType Directory -Path $runsDir -Force | Out-Null

Write-Host "UI popup_overlay stub"
Write-Host "Config: $ConfigPath"
Write-Host "Fixture: $FixturePath ($lineCount steps)"
python .\scripts\validate_ui_fixture.py --fixture $FixturePath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
python .\scripts\score_ui_fixture.py --fixture $FixturePath --mode gold
python .\scripts\score_ui_fixture.py --fixture $FixturePath --mode first
python .\scripts\run_ui_adapter_stub.py --fixture $FixturePath
$env:GOLDEVIDENCEBENCH_UI_SELECTION_MODE = "gold"
$env:GOLDEVIDENCEBENCH_UI_SELECTION_SEED = "0"
python .\scripts\run_ui_adapter_stub.py --fixture $FixturePath --observed .\data\ui_popup_overlay_observed_ok.jsonl --out $gateOut
Remove-Item Env:\GOLDEVIDENCEBENCH_UI_SELECTION_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\GOLDEVIDENCEBENCH_UI_SELECTION_SEED -ErrorAction SilentlyContinue
python .\scripts\summarize_ui_fixture.py --fixture $FixturePath --out $summaryOut
Write-Host "UI gate: $gateOut"
Write-Host "UI summary: $summaryOut"
Write-Host "Adapter not implemented yet. Use this fixture to build the UI adapter."
