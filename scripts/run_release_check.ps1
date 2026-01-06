param(
    [string]$ModelPath = $env:GOLDEVIDENCEBENCH_MODEL,
    [switch]$RunSweeps,
    [switch]$SkipThresholds
)

if ($RunSweeps -and -not $ModelPath) {
    Write-Error "Set -ModelPath or GOLDEVIDENCEBENCH_MODEL before running sweeps."
    exit 1
}

if ($RunSweeps) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stressRoot = "runs\\wall_update_burst_full_linear_bucket10_$stamp"
    $pinRoot = "runs\\wall_update_burst_full_linear_bucket10_pin_$stamp"

    Write-Host "Running stress sweep..."
    .\scripts\run_update_burst_full_linear_bucket10.ps1 `
        -ModelPath $ModelPath `
        -OutRoot $stressRoot `
        -Rates 0.205,0.209,0.22,0.24

    Write-Host "Running pin sweep..."
    .\scripts\run_update_burst_full_linear_bucket10.ps1 `
        -ModelPath $ModelPath `
        -OutRoot $pinRoot `
        -Rates 0.18,0.19,0.195,0.20 `
        -FindWall:$true

    Write-Host "Sweeps complete: $stressRoot, $pinRoot"
}

if (-not $SkipThresholds) {
    Write-Host "Running instruction override gate..."
    .\scripts\run_instruction_override_gate.ps1 -ModelPath $ModelPath
    Write-Host "Running UI same_label stub..."
    .\scripts\run_ui_same_label_stub.ps1
    Write-Host "Running UI popup_overlay stub..."
    .\scripts\run_ui_popup_overlay_stub.ps1
    python .\scripts\check_thresholds.py --config .\configs\usecase_checks.json
}
