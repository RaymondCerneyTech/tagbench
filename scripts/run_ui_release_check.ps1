param(
    [string]$OutRoot = "",
    [int[]]$Duplicates = @(1, 2, 3, 4, 5),
    [int]$Steps = 6,
    [int]$Seed = 0,
    [string]$Labels = "Next,Continue,Save",
    [string]$Adapter = "goldevidencebench.adapters.ui_fixture_adapter:create_adapter",
    [string]$SelectionMode = "",
    [string]$SelectionSeed = "0",
    [switch]$UpdateConfig,
    [string]$ConfigPath = "configs\\usecase_checks.json",
    [string]$CheckId = "ui_same_label_wall",
    [string]$Metric = "metrics.wrong_action_rate",
    [double]$Threshold = 0.10,
    [ValidateSet("gte", "lte")]
    [string]$Direction = "gte",
    [switch]$UseWall
)

if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "runs\\ui_same_label_wall_$stamp"
}

Write-Host "Running UI same_label stub..."
.\scripts\run_ui_same_label_stub.ps1

Write-Host "Running UI popup_overlay stub..."
.\scripts\run_ui_popup_overlay_stub.ps1

Write-Host "Running UI same_label wall sweep..."
.\scripts\run_ui_same_label_wall.ps1 `
    -OutRoot $OutRoot `
    -Duplicates $Duplicates `
    -Steps $Steps `
    -Seed $Seed `
    -Labels $Labels `
    -Adapter $Adapter `
    -SelectionMode $SelectionMode `
    -SelectionSeed $SelectionSeed

if ($UpdateConfig) {
    $useWallFlag = $null
    if ($UseWall) {
        $useWallFlag = "--use-wall"
    }
    python .\scripts\find_ui_wall.py --runs-dir $OutRoot `
        --metric $Metric --threshold $Threshold --direction $Direction `
        --update-config $ConfigPath --check-id $CheckId $useWallFlag
}
