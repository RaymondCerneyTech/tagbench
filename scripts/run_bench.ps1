param(
    [ValidateSet("smoke", "standard")]
    [string]$Preset = "smoke",
    [string]$ModelPath = $env:GOLDEVIDENCEBENCH_MODEL,
    [string]$LegacyModelPath = $env:TAGBENCH_MODEL,
    [string]$OutDir = "runs",
    [switch]$RequireCitations,
    [switch]$UseDerivedQueries
)

if (-not $ModelPath -and $LegacyModelPath) {
    $ModelPath = $LegacyModelPath
}

if (-not $ModelPath) {
    Write-Error "Set -ModelPath or GOLDEVIDENCEBENCH_MODEL before running."
    exit 1
}

$env:GOLDEVIDENCEBENCH_MODEL = $ModelPath
$env:TAGBENCH_MODEL = $ModelPath
if ($RequireCitations) {
    $env:GOLDEVIDENCEBENCH_REQUIRE_CITATIONS = "1"
$env:TAGBENCH_REQUIRE_CITATIONS = "1"
} else {
    $env:GOLDEVIDENCEBENCH_REQUIRE_CITATIONS = "0"
$env:TAGBENCH_REQUIRE_CITATIONS = "0"
}

if ($Preset -eq "smoke") {
    $seeds = 1
    $episodes = 1
    $steps = 50
    $queries = 4
    $stateModes = "kv"
    $profiles = "standard"
} else {
    $seeds = 2
    $episodes = 1
    $steps = 80
    $queries = 8
    $stateModes = "kv,set"
    $profiles = "standard,instruction"
}

$args = @(
    "sweep",
    "--out", $OutDir,
    "--seeds", $seeds,
    "--episodes", $episodes,
    "--steps", $steps,
    "--queries", $queries,
    "--state-modes", $stateModes,
    "--distractor-profiles", $profiles,
    "--adapter", "goldevidencebench.adapters.llama_cpp_adapter:create_adapter",
    "--results-json", (Join-Path $OutDir "combined.json")
)

if (-not $UseDerivedQueries) {
    $args += "--no-derived-queries"
}
if (-not $RequireCitations) {
    $args += "--no-require-citations"
}

& goldevidencebench @args
exit $LASTEXITCODE
