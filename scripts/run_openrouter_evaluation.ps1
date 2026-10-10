param(
    [int]$Limit = 1,
    [switch]$Resume,
    [switch]$DryRun,
    [string]$Python = "python",
    [string]$Model = "openai/gpt-4o-2024-08-06",
    [double]$RequestDelaySeconds = 3.2
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$HypothesesDir = Join-Path $RepoRoot "outputs\hypotheses_pending"
$GenerationCsv = Join-Path $RepoRoot "outputs\generation_local\generation_all_conditions.csv"
$ReferenceFile = Join-Path $RepoRoot "outputs\confirmatory_200\pilot_questions.json"
$EvaluationDir = Join-Path $RepoRoot "outputs\evaluation_openrouter"
$EvaluationLog = Join-Path $EvaluationDir "gpt4o_2024_08_06_all_conditions.jsonl"

if (-not $env:OPENROUTER_API_KEY -and -not $DryRun) {
    throw "OPENROUTER_API_KEY is not set. Set it in this PowerShell session; never put it in the repository."
}

Push-Location $RepoRoot
try {
    & $Python longmemeval_eval.py export `
        --input $GenerationCsv `
        --output $HypothesesDir `
        --split-by-condition
    if ($LASTEXITCODE -ne 0) { throw "Hypothesis export failed." }

    $EvalArgs = @(
        "scripts/evaluate_qa_openrouter.py", "evaluate",
        "--hypotheses", $HypothesesDir,
        "--references", $ReferenceFile,
        "--output", $EvaluationLog,
        "--model", $Model,
        "--limit", $Limit,
        "--request-delay-seconds", $RequestDelaySeconds
    )
    if ($Resume) { $EvalArgs += "--resume" }
    if ($DryRun) { $EvalArgs += "--dry-run" }
    & $Python @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw "OpenRouter evaluation failed." }

    if (-not $DryRun) {
        if ($Limit -eq 0) {
            $ScoredCsv = Join-Path $RepoRoot "outputs\generation_local\generation_all_conditions_scored_openrouter.csv"
            & $Python longmemeval_eval.py merge `
                --generation-results $GenerationCsv `
                --evaluation-log $EvaluationLog `
                --output $ScoredCsv
        } else {
            $ScoredCsv = Join-Path $EvaluationDir "generation_scored_openrouter_partial.csv"
            & $Python longmemeval_eval.py merge `
                --generation-results $GenerationCsv `
                --evaluation-log $EvaluationLog `
                --output $ScoredCsv `
                --allow-partial
        }
        if ($LASTEXITCODE -ne 0) { throw "Score merge failed." }
        Write-Host "Scored CSV: $ScoredCsv"
    }
} finally {
    Pop-Location
}
