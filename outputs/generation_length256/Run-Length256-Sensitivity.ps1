param([int]$Limit = 0)
$ErrorActionPreference = 'Stop'
$generationPython = 'D:\research_pROJECT\.venv-generation\Scripts\python.exe'
$projectDirectory = 'D:\research_pROJECT\longmemeval_retrieval_pilot\longmemeval_retrieval_pilot'
$lengthLauncher = Join-Path $PSScriptRoot 'run_generation_cudnn_length256.py'
$selectedInputs = Join-Path $PSScriptRoot 'length_sensitivity_236_inputs.csv'
$resultFile = 'outputs/generation_length256/generation_limit236_256.csv'
foreach ($requiredFile in @($generationPython, $lengthLauncher, $selectedInputs)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file missing: $requiredFile"
    }
}
Push-Location -LiteralPath $projectDirectory
try {
    $lengthArgs = @(
        '--pilot-file', 'outputs/confirmatory_200/pilot_questions.json',
        '--retrieval-results', $selectedInputs,
        '--output', $resultFile,
        '--model-name', 'meta-llama/Llama-3.1-8B-Instruct',
        '--model-revision', '0e9e39f249a16976918f6564b8830bc894c89659',
        '--quantization', '4bit',
        '--max-new-tokens', '256',
        '--seed', '42'
    )
    # Baselines are already present among the 236 selected input rows.
    if (Test-Path -LiteralPath $resultFile) { $lengthArgs += '--resume' }
    if ($Limit -gt 0) { $lengthArgs += @('--limit', "$Limit") }
    & $generationPython $lengthLauncher @lengthArgs
    if ($LASTEXITCODE -ne 0) { throw "Generation stopped with exit code $LASTEXITCODE; inspect the message above." }
}
finally { Pop-Location }
