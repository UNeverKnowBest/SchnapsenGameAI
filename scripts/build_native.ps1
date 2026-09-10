param([string]$Python = ".venv/Scripts/python.exe")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$savedEnvironment = @{}
foreach ($key in @("RUSTUP_HOME", "CARGO_HOME", "PATH", "PYO3_PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX")) {
    $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}
Push-Location $projectRoot
try {
    if (Test-Path ".tools/rustup") {
        $env:RUSTUP_HOME = Join-Path $projectRoot ".tools/rustup"
        $env:CARGO_HOME = Join-Path $projectRoot ".tools/cargo"
        $env:PATH = "$projectRoot/.tools/cargo/bin;$env:PATH"
    }
    if (Test-Path "C:/msys64/ucrt64/bin/gcc.exe") {
        $env:PATH = "C:/msys64/ucrt64/bin;$env:PATH"
    }
    $env:PYO3_PYTHON = (Resolve-Path $Python).Path
    $env:VIRTUAL_ENV = Split-Path -Parent (Split-Path -Parent $env:PYO3_PYTHON)
    Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
    & $Python -m maturin develop --release --locked
    if ($LASTEXITCODE -ne 0) { throw "Native extension build failed" }
} finally {
    foreach ($key in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], "Process")
    }
    Pop-Location
}
