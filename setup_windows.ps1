param(
    [ValidateSet("gpu", "cpu")]
    [string]$Mode = "gpu",
    [string]$PythonExe = "",
    [string]$VenvDir = ".venv",
    [switch]$IncludeAppDependencies,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

function Get-PythonInfo {
    param([string]$Executable, [string[]]$PrefixArgs = @())
    try {
        $output = & $Executable @PrefixArgs -c 'import json,sys; print(json.dumps({"executable":sys.executable,"version":list(sys.version_info[:3]),"prefix":sys.prefix,"base_prefix":sys.base_prefix}))' 2>$null
        if ($LASTEXITCODE -eq 0) { return ($output | ConvertFrom-Json) }
    } catch { }
    return $null
}

function Test-SupportedPython {
    param($Info)
    return ($null -ne $Info -and $Info.version[0] -eq 3 -and $Info.version[1] -in @(10, 11))
}

function Resolve-ProjectPython {
    if ($PythonExe) {
        $info = Get-PythonInfo -Executable $PythonExe
        if (-not (Test-SupportedPython $info)) {
            throw "-PythonExe must point to a working Python 3.10 or 3.11 interpreter."
        }
        return $info
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("-3.11", "-3.10")) {
            $info = Get-PythonInfo -Executable "py" -PrefixArgs @($version)
            if (Test-SupportedPython $info) { return $info }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $info = Get-PythonInfo -Executable "python"
        if (Test-SupportedPython $info) { return $info }
    }
    throw "No supported Python found. Pass -PythonExe with a Python 3.10/3.11 executable; no system Python will be installed or changed."
}

$venvPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $VenvDir))
$rootPrefix = $projectRoot.TrimEnd('\') + '\'
if (-not $venvPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "VenvDir must be a directory inside this repository."
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPath) {
    $pythonInfo = Get-PythonInfo -Executable $venvPython
    if (-not (Test-SupportedPython $pythonInfo) -or $pythonInfo.prefix -eq $pythonInfo.base_prefix) {
        throw "The existing environment is invalid or uses an unsupported Python. Select a new -VenvDir; existing files are preserved."
    }
    if ($PythonExe) {
        $requested = Resolve-ProjectPython
        if ($requested.version[0] -ne $pythonInfo.version[0] -or $requested.version[1] -ne $pythonInfo.version[1]) {
            throw "The existing venv has a different Python version; select a new -VenvDir."
        }
    }
} else {
    $pythonInfo = Resolve-ProjectPython
}

$flavor = if ($Mode -eq "gpu") { "cu128" } else { "cpu" }
$requirements = if ($IncludeAppDependencies) { "requirements-windows-$Mode.txt" } else { "requirements-detector.txt" }
Write-Host "Python $($pythonInfo.version -join '.') | $Mode | $requirements"
Write-Host "Environment: $venvPath"
if ($CheckOnly) {
    Write-Host "Preflight passed; no environment or packages changed."
    return
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPath)) {
        Invoke-Checked $pythonInfo.executable @("-m", "venv", $venvPath)
    }
    Invoke-Checked $venvPython @("-m", "pip", "--version")
    Invoke-Checked $venvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-Checked $venvPython @("-m", "pip", "install", "torch==2.9.1+$flavor", "torchvision==0.24.1+$flavor", "--index-url", "https://download.pytorch.org/whl/$flavor")
    Invoke-Checked $venvPython @("-m", "pip", "install", "-r", $requirements)
    Invoke-Checked $venvPython @("-m", "pip", "install", "-e", ".", "-c", "requirements-detector.txt")
    Invoke-Checked $venvPython @("-m", "pip", "check")
    $device = if ($Mode -eq "gpu") { "cuda" } else { "cpu" }
    Invoke-Checked $venvPython @("tools/verify_environment.py", "--device", $device)
    Write-Host "Detector environment verified. Interpreter: $venvPython"
} finally {
    Pop-Location
}
