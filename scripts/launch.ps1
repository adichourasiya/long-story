#!/usr/bin/env pwsh
<#
launch.ps1

Activates the project's .venv if not already active, installs dependencies
using `uv` when available (falls back to pip/pyproject/requirements), and
launches the interactive program `novel_ai.py`.

Usage (PowerShell):
  .\scripts\launch.ps1
#>

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir '..')

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Success($msg) { Write-Host $msg -ForegroundColor Green }
function Write-ErrorAndExit($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

# 1) Activate .venv if not already
$venvActivate = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path $venvActivate) {
        Write-Info ("Activating virtual environment from {0}" -f $venvActivate)
        try {
            . $venvActivate
            if (-not $?) { Write-Warning 'Activation script returned non-zero; continuing.' }
        } catch {
            Write-Warning ("Failed to run activation script: {0}" -f $_)
        }
    } else {
        Write-Warning (".venv not found at {0} - continuing without activation." -f $venvActivate)
    }
} else {
    Write-Info ("Virtual environment already active: {0}" -f $env:VIRTUAL_ENV)
}

# Ensure `python` points to the expected interpreter in the venv when possible
function Run-Python {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]
        $CmdArgs
    )
    & python @CmdArgs
    return $LASTEXITCODE
}

# 2) Ensure pip exists (try python -m pip, ensurepip, or get-pip.py)
Write-Info 'Ensuring pip is available...'
 $rc = Run-Python '-m' 'pip' '--version'
if ($rc -ne 0) {
    Write-Info 'Bootstrapping pip via ensurepip (if available)...'
    $rc = Run-Python '-m' 'ensurepip' '--upgrade'
    if ($rc -ne 0) {
        Write-Info 'ensurepip failed or not available; attempting get-pip.py download...'
        try {
            $tmp = Join-Path $env:TEMP ([System.Guid]::NewGuid().ToString() + '.py')
            Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $tmp -UseBasicParsing
            $rc = Run-Python $tmp
            Remove-Item $tmp -ErrorAction SilentlyContinue
            if ($rc -ne 0) { Write-Warning ("get-pip.py install failed (rc={0})" -f $rc) }
        } catch {
            Write-Warning ("Failed to download or run get-pip.py: {0}" -f $_)
        }
    }
}

# 3) Install dependencies: prefer `uv` if present, otherwise fall back to pip
Write-Info 'Installing project dependencies...'
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    Write-Info ("Found 'uv' CLI at {0}. Running 'uv install'" -f $($uvCmd.Source))
    $rc = & uv install
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("'uv install' failed with exit code {0} - falling back to pip." -f $LASTEXITCODE)
    } else {
        Write-Success 'Dependencies installed via uv.'
    }
}

function Test-PackageInstalled([string]$pkg) {
    & python -c ("import {0}" -f $pkg)
    return $LASTEXITCODE
}

if (-not $uvCmd -or $LASTEXITCODE -ne 0) {
    # Prefer pyproject.toml editable install when available
    $pyproject = Join-Path $projectRoot 'pyproject.toml'
    $reqs = Join-Path $projectRoot 'requirements.txt'
    if (Test-Path $pyproject) {
        Write-Info "Installing package in-place from pyproject.toml (editable)."
        $rc = Run-Python '-m' 'pip' 'install' '-e' '.'
        if ($rc -ne 0) { Write-Warning "pip editable install failed (rc=$rc)" }
    } elseif (Test-Path $reqs) {
        Write-Info "Installing from requirements.txt"
        $rc = Run-Python '-m' 'pip' 'install' '-r' $reqs
        if ($rc -ne 0) { Write-Warning "pip install -r requirements.txt failed (rc=$rc)" }
    } else {
        Write-Info 'No pyproject.toml or requirements.txt found - skipping dependency install.'
    }
}

# After attempting installation, ensure prompt_toolkit is available; prefer using uv to install it if missing
Write-Info 'Verifying prompt_toolkit is installed...'
$pkgRc = Test-PackageInstalled 'prompt_toolkit'
if ($pkgRc -ne 0) {
    if ($uvCmd) {
        Write-Info "Attempting to install 'prompt_toolkit' with 'uv run'..."
        try {
            & uv run python -m pip install prompt_toolkit
        } catch {
            Write-Warning ("'uv run' attempt failed: {0}" -f $_)
        }
        $pkgRc = Test-PackageInstalled 'prompt_toolkit'
    }

    if ($pkgRc -ne 0) {
        Write-Info "Falling back to pip to install prompt_toolkit..."
        $rc = Run-Python '-m' 'pip' 'install' 'prompt_toolkit'
        $pkgRc = Test-PackageInstalled 'prompt_toolkit'
    }

    if ($pkgRc -ne 0) {
        Write-ErrorAndExit "prompt_toolkit is required for the interactive CLI but could not be installed. Please install it manually: pip install prompt_toolkit"
    } else {
        Write-Success "prompt_toolkit is installed."
    }
} else {
    Write-Success "prompt_toolkit is already installed."
}

# 4) Launch the program
Write-Info 'Launching interactive CLI: python novel_ai.py'
try {
    & python (Join-Path $projectRoot 'novel_ai.py')
    exit $LASTEXITCODE
} catch {
    Write-ErrorAndExit ("Failed to launch novel_ai.py: {0}" -f $_)
}
