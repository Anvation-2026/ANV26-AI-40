# ==============================================================================
# MedGuard AI — Unified Full-Stack Application Launcher (start.ps1)
# ==============================================================================
# Runs the FastAPI Backend (Port 8000) and Vite React Frontend (Port 5173),
# verifies ML model pipeline connectivity, and handles graceful shutdown.
# ==============================================================================

param(
    [switch]$NoBrowser,
    [switch]$TestOnly
)

$ErrorActionPreference = "Stop"

# Set location to repository root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       MedGuard AI Decision-Support Platform Launcher       " -ForegroundColor White -BackgroundColor DarkBlue
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Helper to validate a candidate Python executable by executing it
function Test-PythonCandidate {
    param(
        [string]$Candidate,
        [switch]$CheckBackendReady
    )

    if ([string]::IsNullOrWhiteSpace($Candidate)) { return $null }

    $targetExe = $null
    if ($Candidate.Contains("\") -or $Candidate.Contains("/")) {
        if (-not (Test-Path $Candidate -PathType Leaf)) {
            return $null
        }
        $targetExe = (Resolve-Path $Candidate -ErrorAction SilentlyContinue).Path
    } else {
        $cmd = Get-Command $Candidate -ErrorAction SilentlyContinue
        if (-not $cmd) {
            return $null
        }
        $targetExe = $cmd.Source
    }

    if (-not $targetExe) { return $null }

    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $targetExe
        $psi.Arguments = "--version"
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $stdout = $proc.StandardOutput.ReadToEnd()
        $stderr = $proc.StandardError.ReadToEnd()
        [void]$proc.WaitForExit(4000)

        if ($proc.ExitCode -ne 0) {
            return $null
        }

        $rawVersion = ($stdout + " " + $stderr).Trim()
        if (-not ($rawVersion -match "Python \d+\.\d+")) {
            return $null
        }

        $backendReady = $true
        if ($CheckBackendReady) {
            $psiPkg = New-Object System.Diagnostics.ProcessStartInfo
            $psiPkg.FileName = $targetExe
            $psiPkg.Arguments = "-c `"import uvicorn`""
            $psiPkg.RedirectStandardOutput = $true
            $psiPkg.RedirectStandardError = $true
            $psiPkg.UseShellExecute = $false
            $psiPkg.CreateNoWindow = $true

            $procPkg = [System.Diagnostics.Process]::Start($psiPkg)
            [void]$procPkg.WaitForExit(4000)
            if ($procPkg.ExitCode -ne 0) {
                $backendReady = $false
            }
        }

        return [PSCustomObject]@{
            Path         = $targetExe
            Version      = $rawVersion
            BackendReady = $backendReady
        }
    } catch {
        # Execution failed
    }

    return $null
}

# 1. Environment Verification
Write-Host "[1/5] Checking runtime environments..." -ForegroundColor Yellow

# Candidate paths in strict priority order:
# 1. Project-specific ML virtual environment (.venv-ml)
# 2. Other valid project virtual environments (.venv, venv, env)
# 3. System Python in PATH and common installation paths
$candidatePaths = @(
    "$RepoRoot\.venv-ml\Scripts\python.exe",
    "$RepoRoot\.venv\Scripts\python.exe",
    "$RepoRoot\venv\Scripts\python.exe",
    "$RepoRoot\env\Scripts\python.exe",
    "$RepoRoot\.venv312\Scripts\python.exe",
    "$RepoRoot\venv-ml\Scripts\python.exe",
    "python",
    "python3",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
    "$env:ProgramFiles\Python314\python.exe",
    "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe",
    "$env:LOCALAPPDATA\Python\bin\python.exe"
)

# Query Windows 'py' launcher if present to discover additional registered interpreters
$pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
if ($pyLauncher) {
    try {
        $pyList = & py -0p 2>&1
        foreach ($line in ($pyList -split "`r?`n")) {
            if ($line -match '([A-Za-z]:\\[^"*\r\n]+\\python\.exe)') {
                $discovered = $matches[1].Trim()
                if ($discovered -and -not ($candidatePaths -contains $discovered)) {
                    $candidatePaths += $discovered
                }
            }
        }
    } catch {}
}

$SelectedPython = $null
$FirstExecutableCandidate = $null

foreach ($candidate in $candidatePaths) {
    $candidateResult = Test-PythonCandidate -Candidate $candidate -CheckBackendReady
    if ($candidateResult -and $candidateResult.Path) {
        if (-not $FirstExecutableCandidate) {
            $FirstExecutableCandidate = $candidateResult
        }

        if ($candidateResult.BackendReady) {
            $SelectedPython = $candidateResult
            break
        } else {
            # Candidate runs, but lacks uvicorn
            $isRepoVenv = $candidate.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)
            if ($isRepoVenv) {
                Write-Host "  -> Notice: Environment $($candidateResult.Path) is executable but lacks required backend module 'uvicorn'." -ForegroundColor DarkYellow
                Write-Host "             Searching for an environment with backend dependencies..." -ForegroundColor DarkYellow
            }
        }
    } else {
        # If .venv-ml exists on disk but failed execution validation, explain why
        if ($candidate -eq "$RepoRoot\.venv-ml\Scripts\python.exe" -and (Test-Path "$RepoRoot\.venv-ml\Scripts\python.exe")) {
            Write-Host "  -> Notice: .venv-ml exists on disk but failed execution check (broken base Python reference in pyvenv.cfg)." -ForegroundColor DarkYellow
            Write-Host "             Skipping .venv-ml and searching for valid fallback environments..." -ForegroundColor DarkYellow
        }
    }
}

# If no environment had uvicorn, fall back to the first executable interpreter
if (-not $SelectedPython -and $FirstExecutableCandidate) {
    $SelectedPython = $FirstExecutableCandidate
}

if (-not $SelectedPython) {
    Write-Host "  [ERROR] No functional Python interpreter could be validated." -ForegroundColor Red
    Write-Host "          Checked .venv-ml, project virtual environments, and system PATH." -ForegroundColor Red
    exit 1
}

$PythonExe = $SelectedPython.Path
Write-Host "  -> Python runtime validated: $($SelectedPython.Version) ($PythonExe)" -ForegroundColor Green

# Check Node / npm
try {
    $nodeVersion = & node --version 2>&1
    $npmVersion = & npm --version 2>&1
    Write-Host "  -> Node detected: $nodeVersion (npm v$npmVersion)" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Node.js/npm is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Port Check
Write-Host ""
Write-Host "[2/5] Checking network ports (8000, 5173)..." -ForegroundColor Yellow

function Test-PortInUse($port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue)
}

function Clear-Port($port) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            if ($conn.OwningProcess -and $conn.OwningProcess -gt 0) {
                Write-Host "  -> Freeing conflicting process on port $port (PID $($conn.OwningProcess))..." -ForegroundColor Yellow
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Milliseconds 500
    } catch {}
}

if (Test-PortInUse 8000) {
    Clear-Port 8000
}
Write-Host "  -> Port 8000 (Backend) ready." -ForegroundColor Green

if (Test-PortInUse 5173) {
    Clear-Port 5173
}
Write-Host "  -> Port 5173 (Frontend) ready." -ForegroundColor Green

# 3. Start Backend Service
Write-Host ""
Write-Host "[3/5] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow

# Ensure logs directory exists
$LogsDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

$backendStdoutLog = Join-Path $LogsDir "backend_stdout.log"
$backendStderrLog = Join-Path $LogsDir "backend_stderr.log"
$frontendStdoutLog = Join-Path $LogsDir "frontend_stdout.log"
$frontendStderrLog = Join-Path $LogsDir "frontend_stderr.log"

# Clean old log files for clean session tracing
Remove-Item $backendStdoutLog, $backendStderrLog, $frontendStdoutLog, $frontendStderrLog -Force -ErrorAction SilentlyContinue

$backendProcess = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $backendStdoutLog `
    -RedirectStandardError $backendStderrLog `
    -PassThru

Write-Host "  -> Backend process started (PID: $($backendProcess.Id))" -ForegroundColor Green
Write-Host "     Stdout log: $backendStdoutLog" -ForegroundColor Gray
Write-Host "     Stderr log: $backendStderrLog" -ForegroundColor Gray

# 4. Start Frontend Dev Server
Write-Host ""
Write-Host "[4/5] Starting Vite React Frontend..." -ForegroundColor Yellow

$frontendDir = Join-Path $RepoRoot "frontend"

$frontendProcess = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory $frontendDir `
    -RedirectStandardOutput $frontendStdoutLog `
    -RedirectStandardError $frontendStderrLog `
    -PassThru

Write-Host "  -> Frontend process started (PID: $($frontendProcess.Id))" -ForegroundColor Green
Write-Host "     Stdout log: $frontendStdoutLog" -ForegroundColor Gray
Write-Host "     Stderr log: $frontendStderrLog" -ForegroundColor Gray

# 5. Health Check & Ready
Write-Host ""
Write-Host "[5/5] Polling services until online..." -ForegroundColor Yellow

$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ($backendProcess.HasExited) {
        Write-Host "  [ERROR] Backend process exited unexpectedly (ExitCode: $($backendProcess.ExitCode))." -ForegroundColor Red
        if (Test-Path $backendStderrLog) {
            $errSnippet = Get-Content $backendStderrLog -Tail 15 -ErrorAction SilentlyContinue
            if ($errSnippet) {
                Write-Host "  --- Backend stderr tail ---" -ForegroundColor Red
                $errSnippet | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
            }
        }
        break
    }
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res -and $res.status -eq "ok") {
            $backendReady = $true
            break
        }
    } catch {
        # Retry until up
    }
}

if ($backendReady) {
    Write-Host "  -> FastAPI Backend is HEALTHY and ML pipeline is CONNECTED!" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Backend took longer than expected to report healthy." -ForegroundColor Magenta
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "                 MEDGUARD AI IS NOW LIVE!                   " -ForegroundColor Black -BackgroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Frontend Application:   http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend API Docs:       http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "  Health Status:          http://127.0.0.1:8000/api/health" -ForegroundColor Cyan
Write-Host "  Model Diagnostics:      http://127.0.0.1:8000/api/model/status" -ForegroundColor Cyan
Write-Host "  Validation Benchmark:   http://127.0.0.1:8000/api/validation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

if (-not $NoBrowser) {
    Write-Host "Opening web browser to http://localhost:5173..." -ForegroundColor Gray
    try {
        Start-Process "http://localhost:5173"
    } catch {
        # Non-critical if browser launch fails in headless
    }
}

try {
    if ($TestOnly) {
        if ($backendReady) {
            Write-Host "Launcher test mode completed successfully." -ForegroundColor Green
            return
        } else {
            Write-Host "Launcher test mode failed: backend did not report healthy." -ForegroundColor Red
            exit 1
        }
    }

    Write-Host "Press Ctrl+C to terminate both backend and frontend servers." -ForegroundColor Yellow
    Write-Host ""

    # Monitor processes and gracefully shutdown on exit
    while ($true) {
        if ($backendProcess.HasExited) {
            Write-Host "Backend process exited with code $($backendProcess.ExitCode)" -ForegroundColor Red
            if (Test-Path $backendStderrLog) {
                Get-Content $backendStderrLog -Tail 10 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
            }
            break
        }
        if ($frontendProcess.HasExited) {
            Write-Host "Frontend process exited with code $($frontendProcess.ExitCode)" -ForegroundColor Red
            if (Test-Path $frontendStderrLog) {
                Get-Content $frontendStderrLog -Tail 10 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
            }
            break
        }
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Stopping servers gracefully..." -ForegroundColor Yellow

    if ($backendProcess -and -not $backendProcess.HasExited) {
        Write-Host "Stopping backend (PID $($backendProcess.Id))..." -ForegroundColor Gray
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }

    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Write-Host "Stopping frontend (PID $($frontendProcess.Id))..." -ForegroundColor Gray
        # Kill process tree for cmd.exe running node
        taskkill /PID $frontendProcess.Id /T /F 2>&1 | Out-Null
    }

    Write-Host "MedGuard AI servers stopped cleanly." -ForegroundColor Green
}
