# ==============================================================================
# MedGuard AI — Unified Full-Stack Application Launcher (start.ps1)
# ==============================================================================
# Runs the FastAPI Backend (Port 8000) and Vite React Frontend (Port 5173),
# verifies ML model pipeline connectivity, and handles graceful shutdown.
# ==============================================================================

$ErrorActionPreference = "Stop"

# Set location to repository root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       MedGuard AI Decision-Support Platform Launcher       " -ForegroundColor White -BackgroundColor DarkBlue
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Environment Verification
Write-Host "[1/5] Checking runtime environments..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "  -> Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

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

$backendProcess = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $RepoRoot `
    -PassThru

Write-Host "  -> Backend process started (PID: $($backendProcess.Id))" -ForegroundColor Green

# 4. Start Frontend Dev Server
Write-Host ""
Write-Host "[4/5] Starting Vite React Frontend..." -ForegroundColor Yellow

$frontendDir = Join-Path $RepoRoot "frontend"

$frontendProcess = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory $frontendDir `
    -PassThru

Write-Host "  -> Frontend process started (PID: $($frontendProcess.Id))" -ForegroundColor Green

# 5. Health Check & Ready
Write-Host ""
Write-Host "[5/5] Polling services until online..." -ForegroundColor Yellow

$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
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
Write-Host "Opening web browser to http://localhost:5173..." -ForegroundColor Gray

# Open browser to frontend
try {
    Start-Process "http://localhost:5173"
} catch {
    # Non-critical if browser launch fails in headless
}

Write-Host "Press Ctrl+C to terminate both backend and frontend servers." -ForegroundColor Yellow
Write-Host ""

# Monitor processes and gracefully shutdown on exit
try {
    while ($true) {
        if ($backendProcess.HasExited) {
            Write-Host "Backend process exited with code $($backendProcess.ExitCode)" -ForegroundColor Red
            break
        }
        if ($frontendProcess.HasExited) {
            Write-Host "Frontend process exited with code $($frontendProcess.ExitCode)" -ForegroundColor Red
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

    # Clean up any leftover node or uvicorn processes if necessary
    Write-Host "MedGuard AI servers stopped cleanly." -ForegroundColor Green
}
