# One-click start: memory DB + API (Docker), local speech server, then the voice agent console.
# Usage (from this folder):  .\start.ps1
# Needs Docker Desktop running and Ollama installed. Everything stays on this machine.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$memoryDir = Join-Path (Split-Path $root -Parent) 'memory-service'

# via cmd: in Windows PowerShell 5.1 a native command's stderr becomes a terminating error under 'Stop'
function Test-Docker { cmd /c 'docker info >nul 2>&1'; return ($LASTEXITCODE -eq 0) }

function Test-Url($url) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $url | Out-Null; return $true } catch { return $false }
}

Write-Host '1/4 memory-service (Postgres + API on :8000)'
if (-not (Test-Docker)) {
    $dd = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'
    if (-not (Test-Path $dd)) { $dd = 'C:\Program Files\Docker\Docker\Docker Desktop.exe' }
    Write-Host '    starting Docker Desktop and waiting for it (up to 2 min)'
    # via WMI, not Start-Process: a child of this terminal tab dies when the tab is closed,
    # and Docker Desktop should outlive the agent
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$dd`"" } | Out-Null
    $deadline = (Get-Date).AddSeconds(120)
    do {
        if ((Get-Date) -gt $deadline) { throw 'Docker Desktop did not start in 120 s' }
        Start-Sleep -Seconds 3
    } while (-not (Test-Docker))
}
docker compose -f (Join-Path $memoryDir 'docker-compose.yml') up -d db api
if ($LASTEXITCODE -ne 0) { throw 'docker compose failed - is Docker Desktop running?' }

Write-Host '2/4 speech server (Whisper + Kokoro on :8880)'
if (-not (Test-Url 'http://127.0.0.1:8880/health')) {
    $servers = Join-Path $root 'servers'
    Start-Process -FilePath (Join-Path $servers '.venv\Scripts\python.exe') -ArgumentList 'speech_server.py' `
        -WorkingDirectory $servers -WindowStyle Minimized
}

Write-Host '3/4 waiting for services (first start loads models, ~30 s)'
$deadline = (Get-Date).AddSeconds(120)
while (-not ((Test-Url 'http://127.0.0.1:8880/health') -and (Test-Url 'http://127.0.0.1:8000/health'))) {
    if ((Get-Date) -gt $deadline) { throw 'services did not come up in 120 s' }
    Start-Sleep -Seconds 2
}

Write-Host '4/4 voice agent console - talk into your mic, Ctrl+C to quit'
Set-Location $root
uv run agent.py console
