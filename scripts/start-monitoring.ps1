<#
.SYNOPSIS
    Start (or stop) the Prometheus + Grafana monitoring stack.

.DESCRIPTION
    On this machine Docker runs *inside WSL2* (not Docker Desktop), while the
    telemetry exporter and inference server run natively on Windows. A container
    in WSL2 can't use the name "host.docker.internal" to reach the Windows host,
    so this script detects the Windows host IP as seen from WSL and injects it
    via the WIN_HOST_IP env var (the compose file maps host.docker.internal to it,
    falling back to host-gateway on Docker Desktop).

.EXAMPLE
    .\scripts\start-monitoring.ps1            # bring the stack up
    .\scripts\start-monitoring.ps1 -Down      # tear the stack down
#>
param([switch]$Down)

$ErrorActionPreference = "Stop"

# Repo's docker/ dir -> WSL /mnt path (works regardless of where the repo lives).
$dockerWin = Join-Path (Split-Path -Parent $PSScriptRoot) "docker"
$drive     = $dockerWin.Substring(0, 1).ToLower()
$dockerWsl = "/mnt/$drive" + ($dockerWin.Substring(2) -replace '\\', '/')

if ($Down) {
    Write-Host "Stopping monitoring stack..." -ForegroundColor Yellow
    wsl -e bash -lc "cd '$dockerWsl' && docker compose down"
    return
}

# Windows host IP as seen from inside WSL (the eth0 default-route gateway).
$winHostIp = (((wsl -e bash -lc "ip route show default") -join " ").Trim() -split "\s+")[2]
Write-Host "Windows host IP (from WSL): $winHostIp" -ForegroundColor Cyan

wsl -e bash -lc "cd '$dockerWsl' && WIN_HOST_IP=$winHostIp docker compose up -d"

$vmip = (wsl -e bash -lc "hostname -I").Trim().Split()[0]

Write-Host ""
Write-Host "Stack is up." -ForegroundColor Green
Write-Host "  Grafana    -> http://localhost:3001   (anonymous admin; dashboard auto-loaded)"
Write-Host "  Prometheus -> http://localhost:9090"
Write-Host ""
Write-Host "If localhost 'refuses' (WSL idle-suspended its port-forward), use the WSL VM IP:" -ForegroundColor Yellow
Write-Host "  Grafana    -> http://${vmip}:3001" -ForegroundColor Yellow
Write-Host "  Prometheus -> http://${vmip}:9090" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run the exporter (:9100) and server (:8000) on Windows for full data." -ForegroundColor DarkGray
