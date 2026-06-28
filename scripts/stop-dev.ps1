param(
    [int[]]$Ports = @(5000, 5173)
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$allProcesses = Get-CimInstance Win32_Process
$processIds = New-Object System.Collections.Generic.HashSet[int]

$connections = Get-NetTCPConnection -LocalPort $Ports -ErrorAction SilentlyContinue
foreach ($connection in $connections) {
    [void]$processIds.Add([int]$connection.OwningProcess)
}

$devPatterns = @(
    "nodemon src/index.js",
    "vite --host",
    "src/index.js",
    "server\node_modules",
    "client\node_modules",
    "concurrently"
)

foreach ($process in $allProcesses) {
    $commandLine = $process.CommandLine
    if (-not $commandLine) {
        continue
    }

    $isProjectProcess = $commandLine.Contains($projectRoot)
    $isDevProcess = $false
    foreach ($pattern in $devPatterns) {
        if ($commandLine.Contains($pattern)) {
            $isDevProcess = $true
            break
        }
    }

    if ($isProjectProcess -and $isDevProcess) {
        [void]$processIds.Add([int]$process.ProcessId)
    }
}

$changed = $true
while ($changed) {
    $changed = $false
    foreach ($process in $allProcesses) {
        if ($processIds.Contains([int]$process.ParentProcessId) -and -not $processIds.Contains([int]$process.ProcessId)) {
            [void]$processIds.Add([int]$process.ProcessId)
            $changed = $true
        }
    }
}

$currentPid = [System.Diagnostics.Process]::GetCurrentProcess().Id
$idsToStop = $processIds |
    Where-Object { $_ -ne $currentPid } |
    Sort-Object -Descending

foreach ($processId in $idsToStop) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Stopping project dev process $processId ($($process.ProcessName))"
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

if (-not $idsToStop) {
    Write-Host "No project dev processes are currently running on the configured ports."
}
