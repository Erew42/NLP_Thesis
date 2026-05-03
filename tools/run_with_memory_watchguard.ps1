param(
  [Parameter(Mandatory = $true)]
  [string]$FilePath,

  [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
  [string[]]$ArgumentList,

  [Parameter(Mandatory = $true)]
  [string]$OutputDirectory,

  [string]$WorkingDirectory = (Get-Location).Path,

  [double]$PrivateLimitGb = 8.0,

  [double]$AvailableLimitMb = 500.0,

  [int]$PollSeconds = 5
)

$ErrorActionPreference = 'Stop'

function ConvertTo-WindowsProcessArgument {
  param(
    [AllowNull()]
    [string]$Argument
  )

  if ($null -eq $Argument -or $Argument.Length -eq 0) {
    return '""'
  }
  if ($Argument -notmatch '[\s"]') {
    return $Argument
  }

  $result = '"'
  $backslashes = 0
  foreach ($char in $Argument.ToCharArray()) {
    if ($char -eq '\') {
      $backslashes += 1
      continue
    }
    if ($char -eq '"') {
      $result += ('\' * (($backslashes * 2) + 1))
      $result += '"'
      $backslashes = 0
      continue
    }
    if ($backslashes -gt 0) {
      $result += ('\' * $backslashes)
      $backslashes = 0
    }
    $result += $char
  }
  if ($backslashes -gt 0) {
    $result += ('\' * ($backslashes * 2))
  }
  $result += '"'
  return $result
}

if ($PollSeconds -lt 1) {
  throw "PollSeconds must be at least 1."
}

$outputDir = [System.IO.Path]::GetFullPath($OutputDirectory)
$workDir = [System.IO.Path]::GetFullPath($WorkingDirectory)
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$statusPath = Join-Path $outputDir 'watchdog_status.json'
$monitorPath = Join-Path $outputDir 'memory_usage_log.csv'
$stdoutPath = Join-Path $outputDir 'worker_stdout.log'
$stderrPath = Join-Path $outputDir 'worker_stderr.log'
$commandPath = Join-Path $outputDir 'watchdog_command.json'

foreach ($path in @($statusPath, $monitorPath, $stdoutPath, $stderrPath, $commandPath)) {
  if (Test-Path -LiteralPath $path) {
    Remove-Item -LiteralPath $path -Force
  }
}

$startedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
$command = [ordered]@{
  file_path = $FilePath
  argument_list = $ArgumentList
  working_directory = $workDir
  output_directory = $outputDir
  private_limit_gb = $PrivateLimitGb
  available_limit_mb = $AvailableLimitMb
  poll_seconds = $PollSeconds
  started_at_utc = $startedAtUtc
}
$command | ConvertTo-Json -Depth 5 | Set-Content -Path $commandPath -Encoding utf8

'utc,pid,private_gb,working_set_gb,available_mb,cpu_seconds,handles,event' |
  Set-Content -Path $monitorPath -Encoding utf8

$proc = $null
$event = 'started'
$peakPrivateGb = 0.0
$peakWorkingSetGb = 0.0
$minAvailableMb = [double]::PositiveInfinity
$sampleCount = 0
$stdoutTask = $null
$stderrTask = $null

try {
  $argumentString = ($ArgumentList | ForEach-Object { ConvertTo-WindowsProcessArgument $_ }) -join ' '

  $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FilePath
  $startInfo.Arguments = $argumentString
  $startInfo.WorkingDirectory = $workDir
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true
  $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

  $proc = [System.Diagnostics.Process]::new()
  $proc.StartInfo = $startInfo
  [void]$proc.Start()
  $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
  $stderrTask = $proc.StandardError.ReadToEndAsync()

  while (-not $proc.HasExited) {
    Start-Sleep -Seconds $PollSeconds
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $p) {
      break
    }

    $availableMb = $null
    try {
      $os = Get-CimInstance Win32_OperatingSystem
      $availableMb = [double]$os.FreePhysicalMemory / 1024.0
    } catch {
      $availableMb = $null
    }
    $privateGb = [double]$p.PrivateMemorySize64 / 1GB
    $workingGb = [double]$p.WorkingSet64 / 1GB
    $peakPrivateGb = [math]::Max($peakPrivateGb, $privateGb)
    $peakWorkingSetGb = [math]::Max($peakWorkingSetGb, $workingGb)
    if ($null -ne $availableMb) {
      $minAvailableMb = [math]::Min($minAvailableMb, $availableMb)
    }
    $sampleCount += 1

    $line = ('{0},{1},{2:N3},{3:N3},{4:N1},{5:N3},{6},{7}' -f `
      (Get-Date).ToUniversalTime().ToString('o'), `
      $p.Id, `
      $privateGb, `
      $workingGb, `
      $availableMb, `
      $p.CPU, `
      $p.HandleCount, `
      'running')
    Add-Content -Path $monitorPath -Value $line -Encoding utf8

    if ($privateGb -gt $PrivateLimitGb) {
      $event = 'killed_private_limit'
      Stop-Process -Id $p.Id -Force
      break
    }
    if ($null -ne $availableMb -and $availableMb -lt $AvailableLimitMb) {
      $event = 'killed_available_limit'
      Stop-Process -Id $p.Id -Force
      break
    }
  }

  $proc.Refresh()
  $proc.WaitForExit()
  if ($stdoutTask) {
    $stdoutTask.Wait()
    [System.IO.File]::WriteAllText($stdoutPath, [string]$stdoutTask.Result, [System.Text.UTF8Encoding]::new($false))
  }
  if ($stderrTask) {
    $stderrTask.Wait()
    [System.IO.File]::WriteAllText($stderrPath, [string]$stderrTask.Result, [System.Text.UTF8Encoding]::new($false))
  }

  if ($event -eq 'started') {
    $event = if ($proc.ExitCode -eq 0) { 'completed' } else { 'exited_nonzero' }
  }

  $completedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
  $minAvailableOut = if ([double]::IsPositiveInfinity($minAvailableMb)) { $null } else { [math]::Round($minAvailableMb, 1) }
  $status = [ordered]@{
    status = $event
    killed_for_memory = $event -in @('killed_private_limit', 'killed_available_limit')
    exit_code = $proc.ExitCode
    pid = $proc.Id
    sample_count = $sampleCount
    started_at_utc = $startedAtUtc
    completed_at_utc = $completedAtUtc
    peak_private_gb = [math]::Round($peakPrivateGb, 3)
    peak_working_set_gb = [math]::Round($peakWorkingSetGb, 3)
    min_available_mb = $minAvailableOut
    private_limit_gb = $PrivateLimitGb
    available_limit_mb = $AvailableLimitMb
    memory_log = $monitorPath
    worker_stdout_log = $stdoutPath
    worker_stderr_log = $stderrPath
    command_log = $commandPath
  }
  $status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding utf8

  if ($event -eq 'killed_private_limit' -or $event -eq 'killed_available_limit') {
    exit 137
  }
  if ($proc.ExitCode -ne 0) {
    exit $proc.ExitCode
  }
  exit 0
} catch {
  $completedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
  $status = [ordered]@{
    status = 'watchdog_error'
    error = $_.Exception.Message
    pid = if ($proc) { $proc.Id } else { $null }
    started_at_utc = $startedAtUtc
    completed_at_utc = $completedAtUtc
    memory_log = $monitorPath
    worker_stdout_log = $stdoutPath
    worker_stderr_log = $stderrPath
    command_log = $commandPath
  }
  $status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding utf8
  throw
} finally {
  if ($proc -and -not $proc.HasExited) {
    try {
      $proc.Kill()
    } catch {
    }
  }
}
