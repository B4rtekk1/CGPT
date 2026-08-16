[CmdletBinding()]
param(
    [string]$BuildDir,
    [string]$OutputMarkdown,
    [double]$PeakMemoryGBps = 192.0
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $scriptRoot '..\cmake-build-default-visual-studio\Release'
}
if ([string]::IsNullOrWhiteSpace($OutputMarkdown)) {
    $OutputMarkdown = Join-Path $scriptRoot '..\kernel_profile.md'
}
$buildPath = [System.IO.Path]::GetFullPath($BuildDir)
$runner = Join-Path $buildPath 'run_all_tests.exe'
$logPath = [System.IO.Path]::ChangeExtension($OutputMarkdown, '.log')

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Profiler runner not found: $runner"
}

$tests = @(
    'rmsnorm_test', 'linear_test', 'linear_backward_test', 'swiglu_test',
    'rope_test', 'tensor_test', 'device_buffer_test', 'dtype_test',
    'flash_attention_test', 'embedding_test', 'cross_entropy_test',
    'cuda_graph_test', 'cuda_benchmark_test', 'kv_cache_test',
    'transformer_block_test', 'transformer_model_test', 'adamw_test'
)

$arguments = [System.Collections.Generic.List[string]]::new()
foreach ($test in $tests) {
    $arguments.Add($test)
    $arguments.Add((Join-Path $buildPath ($test + '.exe')))
}

Write-Host "Starting profiler: $runner"
$rawOutput = @(& $runner @arguments 2>&1 | ForEach-Object { $_.ToString() })
$exitCode = $LASTEXITCODE
$rawOutput | Set-Content -LiteralPath $logPath -Encoding utf8

# The profiler prints ANSI highlighting when attached to a terminal.
$ansi = [char]27
$lines = $rawOutput | ForEach-Object {
    $_ -replace "$ansi\[[0-9;]*m", ''
}

# Minimum tensor traffic for each benchmark, in decimal megabytes.
$trafficMB = @{
    'RMSNorm kernel'                    = 8.3968
    'RMSNorm BF16 kernel'               = 8.3968
    'RMSNorm backward kernel'           = 12.60
    'RMSNorm BF16 backward kernel'      = 12.60
    'Linear kernel'                     = 41.943
    'SwiGLU kernel'                     = 12.583
    'RoPE kernel'                       = 10.617
    'RoPE backward kernel'              = 10.617
    'RoPE BF16 kernel'                  = 10.617
    'RoPE BF16 backward kernel'         = 10.617
    'Flash attention kernel'            = 10.486
    'Flash attention LSE forward kernel'= 10.551
    'Flash attention backward kernel'   = 16.777
    'Flash attention LSE backward kernel'= 21.036
    'Embedding kernel'                  = 8.389
    'Embedding backward kernel'         = 12.583
    'Cross entropy kernel'              = 205.85
}

$rows = [System.Collections.Generic.List[object]]::new()
$pattern = '^\s*(?<name>.+?kernel)\s+\|\s+(?<ms>[0-9]+(?:\.[0-9]+)?)\s*$'
foreach ($line in $lines) {
    $match = [regex]::Match($line, $pattern)
    if (-not $match.Success) { continue }
    $name = $match.Groups['name'].Value.Trim()
    if ($name.StartsWith('CUPTI:')) { continue }
    $ms = [double]$match.Groups['ms'].Value
    if ($ms -le 0) { continue }

    $throughput = 1000.0 / $ms
    if ($trafficMB.ContainsKey($name)) {
        $gbps = $trafficMB[$name] / $ms
        $percent = 100.0 * $gbps / $PeakMemoryGBps
        $rows.Add([pscustomobject]@{
            Kernel = $name
            'Average ms' = $ms
            'Launches/s' = $throughput
            'GB/s' = $gbps
            '% of peak' = $percent
        })
    } else {
        $rows.Add([pscustomobject]@{
            Kernel = $name
            'Average ms' = $ms
            'Launches/s' = $throughput
            'GB/s' = $null
            '% of peak' = $null
        })
    }
}

if ($rows.Count -eq 0) {
    throw "Profiler returned no benchmarks. Raw log: $logPath"
}

$markdown = [System.Collections.Generic.List[string]]::new()
$markdown.Add('# Kernel profile')
$markdown.Add('')
$markdown.Add("- Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$markdown.Add("- GPU memory peak reference: $PeakMemoryGBps GB/s")
$markdown.Add("- Raw log: ``$([System.IO.Path]::GetFileName($logPath))``")
$markdown.Add('')
$markdown.Add('| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |')
$markdown.Add('|---|---:|---:|---:|---:|')
foreach ($row in $rows) {
    $gb = if ($null -eq $row.'GB/s') { '-' } else { '{0:N1}' -f $row.'GB/s' }
    $pct = if ($null -eq $row.'% of peak') { '-' } else { '{0:N1}%' -f $row.'% of peak' }
    $formatArguments = @($row.Kernel, $row.'Average ms', $row.'Launches/s', $gb, $pct)
    $markdown.Add(('| {0} | {1:N4} | {2:N1} | {3} | {4} |' -f $formatArguments))
}

$markdown | Set-Content -LiteralPath $OutputMarkdown -Encoding utf8
Write-Host "Markdown table: $([System.IO.Path]::GetFullPath($OutputMarkdown))"
Write-Host "Raw log:        $([System.IO.Path]::GetFullPath($logPath))"

if ($exitCode -ne 0) {
    throw "Runner failed with exit code $exitCode. Check log: $logPath"
}
