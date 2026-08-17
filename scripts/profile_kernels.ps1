[CmdletBinding()]
param([string]$BuildDir, [string]$OutputMarkdown, [double]$PeakMemoryGBps = 192.0)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($BuildDir)) { $BuildDir = Join-Path $scriptRoot '..\cmake-build-default-visual-studio\Release' }
if ([string]::IsNullOrWhiteSpace($OutputMarkdown)) { $OutputMarkdown = Join-Path $scriptRoot '..\kernel_profile.md' }
python (Join-Path $scriptRoot 'profile_kernels.py') --build-dir $BuildDir --output $OutputMarkdown --peak-memory-gbps $PeakMemoryGBps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
