# Keep Ninja output in a log and report useful progress while the native process runs.
$ErrorActionPreference = 'Stop'
$log = Join-Path $PWD 'build-release/build.log'
$errors = Join-Path $PWD 'build-release/build-errors.log'
$build = Start-Process cmake -ArgumentList @('--build', 'build-release', '-j', '--target',
    'ninfer', 'ninfer-serve', 'ninfer_bench') -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errors
while (-not $build.WaitForExit(15000)) {
    $progress = Get-Content -LiteralPath $log -Tail 100
    Write-Host "[$(Get-Date -Format o)] build PID $($build.Id), log bytes $((Get-Item $log).Length)"
    $progress | Select-String -Pattern '^\[\d+/\d+\]' | Select-Object -Last 1 | ForEach-Object { Write-Host $_.Line }
    $progress | Select-Object -Last 2 | Write-Host
    Get-Content -LiteralPath $errors -Tail 2 | Write-Host
}
$build.WaitForExit()
Get-Content -LiteralPath $log -Tail 30 | Write-Host
Get-Content -LiteralPath $errors -Tail 30 | Write-Host
if ($build.ExitCode -ne 0) { throw "CMake build failed ($($build.ExitCode))" }
