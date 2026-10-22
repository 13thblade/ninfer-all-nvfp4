# Package the configured Windows build. CUDA and architecture labels come from its build tree.
# Example: $env:NINFER_BUILD_ROOT = 'C:\ninfer\build-release'; ./scripts/package-release.ps1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = if ($env:NINFER_BUILD_ROOT) { $env:NINFER_BUILD_ROOT } else { Join-Path $RepoRoot 'build-ninja' }
$BuildRoot = (Resolve-Path -LiteralPath $BuildRoot).Path
$Cache = Get-Content -LiteralPath (Join-Path $BuildRoot 'CMakeCache.txt')
function Read-Cache([string]$Name) {
    $entry = @($Cache | Where-Object { $_ -match "^${Name}:[^=]+=" })
    if ($entry.Count -ne 1) { throw "Expected one $Name entry in CMakeCache.txt" }
    return ($entry[0] -split '=', 2)[1]
}
$Archs = (Read-Cache 'CMAKE_CUDA_ARCHITECTURES').Split(';')
foreach ($arch in $Archs) {
    if ($arch -notmatch '^(70|75|80|86|89|120a)$') { throw "Unsupported architecture: $arch" }
}
$Nvcc = Read-Cache 'CMAKE_CUDA_COMPILER'
# Use the compiler that produced the build, even when another Toolkit is on PATH.
$CudaRoot = Split-Path -Parent (Split-Path -Parent $Nvcc)
$CudaVersionText = (& $Nvcc --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $CudaVersionText -notmatch 'release (\d+\.\d+),') {
    throw 'Cannot read the build compiler CUDA version'
}
$CudaVersion = $Matches[1]
$CudaMajor = $CudaVersion.Split('.')[0]
$Version = (Get-Content -LiteralPath (Join-Path $RepoRoot 'VERSION') -Raw).Trim()
if ($Version -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'VERSION is not a valid release-notes identifier' }
$ReleaseNotes = Join-Path $RepoRoot "RELEASE_NOTES_$($Version.Split('-')[0]).md"
$Revision = (& git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the source revision' }
$SourceState = @(& git -C $RepoRoot status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect source changes' }
if ($env:GITHUB_ACTIONS -eq 'true' -and $SourceState.Count -ne 0) { throw 'CI source tree is modified' }
$Suffix = if ($SourceState.Count) { '-dirty' } else { '' }
$ProductName = "ninfer-windows-x64-cuda$CudaVersion-sm$($Archs -join '-')-$($Revision.Substring(0,12))$Suffix"
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductRoot = Join-Path $DistRoot $ProductName
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-$ProductName.txt"
$ArchivePath = Join-Path $DistRoot "$ProductName.7z"
$SevenZip = (Get-Command 7z -ErrorAction Stop).Source
$Products = @('apps\ninfer.exe', 'apps\ninfer-serve.exe', 'bench\ninfer_bench.exe')
foreach ($product in $Products) {
    if (-not (Test-Path -LiteralPath (Join-Path $BuildRoot $product))) { throw "Missing product: $product" }
}
$Eula = Join-Path $CudaRoot 'EULA.txt'
if (-not (Test-Path -LiteralPath $Eula)) { throw "Missing CUDA redistributable licence: $Eula" }
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
# Only replace this exact build's package, never another variant's assets.
if (Test-Path -LiteralPath $ProductRoot) { Remove-Item -LiteralPath $ProductRoot -Recurse -Force }
Get-ChildItem -LiteralPath $DistRoot -File | Where-Object {
    $_.Name -eq (Split-Path -Leaf $ChecksumPath) -or $_.Name -match "^$([regex]::Escape($ProductName))\.7z\.\d+$"
} | Remove-Item -Force
New-Item -ItemType Directory -Path $ProductRoot | Out-Null
foreach ($product in $Products) {
    Copy-Item -LiteralPath (Join-Path $BuildRoot $product) -Destination $ProductRoot
}
# Include the vcpkg runtime closure, including dependencies of FFmpeg and curl.
$DllDirectories = @((Join-Path $BuildRoot 'apps'), (Join-Path $BuildRoot 'bench'))
$VcpkgEntry = @($Cache | Where-Object { $_ -match '^VCPKG_INSTALLED_DIR:[^=]+=' })
if ($VcpkgEntry.Count -eq 1) {
    $VcpkgInstalled = ($VcpkgEntry[0] -split '=', 2)[1]
    $DllDirectories += Join-Path $VcpkgInstalled "$(Read-Cache 'VCPKG_TARGET_TRIPLET')\bin"
}
foreach ($directory in $DllDirectories) {
    if (Test-Path -LiteralPath $directory) {
        Get-ChildItem -LiteralPath $directory -Filter '*.dll' | Copy-Item -Destination $ProductRoot
    }
}
foreach ($name in "cublas64_$CudaMajor.dll", "cublasLt64_$CudaMajor.dll") {
    $candidates = @((Join-Path $CudaRoot "bin\$name"), (Join-Path $CudaRoot "bin\x64\$name"))
    $found = @($candidates | Where-Object { Test-Path -LiteralPath $_ })
    if ($found.Count -ne 1) { throw "Expected one $name under $CudaRoot" }
    Copy-Item -LiteralPath $found[0] -Destination $ProductRoot
}
Copy-Item -LiteralPath $Eula -Destination (Join-Path $ProductRoot 'NVIDIA-CUDA-EULA.txt')
foreach ($file in 'VERSION', 'LICENSE') { Copy-Item -LiteralPath (Join-Path $RepoRoot $file) -Destination $ProductRoot }
if (Test-Path -LiteralPath $ReleaseNotes) {
    Copy-Item -LiteralPath $ReleaseNotes -Destination $ProductRoot
}
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\release-archive-windows.md') -Destination (Join-Path $ProductRoot 'README.md')
foreach ($script in 'run.bat', 'download-model.bat') {
    Copy-Item -LiteralPath (Join-Path $RepoRoot "scripts\$script") -Destination $ProductRoot
}
# Preserve dependency notices from the triplet used by the build.
if ($VcpkgEntry.Count -eq 1) {
    $share = Join-Path $VcpkgInstalled "$(Read-Cache 'VCPKG_TARGET_TRIPLET')\share"
    foreach ($notice in (Get-ChildItem -LiteralPath $share -Filter copyright -Recurse)) {
        $name = Split-Path -Leaf (Split-Path -Parent $notice.FullName)
        Copy-Item -LiteralPath $notice.FullName -Destination (Join-Path $ProductRoot "LICENSE-$name.txt")
    }
}
$Cuobjdump = Join-Path $CudaRoot 'bin\cuobjdump.exe'
foreach ($product in $Products) {
    $exe = Join-Path $ProductRoot (Split-Path -Leaf $product)
    $images = (& $Cuobjdump --list-elf $exe 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "cuobjdump failed for $exe" }
    foreach ($arch in $Archs) {
        if ($images -notmatch "sm_$arch(?:[.\s_]|$)") { throw "$exe lacks the sm_$arch device image" }
    }
    # No Toolkit, vcpkg or developer directories can satisfy missing runtime DLLs.
    $savedPath = $env:PATH
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        & $exe --help
        if ($LASTEXITCODE -ne 0) { throw "$exe --help failed with a clean PATH ($LASTEXITCODE)" }
    } finally { $env:PATH = $savedPath }
}
@{ source_revision = $Revision; modified_sources = [bool]$SourceState.Count; version = $Version;
   cuda = $CudaVersion; cuda_compiler = $CudaVersionText; architectures = $Archs; platform = 'windows-x64';
   cuda_archive_ptx_pruned = ((Read-Cache 'NINFER_CUDA_PRUNE_PTX') -eq 'ON');
   validation = 'Device images present; all three executables passed --help without Toolkit on PATH. GPU inference was not tested.'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ProductRoot 'BUILD-INFO.json') -Encoding utf8
$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)"
}
[IO.File]::WriteAllText((Join-Path $ProductRoot 'SHA256SUMS.txt'), (($innerHashes -join "`n") + "`n"), [Text.ASCIIEncoding]::new())
# Always use volumes. Every asset stays below GitHub's 2 GiB limit, even as the fatbins grow.
& $SevenZip a -t7z -mx=3 -v1900m $ArchivePath $ProductRoot
if ($LASTEXITCODE -ne 0) { throw 'Archive creation failed' }
& $SevenZip t "$ArchivePath.001"
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed' }
$volumes = @(Get-ChildItem -LiteralPath $DistRoot -File | Where-Object {
    $_.Name -match "^$([regex]::Escape($ProductName))\.7z\.\d+$"
} | Sort-Object Name)
if ($volumes.Count -eq 0) { throw 'No archive volumes were created' }
$outerHashes = foreach ($volume in $volumes) {
    if ($volume.Length -ge 2GB) { throw "Release asset is too large: $($volume.Name)" }
    "$((Get-FileHash -LiteralPath $volume.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($volume.Name)"
}
[IO.File]::WriteAllText($ChecksumPath, (($outerHashes -join "`n") + "`n"), [Text.ASCIIEncoding]::new())
$volumes | Select-Object Name, Length
