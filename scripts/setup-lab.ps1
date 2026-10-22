param([string]$Distribution = 'Ubuntu-20.04', [switch]$Lean)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$wslRoot = (wsl -d $Distribution -- wslpath -a ($projectRoot.Replace('\', '/'))).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve project path in WSL' }
wsl -d $Distribution --cd $wslRoot -- bash scripts/setup_lab_wsl.sh
if ($LASTEXITCODE -ne 0) { throw 'Lab environment setup failed' }
if ($Lean) {
    $archive = Join-Path $projectRoot '.tools/lean-4.35.0-linux.tar.zst'
    $expected = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
    if (-not (Test-Path -LiteralPath $archive) -or
        (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        curl.exe -fL -C - -o $archive https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst
        if ($LASTEXITCODE -ne 0) { throw 'Lean download incomplete; rerun with -Lean to resume' }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        throw 'Lean archive differs from the pinned official release'
    }
    tar.exe -xf $archive -C (Join-Path $projectRoot '.tools')
    if ($LASTEXITCODE -ne 0) { throw 'Lean extraction failed' }
    wsl -d $Distribution --cd $wslRoot -- .tools/lean-4.35.0-linux/bin/lean --version
    if ($LASTEXITCODE -ne 0) { throw 'Lean toolchain unavailable' }
    $proofArchive = Join-Path $projectRoot '.tools/bubblewrap-0.11.0.tar.gz'
    $proofHash = 'cfeeb15fcc47d177d195f06fdf0847e93ee3aa6bf46f6ac0a141fa142759e2c3'
    if (-not (Test-Path -LiteralPath $proofArchive)) {
        curl.exe -fL -o $proofArchive https://codeload.github.com/containers/bubblewrap/tar.gz/refs/tags/v0.11.0
        if ($LASTEXITCODE -ne 0) { throw 'Proof sandbox download failed' }
    }
    if ((Get-FileHash -LiteralPath $proofArchive -Algorithm SHA256).Hash.ToLower() -ne $proofHash) {
        throw 'Proof sandbox source differs from the pinned release'
    }
    tar.exe -xf $proofArchive -C (Join-Path $projectRoot '.tools')
    if ($LASTEXITCODE -ne 0) { throw 'Proof sandbox extraction failed' }
    wsl -d $Distribution --cd $wslRoot -- bash scripts/setup_lab_wsl.sh --proof-tools
    if ($LASTEXITCODE -ne 0) { throw 'Proof sandbox build failed' }
}
