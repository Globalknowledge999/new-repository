<#
.SYNOPSIS
  Installs (or updates) Parallaxa for the current user. No administrator rights needed.

.DESCRIPTION
  Into $InstallDir it puts:
    uv\        the uv tool (fetches Python and packages)
    python\    a private Python 3.12
    venv\      Parallaxa + PyTorch (CUDA build matched to the NVIDIA driver, else CPU)
    ffmpeg\    an FFmpeg build whose NVENC version suits the driver
  It then checks the result, optionally pre-downloads the depth models, and creates shortcuts
  plus an "Apps & features" uninstall entry. Running it again updates Parallaxa in place.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\Parallaxa"),
    [string]$Source = "",                       # Parallaxa source folder; default: bundled/next to this script, else GitHub
    [ValidateSet("auto", "cuda", "cpu")][string]$Torch = "auto",
    [switch]$SkipModels,                        # don't pre-download the depth models
    [switch]$NoShortcuts,
    [switch]$NoRegister                         # don't add an Apps & features entry (the Setup.exe registers its own)
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns anything a native tool (uv, ffmpeg) writes to
# stderr into an error and would abort under "Stop". Native tools are checked via $LASTEXITCODE and
# cmdlets that must succeed use -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"        # Invoke-WebRequest is many times faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$UvVersion = "0.13.6"
$PythonVersion = "3.12"
$GitHubZip = "https://github.com/Globalknowledge999/new-repository/archive/refs/heads/main.zip"
# FFmpeg NVENC needs a minimum NVIDIA driver: 8.0.x -> 570+, 7.1.x -> 550+ (9.x needs 610+, so it isn't used).
$FfmpegBuilds = @(
    @{ Version = "8.0.1"; MinDriver = 570 },
    @{ Version = "7.1.1"; MinDriver = 0 }
)

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "    $text" }
function Download($url, $dest) {
    for ($i = 1; $i -le 3; $i++) {
        try { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -ErrorAction Stop; return }
        catch { if ($i -eq 3) { throw "Download failed: $url`n$_" }; Start-Sleep -Seconds (2 * $i) }
    }
}
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$([IO.Path]::GetFileName($exe)) failed (exit code $LASTEXITCODE): $($arguments -join ' ')" }
}

New-Item -ItemType Directory -Force -Path $InstallDir -ErrorAction Stop | Out-Null
$InstallDir = (Resolve-Path $InstallDir -ErrorAction Stop).Path
$log = Join-Path $InstallDir "install.log"
Start-Transcript -Path $log -Force | Out-Null
try {
    Write-Host "Parallaxa installer -> $InstallDir" -ForegroundColor Green

    # ---------------------------------------------------------------- GPU / driver
    Step "Checking for an NVIDIA GPU"
    $driver = 0
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $q = & $smi.Source --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        if ($q) {
            $name, $ver = $q -split ",\s*"
            $driver = [int]($ver -split "\.")[0]
            Info "$name, driver $ver"
        }
    }
    if ($driver -eq 0) { Info "No NVIDIA GPU found: Parallaxa will run on the CPU (much slower)." }

    $torchIndex = "https://download.pytorch.org/whl/cpu"
    if ($Torch -ne "cpu" -and $driver -ge 570) { $torchIndex = "https://download.pytorch.org/whl/cu128" }
    elseif ($Torch -ne "cpu" -and $driver -ge 560) { $torchIndex = "https://download.pytorch.org/whl/cu126" }
    elseif ($Torch -eq "cuda") { throw "A CUDA build of PyTorch needs NVIDIA driver 560 or newer (found: $driver). Update the driver or use -Torch cpu." }
    elseif ($driver -gt 0) { Info "Driver $driver is too old for current CUDA PyTorch (needs 560+); using the CPU build. Update the driver for GPU speed." }
    Info "PyTorch build: $($torchIndex.Split('/')[-1])"

    # ---------------------------------------------------------------- uv + Python
    Step "Getting uv $UvVersion and Python $PythonVersion"
    $uvDir = Join-Path $InstallDir "uv"
    $uv = Join-Path $uvDir "uv.exe"
    $haveUv = (Test-Path $uv) -and ((& $uv --version 2>$null) -match [regex]::Escape($UvVersion))
    if (-not $haveUv) {
        $zip = Join-Path $env:TEMP "uv-$UvVersion.zip"
        Download "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        Expand-Archive -Path $zip -DestinationPath $uvDir -Force -ErrorAction Stop
        Remove-Item $zip -Force
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir "python"
    $env:UV_CACHE_DIR = Join-Path $InstallDir "cache"
    $env:UV_LINK_MODE = "copy"
    $venv = Join-Path $InstallDir "venv"
    $py = Join-Path $venv "Scripts\python.exe"
    # uv adds a folder junction per minor version (cpython-3.12-windows-...) pointing at the real install
    # (cpython-3.12.N-windows-...). Recent Windows 11 builds refuse to follow junctions created by a
    # non-admin user ("untrusted mount point", os error 448), so remove them (before and after installing:
    # a failed earlier attempt may have left one) and use the real folder.
    $pyRoot = $env:UV_PYTHON_INSTALL_DIR
    function Remove-PythonJunctions {
        Get-ChildItem $pyRoot -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } |
            ForEach-Object { cmd /c rmdir "$($_.FullName)" | Out-Null }
    }
    Remove-PythonJunctions
    & $uv python install $PythonVersion
    $installCode = $LASTEXITCODE
    Remove-PythonJunctions
    $real = Get-ChildItem $pyRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match ("^cpython-" + [regex]::Escape($PythonVersion) + "\.\d+-") } |
        Sort-Object { [version]($_.Name -replace '^cpython-(\d+\.\d+\.\d+).*', '$1') } -Descending |
        Select-Object -First 1
    if (-not $real) { throw "Python $PythonVersion was not found in $pyRoot after installing it (uv exit code $installCode)" }
    if ($installCode -ne 0) { Info "uv reported a problem after installing Python (exit code $installCode); continuing with the installed copy." }
    $basePython = Join-Path $real.FullName "python.exe"
    Info "Python: $basePython"

    # A venv left by an earlier, failed attempt may point at the junction: rebuild it if it doesn't run.
    if (Test-Path $py) {
        & $py -c "import sys" 2>$null
        if ($LASTEXITCODE -ne 0) { Remove-Item $venv -Recurse -Force -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $py)) { Run $uv @("venv", $venv, "--python", $basePython) }
    if (-not (Test-Path $py)) { throw "Could not create the Python environment in $venv" }
    Info (& $py --version)

    # ---------------------------------------------------------------- PyTorch + Parallaxa
    Step "Installing PyTorch (about 3 GB for the CUDA build; this is the slow part)"
    Run $uv @("pip", "install", "--python", $py, "torch", "torchvision", "--index-url", $torchIndex)

    Step "Installing Parallaxa"
    if (-not $Source) {
        foreach ($cand in @((Join-Path $PSScriptRoot "app"), (Split-Path $PSScriptRoot -Parent))) {
            if (Test-Path (Join-Path $cand "pyproject.toml")) { $Source = $cand; break }
        }
    }
    if (-not $Source) {
        $zip = Join-Path $env:TEMP "parallaxa-src.zip"
        $tmp = Join-Path $env:TEMP "parallaxa-src"
        Download $GitHubZip $zip
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $tmp -Force -ErrorAction Stop
        $Source = (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName
    }
    Info "from $Source"
    # Keep the already-installed CUDA torch: resolve other packages from PyPI only.
    Run $uv @("pip", "install", "--python", $py, "--reinstall-package", "parallaxa", $Source)

    # ---------------------------------------------------------------- FFmpeg
    Step "Getting FFmpeg"
    $ffDir = Join-Path $InstallDir "ffmpeg"
    $ffExe = Join-Path $ffDir "bin\ffmpeg.exe"
    $wanted = $FfmpegBuilds | Where-Object { $driver -ge $_.MinDriver -or $driver -eq 0 } | Select-Object -First 1
    $versionFile = Join-Path $ffDir "VERSION.txt"
    $installedVersion = ""
    if (Test-Path $versionFile) { $installedVersion = (Get-Content $versionFile -Raw).Trim() }

    function Install-Ffmpeg($version) {
        $zip = Join-Path $env:TEMP "ffmpeg-$version.zip"
        $tmp = Join-Path $env:TEMP "ffmpeg-$version"
        Download "https://github.com/GyanD/codexffmpeg/releases/download/$version/ffmpeg-$version-essentials_build.zip" $zip
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $tmp -Force -ErrorAction Stop
        if (Test-Path $ffDir) { Remove-Item $ffDir -Recurse -Force }
        New-Item -ItemType Directory -Force -Path (Join-Path $ffDir "bin") | Out-Null
        $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
        Copy-Item (Join-Path $inner.FullName "bin\*.exe") (Join-Path $ffDir "bin") -ErrorAction Stop
        Copy-Item (Join-Path $inner.FullName "LICENSE*") $ffDir -ErrorAction SilentlyContinue
        Set-Content -Path $versionFile -Value $version
        Remove-Item $zip, $tmp -Recurse -Force
    }
    function Test-Nvenc {
        $out = & $ffExe -hide_banner -loglevel error -f lavfi -i "color=black:s=256x256:d=0.1" -frames:v 1 -c:v hevc_nvenc -f null - 2>&1
        return @{ Ok = ($LASTEXITCODE -eq 0); Message = ($out | Out-String).Trim() }
    }
    if ($installedVersion -ne $wanted.Version -or -not (Test-Path $ffExe)) { Install-Ffmpeg $wanted.Version }
    Info ((& $ffExe -version | Select-Object -First 1))
    if ($driver -gt 0) {
        $nv = Test-Nvenc
        if (-not $nv.Ok -and $nv.Message -match "nvenc API version" -and $wanted.Version -ne $FfmpegBuilds[-1].Version) {
            Info "NVENC not supported by this driver with FFmpeg $($wanted.Version); falling back to $($FfmpegBuilds[-1].Version)."
            Install-Ffmpeg $FfmpegBuilds[-1].Version
            $nv = Test-Nvenc
        }
        if ($nv.Ok) { Info "GPU video encoding (NVENC): OK" } else { Info "GPU video encoding unavailable; videos will be encoded on the CPU. ($($nv.Message))" }
    }

    # ---------------------------------------------------------------- check
    Step "Checking the installation"
    $cli = Join-Path $venv "Scripts\parallaxa.exe"
    Run $cli @("info")

    if (-not $SkipModels) {
        Step "Downloading the depth models (about 1.4 GB, once)"
        & $cli download
        if ($LASTEXITCODE -ne 0) { Info "Model download did not finish; Parallaxa will download them when first needed." }
    }

    # ---------------------------------------------------------------- shortcuts, uninstaller
    $gui = Join-Path $venv "Scripts\parallaxa-gui.exe"
    if ((Join-Path $PSScriptRoot "uninstall.ps1") -ne (Join-Path $InstallDir "uninstall.ps1")) {
        Copy-Item (Join-Path $PSScriptRoot "uninstall.ps1") (Join-Path $InstallDir "uninstall.ps1") -Force -ErrorAction Stop
    }
    if (-not $NoShortcuts) {
        Step "Creating shortcuts"
        $shell = New-Object -ComObject WScript.Shell
        $menu = Join-Path ([Environment]::GetFolderPath("Programs")) "Parallaxa"
        New-Item -ItemType Directory -Force -Path $menu | Out-Null
        foreach ($lnkPath in @((Join-Path $menu "Parallaxa.lnk"), (Join-Path ([Environment]::GetFolderPath("Desktop")) "Parallaxa.lnk"))) {
            $lnk = $shell.CreateShortcut($lnkPath)
            $lnk.TargetPath = $gui
            $lnk.WorkingDirectory = [Environment]::GetFolderPath("MyPictures")
            $lnk.Description = "Convert 2D photos and videos to 3D"
            $lnk.Save()
        }
        $un = $shell.CreateShortcut((Join-Path $menu "Uninstall Parallaxa.lnk"))
        $un.TargetPath = "powershell.exe"
        $un.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallDir 'uninstall.ps1')`""
        $un.Save()
        Info "Start menu > Parallaxa, and a desktop icon"
    }
    if (-not $NoRegister) {
        $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallaxa"
        New-Item -Path $key -Force | Out-Null
        $version = (& $py -c "import parallaxa; print(parallaxa.__version__)").Trim()
        Set-ItemProperty $key DisplayName "Parallaxa"
        Set-ItemProperty $key DisplayVersion $version
        Set-ItemProperty $key Publisher "Parallaxa (personal project)"
        Set-ItemProperty $key InstallLocation $InstallDir
        Set-ItemProperty $key DisplayIcon $gui
        Set-ItemProperty $key UninstallString "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallDir 'uninstall.ps1')`""
        Set-ItemProperty $key NoModify 1
        Set-ItemProperty $key NoRepair 1
    }

    Step "Tidying up"
    & $uv cache clean 2>$null | Out-Null
    Write-Host ""
    Write-Host "Parallaxa is installed. Start it from the Start menu or the desktop icon." -ForegroundColor Green
    Write-Host "(Log: $log)"
}
catch {
    Write-Host ""
    Write-Host "INSTALL FAILED: $_" -ForegroundColor Red
    Write-Host "The full log is in $log"
    Stop-Transcript | Out-Null
    exit 1
}
Stop-Transcript | Out-Null
exit 0
