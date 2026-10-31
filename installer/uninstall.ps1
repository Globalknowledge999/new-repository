<#
.SYNOPSIS
  Removes Parallaxa installed by install.ps1. Downloaded models (in the shared Hugging Face cache)
  and your converted files are kept unless you choose otherwise.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = $PSScriptRoot,
    [switch]$Quiet,
    [switch]$RemoveModels
)
$ErrorActionPreference = "Continue"

if (-not (Test-Path (Join-Path $InstallDir "venv"))) {
    Write-Host "No Parallaxa installation found in $InstallDir" -ForegroundColor Yellow
    if (-not $Quiet) { Read-Host "Press Enter to close" }
    exit 1
}
if (-not $Quiet) {
    $answer = Read-Host "Remove Parallaxa from $InstallDir ? (y/n)"
    if ($answer -notmatch "^[yY]") { exit 0 }
    if (-not $RemoveModels) {
        $m = Read-Host "Also delete the downloaded AI models (several GB in your Hugging Face cache, shared with other apps)? (y/n)"
        $RemoveModels = $m -match "^[yY]"
    }
}

# Make sure nothing from the install is running.
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase) } |
    Stop-Process -Force -ErrorAction SilentlyContinue

$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "Parallaxa"
Remove-Item $menu -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path ([Environment]::GetFolderPath("Desktop")) "Parallaxa.lnk") -Force -ErrorAction SilentlyContinue
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallaxa" -Recurse -Force -ErrorAction SilentlyContinue

if ($RemoveModels) {
    $hub = if ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
    $patterns = @("models--depth-anything--*", "models--stable-diffusion-v1-5--stable-diffusion-inpainting",
                  "models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1", "models--stabilityai--stable-diffusion-2-inpainting")
    foreach ($p in $patterns) { Get-ChildItem $hub -Directory -Filter $p -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force }
    Write-Host "Removed downloaded models."
}

foreach ($d in @("venv", "python", "uv", "ffmpeg", "cache", "app", "installer")) {
    Remove-Item (Join-Path $InstallDir $d) -Recurse -Force -ErrorAction SilentlyContinue
}
Remove-Item (Join-Path $InstallDir "install.log") -Force -ErrorAction SilentlyContinue
# The folder itself (and this script) go last, from a detached process, because this script is inside it.
Start-Process -WindowStyle Hidden -FilePath "cmd.exe" -ArgumentList "/c timeout /t 2 >nul & rmdir /s /q `"$InstallDir`""
Write-Host "Parallaxa has been removed. Your settings are in %APPDATA%\Parallaxa (delete that folder to forget them)." -ForegroundColor Green
if (-not $Quiet) { Read-Host "Press Enter to close" }
