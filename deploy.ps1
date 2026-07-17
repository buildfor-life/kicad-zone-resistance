# Deploy the plugin into the KiCad user plugins directory.
#   .\deploy.ps1                 -> NTFS junction (best for development)
#   .\deploy.ps1 -Mode Copy      -> plain copy (no repo link)
param(
    [ValidateSet('Junction', 'Copy')]
    [string]$Mode = 'Junction',
    [string]$KiCadVersion = '10.0'
)

$src = $PSScriptRoot
$pluginsDir = Join-Path ([Environment]::GetFolderPath('MyDocuments')) "KiCad\$KiCadVersion\plugins"
$dst = Join-Path $pluginsDir 'fill-resistance'

if (-not (Test-Path $pluginsDir)) {
    Write-Error "KiCad plugins dir not found: $pluginsDir"
    exit 1
}

if (Test-Path $dst) {
    $item = Get-Item $dst -Force
    if ($item.LinkType) {
        # junction: rmdir removes the link only, never the target contents
        cmd /c rmdir "$dst"
    } else {
        Remove-Item $dst -Recurse -Force
    }
}

if ($Mode -eq 'Junction') {
    New-Item -ItemType Junction -Path $dst -Target $src | Out-Null
    Write-Host "junction created: $dst -> $src"
} else {
    $exclude = @('.venv', '.git', 'tests', 'tools', '__pycache__', '.pytest_cache',
                 'pyproject.toml', 'uv.lock')
    New-Item -ItemType Directory -Force $dst | Out-Null
    Get-ChildItem $src -Force | Where-Object { $exclude -notcontains $_.Name } |
        ForEach-Object { Copy-Item $_.FullName -Destination $dst -Recurse -Force }
    Write-Host "copied plugin to: $dst"
}
Write-Host "Restart KiCad (or refresh plugins) and wait for the plugin venv build."
