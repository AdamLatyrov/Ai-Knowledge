param([string]$DatabasePath = '')

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
if ([string]::IsNullOrWhiteSpace($DatabasePath)) {
    $database = Join-Path $root 'knowledge.sqlite3'
} elseif ([System.IO.Path]::IsPathRooted($DatabasePath)) {
    $database = [System.IO.Path]::GetFullPath($DatabasePath)
} else {
    $database = [System.IO.Path]::GetFullPath((Join-Path $root $DatabasePath))
}
$databaseDirectory = Split-Path -Parent $database
New-Item -ItemType Directory -Path $databaseDirectory -Force | Out-Null
$env:AI_KNOWLEDGE_DATABASE = $database

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js is required. Install Node.js 20 or newer, then rerun setup.ps1.' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'npm is required. Install Node.js 20 or newer, then rerun setup.ps1.' }
$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if (-not $launcher) { throw 'Install Python 3.12 and the Python Launcher for Windows, then rerun setup.ps1.' }

if (-not (Test-Path -LiteralPath $python)) {
    & $launcher.Source -3.12 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Python package manager setup failed.' }
& $python -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies could not be installed.' }

Push-Location $root
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Node dependencies could not be installed.' }
    & npm run build:mcp
    if ($LASTEXITCODE -ne 0) { throw 'MCP server build failed.' }

    $oldOffline = $env:HF_HUB_OFFLINE
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
    try {
        $warmup = @'
import json
from pathlib import Path
from fastembed import TextEmbedding
root = Path.cwd()
config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
vector = config['vector']
cache = root / vector.get('cache_dir', 'models')
cache.mkdir(parents=True, exist_ok=True)
TextEmbedding(model_name=vector['model'], cache_dir=str(cache))
print('Embedding model is ready.')
'@
        $warmup | & $python -
        if ($LASTEXITCODE -ne 0) { throw 'Could not download or initialize the embedding model. Check network access and rerun setup.ps1.' }
    } finally {
        if ($null -ne $oldOffline) { $env:HF_HUB_OFFLINE = $oldOffline }
    }

    & $python (Join-Path $root 'tools\knowledge.py') sync
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the knowledge database.' }
    & $python (Join-Path $root 'tools\memory_service.py') build
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the local search index.' }
} finally {
    Pop-Location
}

$codexDir = Join-Path $root '.codex'
New-Item -ItemType Directory -Path $codexDir -Force | Out-Null
$rootToml = $root.Replace('\', '/')
$serverToml = (Join-Path $root 'build\mcp\server.js').Replace('\', '/')
$dbToml = $database.Replace('\', '/')
$toml = @(
    '[mcp_servers.ai_knowledge]',
    'command = "node"',
    "args = [`"$serverToml`"]",
    "cwd = `"$rootToml`"",
    '',
    '[mcp_servers.ai_knowledge.env]',
    "AI_KNOWLEDGE_DATABASE = `"$dbToml`"",
    'HF_HUB_OFFLINE = "1"',
    'PYTHONUTF8 = "1"'
) -join [Environment]::NewLine
$configPath = Join-Path $root 'config.json'
$localConfig = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$rootPrefix = $root.TrimEnd('\') + '\'
if ($database.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    $localConfig.database = $database.Substring($rootPrefix.Length).Replace('\', '/')
} else {
    $localConfig.database = $database
}
[System.IO.File]::WriteAllText($configPath, ($localConfig | ConvertTo-Json -Depth 100), [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText((Join-Path $codexDir 'config.toml'), $toml, [System.Text.UTF8Encoding]::new($false))

Write-Host 'AI Knowledge is installed. Trust this folder in Codex and reopen the project to load the local MCP server.' -ForegroundColor Green
