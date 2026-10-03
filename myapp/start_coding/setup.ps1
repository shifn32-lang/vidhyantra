# __BRAND__ - Start coding: one-line setup for the OpenCode terminal agent (Windows).
#
#   irm __BASE_URL__/start-coding/setup.ps1 | iex
#
# What it does, in order:
#   1. installs OpenCode if it is missing (npm or scoop)
#   2. shows a short code and opens your browser so YOU approve this computer
#   3. saves the key it receives in OpenCode's own credentials file
#   4. adds __MODEL_LABEL__ to OpenCode's config
# It never asks you to paste a key, and it only touches OpenCode's two files:
#   %USERPROFILE%\.local\share\opencode\auth.json
#   %USERPROFILE%\.config\opencode\opencode.json

$ErrorActionPreference = 'Stop'
$BaseUrl    = '__BASE_URL__'
$Brand      = '__BRAND__'
$ModelId    = '__MODEL_ID__'
$ModelLabel = '__MODEL_LABEL__'

function Say($text, $color = $null) { if ($color) { Write-Host $text -ForegroundColor $color } else { Write-Host $text } }
function Step($text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Green }
function Fail($text) { Write-Host "x $text" -ForegroundColor Red; throw $text }

# Windows PowerShell 5.1 defaults to old TLS versions on some machines.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

# Write UTF-8 without a byte-order mark (OpenCode's JSON reader dislikes one).
function Write-Utf8($path, $text) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

# Read a JSON object file for merging. $null = file missing, $false = unreadable
# (e.g. it has comments) - in that case we do not overwrite it.
function Read-JsonFile($path) {
    if (-not (Test-Path $path)) { return $null }
    try {
        $raw = Get-Content -Raw -Encoding UTF8 $path
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        $obj = $raw | ConvertFrom-Json
        if ($obj -isnot [System.Management.Automation.PSCustomObject]) { return $false }
        return $obj
    } catch { return $false }
}

function Set-Prop($obj, $name, $value) {
    if ($obj.PSObject.Properties[$name]) { $obj.$name = $value } else { $obj | Add-Member -NotePropertyName $name -NotePropertyValue $value }
}

Say ''
Say "$Brand - Start coding" 'White'
Say "Sets up OpenCode with $ModelLabel on this computer." 'DarkGray'

# ---------------------------------------------------------------- 1. OpenCode
Step '1/4  Checking OpenCode'
$opencode = Get-Command opencode -ErrorAction SilentlyContinue
if ($opencode) {
    Say 'OpenCode is already installed.'
} else {
    Say 'OpenCode is not installed yet. Installing it...'
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        npm install -g opencode-ai
        if ($LASTEXITCODE -ne 0) { Fail 'npm could not install OpenCode. See https://opencode.ai/docs, then run this again.' }
    } elseif (Get-Command scoop -ErrorAction SilentlyContinue) {
        scoop install opencode
        if ($LASTEXITCODE -ne 0) { Fail 'scoop could not install OpenCode. See https://opencode.ai/docs, then run this again.' }
    } else {
        Fail 'Install Node.js first (https://nodejs.org, the LTS version), open a NEW PowerShell window, and run this command again.'
    }
}

# ----------------------------------------------------------- 2. browser approval
Step '2/4  Approve this computer in your browser'
$machine = ($env:COMPUTERNAME -replace '[^A-Za-z0-9._ -]', '')
if (-not $machine) { $machine = 'my computer' }
try {
    $start = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/code/device/start" -ContentType 'application/json' -Body (@{ machine = $machine } | ConvertTo-Json) -TimeoutSec 30
} catch {
    $detail = $null
    try { $detail = ($_.ErrorDetails.Message | ConvertFrom-Json).message } catch {}
    if ($detail) { Fail $detail } else { Fail "Could not reach $BaseUrl. Check your internet connection and try again." }
}

Say ''
Say "  Your code:  $($start.user_code)" 'White'
Say '  Open this link, check the code matches, and click Approve:'
Say "  $($start.verification_url)" 'White'
Say ''
try { Start-Process $start.verification_url } catch {}
Say "Waiting for you to approve (this code expires in $([int]($start.expires_in / 60)) minutes)..."

$apiKey = $null
$deadline = (Get-Date).AddSeconds([int]$start.expires_in)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds ([int]$start.interval)
    try {
        $reply = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/code/device/poll" -ContentType 'application/json' -Body (@{ device_code = $start.device_code } | ConvertTo-Json) -TimeoutSec 30
    } catch { continue }
    if ($reply.status -eq 'approved') { $apiKey = $reply.api_key; break }
    if ($reply.status -eq 'denied')   { Fail 'The request was denied in the browser. Nothing was changed.' }
    if ($reply.status -eq 'expired')   { Fail 'The code expired. Run the setup command again.' }
    if ($reply.status -eq 'invalid')   { Fail 'This setup request is no longer valid. Run the setup command again.' }
}
if (-not $apiKey) { Fail 'Timed out waiting for approval. Run the setup command again.' }
Say 'Approved.' 'Green'

# ------------------------------------------------------------ 3. save the login
Step '3/4  Saving your sign-in'
$home_ = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
$authFile = Join-Path $home_ '.local\share\opencode\auth.json'
$auth = Read-JsonFile $authFile
if ($auth -eq $false) {
    Say "Could not safely edit $authFile. Sign in yourself: run  opencode auth login  -> Other -> vidhyora -> paste this key:" 'Yellow'
    Say "   $apiKey"
} else {
    if ($null -eq $auth) { $auth = [PSCustomObject]@{} }
    Set-Prop $auth 'vidhyora' ([PSCustomObject]@{ type = 'api'; key = $apiKey })
    Write-Utf8 $authFile (($auth | ConvertTo-Json -Depth 10) + "`n")
    Say "Saved your key in $authFile"
}

# ------------------------------------------------------------- 4. the config
Step "4/4  Adding $ModelLabel to OpenCode"
$configFile = Join-Path $home_ '.config\opencode\opencode.json'
$config = Read-JsonFile $configFile
if ($config -eq $false) {
    Say "Your existing $configFile could not be edited automatically (it may contain comments)." 'Yellow'
    Say "Add a provider named vidhyora (npm @ai-sdk/openai-compatible, baseURL $BaseUrl/api/v1/code, model $ModelId) and set model to vidhyora/$ModelId." 'Yellow'
} else {
    if ($null -eq $config) { $config = [PSCustomObject]@{} }
    if (-not $config.PSObject.Properties['$schema']) { Set-Prop $config '$schema' 'https://opencode.ai/config.json' }
    if (-not $config.PSObject.Properties['provider']) { Set-Prop $config 'provider' ([PSCustomObject]@{}) }
    $models = [PSCustomObject]@{}
    Set-Prop $models $ModelId ([PSCustomObject]@{ name = $ModelLabel; limit = [PSCustomObject]@{ context = 128000; output = 8192 } })
    $provider = [PSCustomObject]@{
        npm     = '@ai-sdk/openai-compatible'
        name    = $Brand
        options = [PSCustomObject]@{ baseURL = "$BaseUrl/api/v1/code" }
        models  = $models
    }
    Set-Prop $config.provider 'vidhyora' $provider
    if (-not $config.PSObject.Properties['model']) { Set-Prop $config 'model' "vidhyora/$ModelId" }
    Write-Utf8 $configFile (($config | ConvertTo-Json -Depth 10) + "`n")
    Say "Updated $configFile"
}

Say ''
Say 'All set. Open a terminal in your project folder and run:' 'Green'
Say ''
Say '    opencode' 'White'
Say ''
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    Say "If 'opencode' is not found, open a NEW PowerShell window first (the installer updates your PATH)." 'Yellow'
}
Say "First time in a project? Type /init inside OpenCode. Manage connected computers under Start coding in your $Brand account." 'DarkGray'
