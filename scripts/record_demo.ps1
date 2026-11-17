# Record one demonstration match: you (blue, human) vs BotCore (orange, deployed bot/model, its default sampled mode).
# Thin wrapper around scripts/live_check.py so every match lands in its own folder and nothing is overwritten.
#
#   .\scripts\record_demo.ps1 -Match 1            # -> runs\demos\s1\m01
#   .\scripts\record_demo.ps1 -Match 2 -Session s1
#
# Afterwards build the dataset (kickoffs and unreliable segments are excluded there, not during play):
#   .\.venv\Scripts\python.exe scripts\il_feasibility\build_demos.py runs\demos\s1.npz (Get-ChildItem runs\demos\s1\m*\*.bcp)
param(
    [Parameter(Mandatory = $true)][int]$Match,
    [string]$Session = "s1"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = "runs\demos\{0}\m{1:D2}" -f $Session, $Match
if (Test-Path $out) { throw "$out already exists; pick another -Match number" }
if ($env:BOTCORE_DETERMINISTIC) { Write-Warning "BOTCORE_DETERMINISTIC is set ($env:BOTCORE_DETERMINISTIC): the bot will use argmax" }
& .\.venv\Scripts\python.exe scripts\live_check.py --match bot\match_human_vs_bot.toml --out $out --save-replay --max-minutes 12
if ($LASTEXITCODE -ne 0) { throw "live_check.py exited with $LASTEXITCODE" }
Write-Host "recorded $out"
