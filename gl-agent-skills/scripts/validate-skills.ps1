[CmdletBinding()]
param(
    [string]$Root
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $PSScriptRoot
}

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$errors = [System.Collections.Generic.List[string]]::new()
$names = @{}
$skills = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter SKILL.md |
    Where-Object { $_.FullName -notmatch '[\\/]scripts[\\/]' }

if ($skills.Count -eq 0) {
    $errors.Add('No SKILL.md files found.')
}

foreach ($skill in $skills) {
    $relative = $skill.FullName.Substring($rootPath.Length + 1)
    $body = [System.IO.File]::ReadAllText($skill.FullName)

    if ($body -notmatch '(?s)\A---\r?\n(?<frontmatter>.*?)\r?\n---\r?\n' ) {
        $errors.Add("${relative}: missing YAML frontmatter.")
        continue
    }

    $frontmatter = $Matches.frontmatter
    $nameMatch = [regex]::Match($frontmatter, '(?m)^name:\s*(?<name>[a-z0-9-]+)\s*$')
    $descriptionMatch = [regex]::Match($frontmatter, '(?m)^description:\s*.+')

    if (-not $nameMatch.Success) {
        $errors.Add("${relative}: missing or invalid name (use lowercase letters, digits, and hyphens).")
    }
    else {
        $name = $nameMatch.Groups['name'].Value
        if ($names.ContainsKey($name)) {
            $errors.Add("${relative}: duplicate skill name '$name' (also $($names[$name])).")
        }
        else {
            $names[$name] = $relative
        }
    }

    if (-not $descriptionMatch.Success) {
        $errors.Add("${relative}: missing description.")
    }

    foreach ($match in [regex]::Matches($body, '\[[^\]]*\]\((?<target>[^)#]+)(?:#[^)]*)?\)')) {
        $target = $match.Groups['target'].Value
        if ($target -match '^(https?://|mailto:)') {
            continue
        }

        $destination = [System.IO.Path]::GetFullPath((Join-Path $skill.DirectoryName ([uri]::UnescapeDataString($target))))
        if (-not (Test-Path -LiteralPath $destination)) {
            $errors.Add("${relative}: broken local link '$target'.")
        }
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output "Validated $($skills.Count) skills and $($names.Count) unique names."
