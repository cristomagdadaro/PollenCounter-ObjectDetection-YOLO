Add-Type -AssemblyName System.IO.Compression.FileSystem

# Identify latest run folder to include
$latestRun = Get-ChildItem -Path "$PWD\runs\detect" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$latestRunName = if ($latestRun) { $latestRun.Name } else { "PollenCounter-Release" }
Write-Host "Latest model run detected as: $latestRunName"

# Set the zip file name to match the latest model folder
$zipPath = "$PWD\$latestRunName.zip"
if (Test-Path $zipPath) { 
    try { Remove-Item $zipPath -Force } catch {}
}

Write-Host "Creating archive $zipPath..."
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')

# Get tracked files and un-ignored untracked files from git
$files = git ls-files
$files += git ls-files --others --exclude-standard

$count = 0
foreach ($file in $files) {
    $file = $file.Replace('\', '/')
    
    # 1. Skip .zip files (to prevent recursive locking)
    if ($file -match "\.zip$") { continue }
    
    # 2. Skip the run.ps1 script itself
    if ($file -match "run\.ps1$") { continue }
    
    # 3. Exclude the heavy datasets folder
    if ($file -match "^datasets/") { continue }
    
    # 4. Exclude all runs EXCEPT the latest one
    if ($file -match "^runs/detect/") {
        if ($latestRunName -and ($file -notmatch "^runs/detect/$latestRunName/")) {
            continue
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($file)) {
        if (Test-Path $file -PathType Leaf) {
            # Use the .NET CreateEntryFromFile method to preserve the exact folder structure in the zip
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, "$PWD\$file", $file)
            $count++
        }
    }
}

$zip.Dispose()
Write-Host "Successfully compressed $count files into $zipPath!"
Write-Host "(Excluded datasets, older model runs, and run.ps1)"
