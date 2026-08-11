Write-Host "========================================="
Write-Host "   Building PollenCounter Studio .exe    "
Write-Host "========================================="

Write-Host "`n[1/3] Installing PyInstaller..."
.venv\Scripts\pip install pyinstaller

Write-Host "`n[2/3] Compiling Launcher... (This may take 5-10 minutes)"
.venv\Scripts\pyinstaller --noconfirm --onedir --windowed `
    --name "PollenCounterStudio" `
    --hidden-import "scripts.annotate" `
    --hidden-import "scripts.inference" `
    --hidden-import "scripts.compare" `
    --hidden-import "scripts.monitor" `
    --hidden-import "scripts.visualize" `
    --hidden-import "scripts.augment_preview" `
    --hidden-import "scripts.dataset_analytics" `
    --hidden-import "scripts.export" `
    --collect-data "ultralytics" `
    --copy-metadata "torch" `
    --copy-metadata "tqdm" `
    --copy-metadata "ultralytics" `
    launcher.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[3/3] Copying project structure and configs to dist..."
# Create necessary empty folders so the app doesn't crash
New-Item -Path "dist\PollenCounterStudio\pretrained_models" -ItemType Directory -Force | Out-Null
New-Item -Path "dist\PollenCounterStudio\runs\detect" -ItemType Directory -Force | Out-Null

# Copy config and models if they exist
Copy-Item -Path "config" -Destination "dist\PollenCounterStudio\" -Recurse -Force
if (Test-Path "pretrained_models\*") {
    Copy-Item -Path "pretrained_models\*" -Destination "dist\PollenCounterStudio\pretrained_models\" -Force
}
if (Test-Path "runs\detect\*") {
    Write-Host "Copying trained models..."
    Copy-Item -Path "runs\detect\*" -Destination "dist\PollenCounterStudio\runs\detect\" -Recurse -Force
}

Write-Host "`n========================================="
Write-Host " Build Complete!" -ForegroundColor Green
Write-Host " You can run the application by double-clicking:"
Write-Host " dist\PollenCounterStudio\PollenCounterStudio.exe" -ForegroundColor Cyan
Write-Host "========================================="
