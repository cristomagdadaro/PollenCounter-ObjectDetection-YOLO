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
    --hidden-import "sahi" `
    --hidden-import "sahi.predict" `
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

# Copy README
if (Test-Path "README.md") {
    Copy-Item "README.md" "dist\PollenCounterStudio\" -Force
}

# Copy config and pretrained models
Copy-Item -Path "config" -Destination "dist\PollenCounterStudio\" -Recurse -Force
if (Test-Path "pretrained_models\*") {
    Copy-Item -Path "pretrained_models\*" -Destination "dist\PollenCounterStudio\pretrained_models\" -Recurse -Force
}

# Parse inference_settings.json to find the active model
$settings_path = "config\inference_settings.json"
if (Test-Path $settings_path) {
    Write-Host "Parsing active model from settings..."
    $settings = Get-Content -Raw $settings_path | ConvertFrom-Json
    $active_model = $settings.weights
    
    if ($active_model) {
        $model_path = "runs\detect\$active_model"
        if (Test-Path $model_path) {
            Write-Host "Copying active model: $active_model"
            # Copy just this model's folder recursively to get all exports (pt, onnx, engine, etc.)
            Copy-Item -Path $model_path -Destination "dist\PollenCounterStudio\runs\detect\$active_model" -Recurse -Force
        } else {
            Write-Host "Warning: Active model '$active_model' not found in runs/detect/" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Warning: config/inference_settings.json not found." -ForegroundColor Yellow
}

Write-Host "`n========================================="
Write-Host " Build Complete!" -ForegroundColor Green
Write-Host " You can run the application by double-clicking:"
Write-Host " dist\PollenCounterStudio\PollenCounterStudio.exe" -ForegroundColor Cyan
Write-Host "========================================="
