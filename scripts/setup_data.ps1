# setup_data.ps1
# Downloads and prepares the Rakuten datasets for the project

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Rakuten dataset setup starting..."
Write-Host ""

# --------------------------------------------------
# Project paths
# --------------------------------------------------

$ProjectRoot = Split-Path -Parent $PSScriptRoot

			  
$DataFolder     = Join-Path $ProjectRoot "data"
$RawFolder      = Join-Path $DataFolder "raw"
$ImagesFolder   = Join-Path $RawFolder "images"
$DownloadFolder = Join-Path $DataFolder "_downloads"

									
$Folders = @(
    $DataFolder,
    $RawFolder,
    $ImagesFolder,
    $DownloadFolder,
    (Join-Path $DataFolder "processed"),
    (Join-Path $DataFolder "splits")
)

foreach ($folder in $Folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created folder: $folder"
    }
    else {
        Write-Host "Folder already exists: $folder"
    }
}

Write-Host ""
Write-Host "Dataset folder structure is ready."
Write-Host ""

# --------------------------------------------------
# Kaggle helper
# --------------------------------------------------

														 

			   
										

function Invoke-KaggleDownload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetSlug,

        [Parameter(Mandatory = $true)]
        [string]$DownloadPath
    )

    Write-Host "Downloading dataset: $DatasetSlug"
																	   

    $kaggleCmd = Get-Command kaggle -ErrorAction SilentlyContinue

    if ($null -ne $kaggleCmd) {
        Write-Host "Using Kaggle CLI from PATH..."
        & kaggle datasets download -d $DatasetSlug -p $DownloadPath
        return
    }

    Write-Host "Kaggle CLI not found in PATH. Trying uv fallback..."
    & uv run python -c "from kaggle.cli import main; main()" datasets download -d $DatasetSlug -p $DownloadPath
}
					  

# --------------------------------------------------
# Download datasets from Kaggle
# --------------------------------------------------

Write-Host "Starting Kaggle dataset download..."
Write-Host ""

																		  
$ImagesDataset = "arturillenseer/rakuten-product-images-ml"
$CsvDataset    = "arturillenseer/csv-files"

Invoke-KaggleDownload -DatasetSlug $ImagesDataset -DownloadPath $DownloadFolder
Invoke-KaggleDownload -DatasetSlug $CsvDataset -DownloadPath $DownloadFolder
					

Write-Host ""
Write-Host "Kaggle download completed."
Write-Host ""

# --------------------------------------------------
# Extract zip files from data/_downloads
# --------------------------------------------------

Write-Host "Checking whether zip extraction is needed..."

$CsvZipPath = Join-Path $DownloadFolder "csv-files.zip"
$ImagesZipPath = Join-Path $DownloadFolder "rakuten-product-images-ml.zip"

$CsvAlreadyExtracted =
    (Test-Path (Join-Path $DownloadFolder "X_train.csv")) -and
    (Test-Path (Join-Path $DownloadFolder "Y_train.csv")) -and
    (Test-Path (Join-Path $DownloadFolder "X_test.csv"))

$ImagesAlreadyExtracted =
    (Test-Path (Join-Path $DownloadFolder "image_train")) -and
    (Test-Path (Join-Path $DownloadFolder "image_test"))

if ((Test-Path $CsvZipPath) -and (-not $CsvAlreadyExtracted)) {
    Write-Host "Extracting: csv-files.zip"
    Expand-Archive -Path $CsvZipPath -DestinationPath $DownloadFolder -Force
}
elseif (Test-Path $CsvZipPath) {
    Write-Host "Skipping extraction of csv-files.zip (already extracted)."
}
else {
    Write-Host "csv-files.zip not found."
}

if ((Test-Path $ImagesZipPath) -and (-not $ImagesAlreadyExtracted)) {
    Write-Host "Extracting: rakuten-product-images-ml.zip"
    Expand-Archive -Path $ImagesZipPath -DestinationPath $DownloadFolder -Force
}
elseif (Test-Path $ImagesZipPath) {
    Write-Host "Skipping extraction of rakuten-product-images-ml.zip (already extracted)."
}
else {
    Write-Host "rakuten-product-images-ml.zip not found."
}

Write-Host "Zip extraction step finished."

# --------------------------------------------------
# Organize extracted files into data/raw
# --------------------------------------------------

Write-Host "Organizing extracted dataset files..."

$XTrainTarget = Join-Path $RawFolder "X_train.csv"
$YTrainTarget = Join-Path $RawFolder "Y_train.csv"
$XTestTarget  = Join-Path $RawFolder "X_test.csv"

function Find-FirstFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,

        [Parameter(Mandatory = $true)]
        [string]$ExactName
    )

    return Get-ChildItem -Path $RootPath -Recurse -File |
        Where-Object { $_.Name -ieq $ExactName } |
        Select-Object -First 1
}

function Find-FirstDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,

        [Parameter(Mandatory = $true)]
        [string]$ExactName
    )

    return Get-ChildItem -Path $RootPath -Recurse -Directory |
        Where-Object { $_.Name -ieq $ExactName } |
        Select-Object -First 1
}

$xTrainFile = Find-FirstFile -RootPath $DownloadFolder -ExactName "X_train.csv"
$yTrainFile = Find-FirstFile -RootPath $DownloadFolder -ExactName "Y_train.csv"
$xTestFile  = Find-FirstFile -RootPath $DownloadFolder -ExactName "X_test.csv"

if ($null -ne $xTrainFile) {
    Copy-Item -Path $xTrainFile.FullName -Destination $XTrainTarget -Force
    Write-Host "Copied X_train.csv to: $XTrainTarget"
}
else {
    Write-Host "X_train.csv not found."
}

if ($null -ne $yTrainFile) {
    Copy-Item -Path $yTrainFile.FullName -Destination $YTrainTarget -Force
    Write-Host "Copied Y_train.csv to: $YTrainTarget"
}
else {
    Write-Host "Y_train.csv not found."
}

if ($null -ne $xTestFile) {
    Copy-Item -Path $xTestFile.FullName -Destination $XTestTarget -Force
    Write-Host "Copied X_test.csv to: $XTestTarget"
}
else {
    Write-Host "X_test.csv not found."
}

					
$ImageTrainSource = Find-FirstDirectory -RootPath $DownloadFolder -ExactName "image_train"
$ImageTestSource  = Find-FirstDirectory -RootPath $DownloadFolder -ExactName "image_test"

if ($null -ne $ImageTrainSource) {
    $ImageTrainTarget = Join-Path $ImagesFolder "image_train"
    if (-not (Test-Path $ImageTrainTarget)) {
        New-Item -ItemType Directory -Path $ImageTrainTarget | Out-Null
    }
    Copy-Item -Path (Join-Path $ImageTrainSource.FullName "*") -Destination $ImageTrainTarget -Recurse -Force
    Write-Host "Copied image_train to: $ImageTrainTarget"
}
else {
    Write-Host "image_train folder not found."
}

if ($null -ne $ImageTestSource) {
    $ImageTestTarget = Join-Path $ImagesFolder "image_test"
    if (-not (Test-Path $ImageTestTarget)) {
        New-Item -ItemType Directory -Path $ImageTestTarget | Out-Null
    }
    Copy-Item -Path (Join-Path $ImageTestSource.FullName "*") -Destination $ImageTestTarget -Recurse -Force
    Write-Host "Copied image_test to: $ImageTestTarget"
}
else {
    Write-Host "image_test folder not found."
}

Write-Host "Dataset organization step finished."

Write-Host ""
Write-Host "Current data folder structure:"
tree $DataFolder
