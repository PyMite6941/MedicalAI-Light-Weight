<#
.SYNOPSIS
    Set up Google Cloud project for MedicalAI training on Vertex AI
.DESCRIPTION
    Creates/verifies GCP project, enables APIs, creates service account,
    sets up GCS bucket, and builds + pushes the training container.
.EXAMPLE
    .\setup_gcp.ps1 -ProjectId "medicalai-prod" -BucketName "medicalai-data-prod"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,

    [Parameter(Mandatory=$true)]
    [string]$BucketName,

    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",

    [Parameter(Mandatory=$false)]
    [string]$ServiceAccountName = "medicalai-trainer",

    [Parameter(Mandatory=$false)]
    [switch]$SkipApis
)

function Step($Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function RunGcloud($Cmd) {
    Write-Host "  $ gcloud $Cmd" -ForegroundColor Gray
    $result = cmd /c "gcloud $Cmd 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Command failed (exit $LASTEXITCODE): gcloud $Cmd"
        Write-Host ($result -join "`n") -ForegroundColor Yellow
    }
    return $result
}

Write-Host @"

╔══════════════════════════════════════════════════════╗
║   MedicalAI - Google Cloud Setup                    ║
║   Configures Vertex AI for large-scale training     ║
╚══════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

# ── 1. Check gcloud ──
Step "1. Checking gcloud CLI..."
$ver = RunGcloud "--version"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: gcloud CLI not found." -ForegroundColor Red
    Write-Host "Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

$acct = RunGcloud "auth print-access-token --quiet"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not authenticated. Running: gcloud auth login" -ForegroundColor Yellow
    RunGcloud "auth login"
}

Write-Host "  Authenticated as: $(RunGcloud config get-value account 2>$null | Select-Object -Last 1)" -ForegroundColor Green

# ── 2. Create / verify project ──
Step "2. Setting up project: $ProjectId"
RunGcloud "projects describe $ProjectId --quiet"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Project does not exist. Creating..." -ForegroundColor Yellow
    RunGcloud "projects create $ProjectId --name=MedicalAI --quiet"
}
RunGcloud "config set project $ProjectId"

# ── 3. Enable required APIs ──
if (-not $SkipApis) {
    Step "3. Enabling required APIs..."
    $apis = @(
        "aiplatform.googleapis.com",
        "containerregistry.googleapis.com",
        "artifactregistry.googleapis.com",
        "storage.googleapis.com",
        "cloudbuild.googleapis.com",
        "iam.googleapis.com"
    )
    foreach ($api in $apis) {
        Write-Host "  Enabling $api..." -ForegroundColor Gray
        RunGcloud "services enable $api --quiet"
    }
    Write-Host "  APIs enabled." -ForegroundColor Green
}

# ── 4. Create Artifact Registry repo ──
Step "4. Creating Artifact Registry repository..."
$repoExists = RunGcloud "artifacts repositories describe medicalai --location=$Region --quiet 2>`$null"
if ($LASTEXITCODE -ne 0) {
    RunGcloud "artifacts repositories create medicalai --repository-format=docker --location=$Region --quiet"
    Write-Host "  Repository created: $Region-docker.pkg.dev/$ProjectId/medicalai" -ForegroundColor Green
} else {
    Write-Host "  Repository already exists." -ForegroundColor Green
}

# ── 5. Create GCS bucket ──
Step "5. Creating GCS bucket: gs://$BucketName"
$bucketExists = RunGcloud "storage buckets describe gs://$BucketName --quiet 2>`$null"
if ($LASTEXITCODE -ne 0) {
    RunGcloud "storage buckets create gs://$BucketName --location=$Region --uniform-bucket-level-access --quiet"
    Write-Host "  Bucket created." -ForegroundColor Green
} else {
    Write-Host "  Bucket already exists." -ForegroundColor Green
}

# ── 6. Create service account ──
Step "6. Creating service account: $ServiceAccountName@$ProjectId.iam.gserviceaccount.com"
$saEmail = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"
$saExists = RunGcloud "iam service-accounts describe $saEmail --quiet 2>`$null"
if ($LASTEXITCODE -ne 0) {
    RunGcloud "iam service-accounts create $ServiceAccountName --display-name='MedicalAI Trainer' --quiet"
    Write-Host "  Service account created." -ForegroundColor Green
} else {
    Write-Host "  Service account already exists." -ForegroundColor Green
}

# ── 7. Grant IAM roles ──
Step "7. Granting IAM roles..."
$roles = @(
    "roles/aiplatform.user",
    "roles/artifactregistry.reader",
    "roles/artifactregistry.writer",
    "roles/storage.objectAdmin",
    "roles/iam.serviceAccountUser"
)
foreach ($role in $roles) {
    RunGcloud "projects add-iam-policy-binding $ProjectId --member=serviceAccount:$saEmail --role=$role --quiet"
    Write-Host "  Granted $role" -ForegroundColor Gray
}

# Also grant the compute engine default SA access to read from GCS
$computeSa = "$(RunGcloud projects describe $ProjectId --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
RunGcloud "storage buckets add-iam-policy-binding gs://$BucketName --member=serviceAccount:$computeSa --role=roles/storage.objectAdmin --quiet"
RunGcloud "storage buckets add-iam-policy-binding gs://$BucketName --member=serviceAccount:$saEmail --role=roles/storage.objectAdmin --quiet"

Write-Host "  IAM roles granted." -ForegroundColor Green

# ── 8. Create GCS folder structure ──
Step "8. Creating GCS folder structure..."
$folders = @("data/", "data/images/", "models/", "logs/")
foreach ($folder in $folders) {
    $null > "C:\Temp\gcs_placeholder.txt"
    RunGcloud "storage cp C:\Temp\gcs_placeholder.txt gs://$BucketName/$folder`placeholder.txt --quiet"
}
Write-Host "  Folder structure created." -ForegroundColor Green

# ── 9. Summary ──
Write-Host @"

╔══════════════════════════════════════════════════════╗
║   Setup Complete!                                   ║
╚══════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "  Project:           $ProjectId" -ForegroundColor White
Write-Host "  Region:            $Region" -ForegroundColor White
Write-Host "  Bucket:            gs://$BucketName" -ForegroundColor White
Write-Host "  SA Email:          $saEmail" -ForegroundColor White
Write-Host "  Registry:          $Region-docker.pkg.dev/$ProjectId/medicalai" -ForegroundColor White

Write-Host @"

  Next steps:

  1. Upload your expanded data to GCS:
     gsutil cp ./data/dataset.csv gs://$BucketName/data/
     gsutil -m cp ./data/images/*.jpg gs://$BucketName/data/images/

  2. Build and submit the training job:
     python cloud/submit_job.py ^
         --project $ProjectId ^
         --bucket $BucketName ^
         --service-account $saEmail ^
         --region $Region ^
         --epochs 30 ^
         --batch-size 64 ^
         --accelerator NVIDIA_TESLA_A100 ^
         --accelerator-count 4

  3. Monitor in console:
     https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$ProjectId

  4. Download trained model:
     gsutil cp -r gs://$BucketName/models/ ./checkpoints/vertex_models/

"@ -ForegroundColor Cyan