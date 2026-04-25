# ============================================================================
# FACIAL VERIFICATION - GCP INFRASTRUCTURE PROVISIONING
# ============================================================================
# Project:  hoppwhistle
# Region:   us-central1
# Purpose:  Production-ready biometric facial verification backend
# ============================================================================

$ErrorActionPreference = "Continue"

# --- Configuration ---
$PROJECT_ID       = "hoppwhistle"
$REGION           = "us-central1"
$ZONE             = "us-central1-a"
$APP_NAME         = "facial-verify"

# Resource Names
$BUCKET_NAME      = "${PROJECT_ID}-facial-uploads"
$SQL_INSTANCE     = "facial-pg-instance"
$SQL_DB_NAME      = "facial_db"
$SQL_USER         = "facial_app"
$SQL_PASSWORD     = "Fv$(Get-Random -Minimum 100000 -Maximum 999999)!Sec$(Get-Random -Minimum 1000 -Maximum 9999)"
$REGISTRY_NAME    = "facial-docker-repo"
$CLOUD_RUN_SVC    = "facial-verify-api"
$SA_NAME          = "facial-runtime-sa"
$SA_EMAIL         = "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GCP Infrastructure Provisioning Starting  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project:  $PROJECT_ID"
Write-Host "Region:   $REGION"
Write-Host "App:      $APP_NAME"
Write-Host ""

# ============================================================================
# 1. GOOGLE CLOUD STORAGE - Secure Upload Bucket
# ============================================================================
Write-Host ">>> [1/6] Creating GCS Bucket: $BUCKET_NAME" -ForegroundColor Yellow

gcloud storage buckets create "gs://${BUCKET_NAME}" `
    --project=$PROJECT_ID `
    --location=$REGION `
    --uniform-bucket-level-access `
    --public-access-prevention `
    --default-storage-class=STANDARD `
    2>&1

# Set lifecycle policy: auto-delete raw uploads after 7 days
$lifecycleJson = @"
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 7}
    }
  ]
}
"@
$lifecycleFile = "$env:TEMP\gcs_lifecycle.json"
$lifecycleJson | Out-File -FilePath $lifecycleFile -Encoding utf8 -Force
gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=$lifecycleFile 2>&1

# Enable versioning for audit trail
gcloud storage buckets update "gs://${BUCKET_NAME}" --versioning 2>&1

Write-Host "<<< GCS Bucket created successfully." -ForegroundColor Green
Write-Host ""

# ============================================================================
# 2. CLOUD SQL - PostgreSQL with pgvector
# ============================================================================
Write-Host ">>> [2/6] Creating Cloud SQL Instance: $SQL_INSTANCE" -ForegroundColor Yellow
Write-Host "    (This will take 5-10 minutes...)" -ForegroundColor DarkGray

gcloud sql instances create $SQL_INSTANCE `
    --project=$PROJECT_ID `
    --database-version=POSTGRES_15 `
    --tier=db-custom-2-8192 `
    --region=$REGION `
    --storage-size=20GB `
    --storage-auto-increase `
    --availability-type=zonal `
    --backup `
    --backup-start-time=03:00 `
    --database-flags=cloudsql.enable_pgaudit=on,max_connections=100 `
    --insights-config-query-insights-enabled `
    --maintenance-window-day=SUN `
    --maintenance-window-hour=4 `
    --deletion-protection `
    2>&1

Write-Host "    Creating database: $SQL_DB_NAME" -ForegroundColor DarkGray
gcloud sql databases create $SQL_DB_NAME `
    --instance=$SQL_INSTANCE `
    --project=$PROJECT_ID `
    2>&1

Write-Host "    Creating user: $SQL_USER" -ForegroundColor DarkGray
gcloud sql users create $SQL_USER `
    --instance=$SQL_INSTANCE `
    --project=$PROJECT_ID `
    --password=$SQL_PASSWORD `
    2>&1

Write-Host "<<< Cloud SQL PostgreSQL instance created." -ForegroundColor Green
Write-Host "    NOTE: pgvector extension will be enabled via SQL after first connection." -ForegroundColor DarkGray
Write-Host "    Run: CREATE EXTENSION IF NOT EXISTS vector;" -ForegroundColor DarkGray
Write-Host ""

# ============================================================================
# 3. ARTIFACT REGISTRY - Docker Repository
# ============================================================================
Write-Host ">>> [3/6] Creating Artifact Registry: $REGISTRY_NAME" -ForegroundColor Yellow

gcloud artifacts repositories create $REGISTRY_NAME `
    --project=$PROJECT_ID `
    --repository-format=docker `
    --location=$REGION `
    --description="Facial verification CV/ML Docker images" `
    --immutable-tags `
    2>&1

Write-Host "<<< Artifact Registry created." -ForegroundColor Green
Write-Host ""

# ============================================================================
# 4. IAM - Runtime Service Account
# ============================================================================
Write-Host ">>> [4/6] Creating Runtime Service Account: $SA_NAME" -ForegroundColor Yellow

gcloud iam service-accounts create $SA_NAME `
    --project=$PROJECT_ID `
    --display-name="Facial Verify Cloud Run Runtime SA" `
    --description="Dedicated SA for facial verification Cloud Run service" `
    2>&1

# Wait for SA propagation
Start-Sleep -Seconds 5

Write-Host "<<< Service Account created: $SA_EMAIL" -ForegroundColor Green
Write-Host ""

# ============================================================================
# 5. IAM ROLE BINDINGS
# ============================================================================
Write-Host ">>> [5/6] Binding IAM Roles to $SA_EMAIL" -ForegroundColor Yellow

# GCS: Read/Write to the upload bucket
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/storage.objectAdmin" `
    --condition="expression=resource.name.startsWith('projects/_/buckets/${BUCKET_NAME}'),title=facial-bucket-access,description=Restrict to facial upload bucket" `
    2>&1

# Cloud SQL: Client connect
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/cloudsql.client" `
    2>&1

# Cloud Run: Invoker (for authenticated invocations)
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/run.invoker" `
    2>&1

# Artifact Registry: Reader (pull images)
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/artifactregistry.reader" `
    2>&1

# Logging: Write logs
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/logging.logWriter" `
    2>&1

# Monitoring: Write metrics
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:${SA_EMAIL}" `
    --role="roles/monitoring.metricWriter" `
    2>&1

Write-Host "<<< IAM bindings complete." -ForegroundColor Green
Write-Host ""

# ============================================================================
# 6. CLOUD RUN - Placeholder Service
# ============================================================================
Write-Host ">>> [6/6] Deploying Cloud Run Placeholder: $CLOUD_RUN_SVC" -ForegroundColor Yellow

$SQL_CONNECTION_NAME = gcloud sql instances describe $SQL_INSTANCE --project=$PROJECT_ID --format="value(connectionName)" 2>$null

gcloud run deploy $CLOUD_RUN_SVC `
    --project=$PROJECT_ID `
    --region=$REGION `
    --image=us-docker.pkg.dev/cloudrun/container/hello `
    --service-account=$SA_EMAIL `
    --memory=4Gi `
    --cpu=2 `
    --min-instances=0 `
    --max-instances=10 `
    --timeout=300 `
    --concurrency=10 `
    --port=8080 `
    --no-allow-unauthenticated `
    --set-env-vars="GCS_BUCKET=${BUCKET_NAME},DB_NAME=${SQL_DB_NAME},DB_USER=${SQL_USER},PROJECT_ID=${PROJECT_ID}" `
    --add-cloudsql-instances="${SQL_CONNECTION_NAME}" `
    2>&1

Write-Host "<<< Cloud Run placeholder deployed." -ForegroundColor Green
Write-Host ""

# ============================================================================
# OUTPUT - Critical Environment Variables
# ============================================================================
$REGISTRY_URL = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY_NAME}"
$DB_CONNECTION_STRING = "postgresql://${SQL_USER}:${SQL_PASSWORD}@/${SQL_DB_NAME}?host=/cloudsql/${SQL_CONNECTION_NAME}"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  INFRASTRUCTURE PROVISIONING COMPLETE      " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "--- Critical Environment Variables ---" -ForegroundColor Yellow
Write-Host ""
Write-Host "GCS_BUCKET_NAME       = $BUCKET_NAME" -ForegroundColor White
Write-Host "GCS_BUCKET_URI        = gs://${BUCKET_NAME}" -ForegroundColor White
Write-Host ""
Write-Host "SQL_INSTANCE          = $SQL_INSTANCE" -ForegroundColor White
Write-Host "SQL_CONNECTION_NAME   = $SQL_CONNECTION_NAME" -ForegroundColor White
Write-Host "SQL_DB_NAME           = $SQL_DB_NAME" -ForegroundColor White
Write-Host "SQL_USER              = $SQL_USER" -ForegroundColor White
Write-Host "SQL_PASSWORD          = $SQL_PASSWORD" -ForegroundColor White
Write-Host "DB_CONNECTION_STRING  = $DB_CONNECTION_STRING" -ForegroundColor White
Write-Host ""
Write-Host "ARTIFACT_REGISTRY_URL = $REGISTRY_URL" -ForegroundColor White
Write-Host "DOCKER_TAG_CMD        = docker tag <image> ${REGISTRY_URL}/<image>:<tag>" -ForegroundColor DarkGray
Write-Host "DOCKER_PUSH_CMD       = docker push ${REGISTRY_URL}/<image>:<tag>" -ForegroundColor DarkGray
Write-Host ""
Write-Host "CLOUD_RUN_SERVICE     = $CLOUD_RUN_SVC" -ForegroundColor White
Write-Host "CLOUD_RUN_REGION      = $REGION" -ForegroundColor White
Write-Host "RUNTIME_SA            = $SA_EMAIL" -ForegroundColor White
Write-Host ""
Write-Host "--- pgvector Setup (run after first connection) ---" -ForegroundColor Yellow
Write-Host "  gcloud sql connect $SQL_INSTANCE --user=$SQL_USER --database=$SQL_DB_NAME" -ForegroundColor DarkGray
Write-Host "  SQL> CREATE EXTENSION IF NOT EXISTS vector;" -ForegroundColor DarkGray
Write-Host ""
Write-Host "--- Docker Auth for Artifact Registry ---" -ForegroundColor Yellow
Write-Host "  gcloud auth configure-docker ${REGION}-docker.pkg.dev" -ForegroundColor DarkGray
Write-Host ""

# Save env vars to file for application use
$envContent = @"
# ============================================================================
# FACIAL VERIFICATION - GCP Environment Variables
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
# ============================================================================

# Google Cloud
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION

# Cloud Storage
GCS_BUCKET_NAME=$BUCKET_NAME

# Cloud SQL (PostgreSQL + pgvector)
SQL_INSTANCE_NAME=$SQL_INSTANCE
SQL_CONNECTION_NAME=$SQL_CONNECTION_NAME
DB_NAME=$SQL_DB_NAME
DB_USER=$SQL_USER
DB_PASSWORD=$SQL_PASSWORD
DB_CONNECTION_STRING=$DB_CONNECTION_STRING

# Artifact Registry
ARTIFACT_REGISTRY_URL=$REGISTRY_URL

# Cloud Run
CLOUD_RUN_SERVICE=$CLOUD_RUN_SVC
CLOUD_RUN_REGION=$REGION

# IAM
RUNTIME_SERVICE_ACCOUNT=$SA_EMAIL
"@

$envContent | Out-File -FilePath "c:\Users\jimbo\OneDrive\Documents\facial\.env.gcp" -Encoding utf8 -Force
Write-Host "Environment variables saved to: .env.gcp" -ForegroundColor Green
Write-Host ""
