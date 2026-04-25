$PROJECT_ID = "hoppwhistle"
$REGION = "us-central1"
$SA_KEY_FILE = "C:\Users\jimbo\OneDrive\Documents\facial\hoppwhistle-289819c0e326.json"

Write-Host "Authenticating with Service Account..."
gcloud auth activate-service-account --key-file=$SA_KEY_FILE
gcloud config set project $PROJECT_ID

Write-Host "Enabling APIs..."
gcloud services enable sqladmin.googleapis.com artifactregistry.googleapis.com run.googleapis.com iam.googleapis.com cloudresourcemanager.googleapis.com

# 1. GCS Bucket
$BUCKET_NAME = "hoppwhistle-facial-raw-images-bucket"
Write-Host "Creating GCS Bucket..."
try {
    $bucket_exists = gcloud storage buckets describe gs://$BUCKET_NAME --project=$PROJECT_ID 2>$null
    if ($LASTEXITCODE -ne 0) {
        gcloud storage buckets create gs://$BUCKET_NAME --location=US --project=$PROJECT_ID
    } else {
        Write-Host "Bucket already exists."
    }
} catch {
    Write-Host "Creating Bucket..."
    gcloud storage buckets create gs://$BUCKET_NAME --location=US --project=$PROJECT_ID
}

# 2. Cloud SQL
$DB_INSTANCE = "facial-db-instance"
$DB_NAME = "facial_db"
$DB_USER = "facial_app_user"
$DB_PASS = "SuperSecretPassword123!"

Write-Host "Creating Cloud SQL Instance..."
# Check if instance exists to avoid error
$instance_exists = gcloud sql instances describe $DB_INSTANCE --project=$PROJECT_ID --format="value(name)" 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud sql instances create $DB_INSTANCE --database-version=POSTGRES_15 --region=$REGION --tier=db-f1-micro --database-flags=cloudsql.enable_pgvector=on --project=$PROJECT_ID
} else {
    Write-Host "Cloud SQL Instance already exists."
}

Write-Host "Creating Database..."
$db_exists = gcloud sql databases describe $DB_NAME --instance=$DB_INSTANCE --project=$PROJECT_ID --format="value(name)" 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud sql databases create $DB_NAME --instance=$DB_INSTANCE --project=$PROJECT_ID
} else {
    Write-Host "Database already exists."
}

Write-Host "Creating Database User..."
$user_exists = gcloud sql users describe $DB_USER --instance=$DB_INSTANCE --project=$PROJECT_ID 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud sql users create $DB_USER --instance=$DB_INSTANCE --password=$DB_PASS --project=$PROJECT_ID
} else {
    Write-Host "Database User already exists."
}

# 3. Artifact Registry
$REPO_NAME = "facial-app-repo"
Write-Host "Creating Artifact Registry..."
$repo_exists = gcloud artifacts repositories describe $REPO_NAME --location=$REGION --project=$PROJECT_ID --format="value(name)" 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repository for facial verification app" --project=$PROJECT_ID
} else {
    Write-Host "Artifact Registry already exists."
}

# 4. IAM Service Account
$SA_NAME = "facial-cloudrun-sa"
$SA_EMAIL = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

Write-Host "Creating Service Account..."
$sa_exists = gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID --format="value(email)" 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud iam service-accounts create $SA_NAME --display-name="Cloud Run Service Account for Facial App" --project=$PROJECT_ID
} else {
    Write-Host "Service Account already exists."
}

Write-Host "Adding IAM Policy Bindings..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/cloudsql.client"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/run.invoker"

# 5. Cloud Run Placeholder
$RUN_SERVICE = "facial-app-service"
Write-Host "Deploying Cloud Run Placeholder..."
gcloud run deploy $RUN_SERVICE --image="us-docker.pkg.dev/cloudrun/container/hello" --region=$REGION --memory="4Gi" --service-account=$SA_EMAIL --project=$PROJECT_ID --allow-unauthenticated

Write-Host "`n--- Deployment Complete ---"
Write-Host "Bucket Name: $BUCKET_NAME"
$DB_CONN = gcloud sql instances describe $DB_INSTANCE --project=$PROJECT_ID --format="value(connectionName)"
Write-Host "DB Connection String: $DB_CONN"
Write-Host "DB User: $DB_USER"
Write-Host "DB Password: $DB_PASS"
Write-Host "Artifact Registry URL: $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
$RUN_URL = gcloud run services describe $RUN_SERVICE --region=$REGION --project=$PROJECT_ID --format="value(status.url)"
Write-Host "Cloud Run URL: $RUN_URL"
