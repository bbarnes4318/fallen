# Fallen

## Architecture
- **Frontend:** The production Next.js frontend lives in the `frontend/` directory.
- **Backend:** The FastAPI backend lives in the `backend/` directory.
- **Scripts:** Data ingestion, identity graph generation, and pipeline scripts live in the `scripts/` directory.

## Deployment
Deployment is handled through a Cloud Run workflow.
Required environment variables (e.g., database credentials, KMS keys, bucket names) must be supplied through GitHub/GCP secrets.

## Important Note
Do not edit files in `archive/prototypes` for production behavior. The archived prototype files are strictly for historical reference.
