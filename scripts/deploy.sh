#!/usr/bin/env bash
# ============================================================================
# ARCFACE ENGINE — AUTOMATED DEPLOYMENT & SCORCHED-EARTH MIGRATION
# ============================================================================
# Project:  hoppwhistle
# Region:   us-east4
# Purpose:  Deploy TF-scaled backend, wipe legacy 1404-D vault,
#           re-ingest with 512-D ArcFace embeddings
#
# EVERYTHING RUNS IN THE CLOUD. No local tooling required.
# ============================================================================
set -e

PROJECT_ID="hoppwhistle"
REGION="us-east4"
SERVICE_NAME="facial-backend"
BACKEND_IMAGE="us-east4-docker.pkg.dev/hoppwhistle/facial-app-repo/facial-backend:latest"
SERVICE_ACCOUNT="facial-runtime-sa@hoppwhistle.iam.gserviceaccount.com"
CLOUD_SQL_INSTANCE="hoppwhistle:us-central1:facial-pg-instance"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  ARCFACE ENGINE — DEPLOYMENT & SCORCHED-EARTH MIGRATION"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── PHASE 1: Deploy backend with TF-scaled resources ──
echo "▶ PHASE 1: DEPLOYING ARCFACE-SCALED BACKEND TO CLOUD RUN"
echo "  Memory: 8Gi  |  CPU: 4  |  TensorFlow + ArcFace"
echo ""

gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${BACKEND_IMAGE}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --memory=8Gi \
  --cpu=4 \
  --timeout=300 \
  --concurrency=10 \
  --allow-unauthenticated \
  --add-cloudsql-instances="${CLOUD_SQL_INSTANCE}" \
  --set-env-vars="KMS_KEY_NAME=projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek,BUCKET_NAME=hoppwhistle-facial-uploads,DB_USER=facial_app,DB_PASS=Fv966468!Sec5677,DB_NAME=facial_db,CLOUD_SQL_CONNECTION_NAME=${CLOUD_SQL_INSTANCE}"

echo ""
echo "✓ DEPLOYMENT SUCCESSFUL"

# ── PHASE 2: Safety guardrail ──
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  WARNING: Proceeding will PERMANENTLY DESTROY the legacy"
echo "  1404-D production vault. This action is IRREVERSIBLE."
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo -n "Type 'YES' to trigger scorched-earth migration: "
read -r CONFIRM

if [ "${CONFIRM}" != "YES" ]; then
  echo ""
  echo "⚠ Migration aborted. Database untouched."
  exit 0
fi

# ── PHASE 3 + 4: Trigger migration Cloud Run Job (truncation + re-ingestion) ──
echo ""
echo "▶ TRIGGERING ARCFACE MIGRATION JOB (Cloud Run)..."
echo "  This runs REMOTELY — truncation + ArcFace re-ingestion in the cloud."
echo ""

gcloud run jobs execute arcface-migration-job \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --wait

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  ✓ DEPLOYMENT & MIGRATION COMPLETE"
echo "══════════════════════════════════════════════════════════════════"
echo ""
