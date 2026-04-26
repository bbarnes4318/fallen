#!/usr/bin/env bash
# ============================================================================
# ARCFACE ENGINE — AUTOMATED DEPLOYMENT & SCORCHED-EARTH MIGRATION
# ============================================================================
# Project:  hoppwhistle
# Region:   us-east4
# Purpose:  Deploy TF-scaled backend, wipe legacy 1404-D vault,
#           re-ingest with 512-D ArcFace embeddings
# ============================================================================
set -e

# ── ANSI Colors ──
GOLD="\033[38;2;212;175;55m"
GREEN="\033[38;2;0;255;128m"
RED="\033[38;2;255;60;60m"
YELLOW="\033[38;2;255;200;60m"
CYAN="\033[38;2;80;200;255m"
DIM="\033[2m"
BOLD="\033[1m"
RESET="\033[0m"

# ── Configuration (sourced from .env.gcp + deploy.yml) ──
PROJECT_ID="hoppwhistle"
REGION="us-east4"
SERVICE_NAME="facial-backend"
BACKEND_IMAGE="us-east4-docker.pkg.dev/hoppwhistle/facial-app-repo/facial-backend:latest"
SERVICE_ACCOUNT="facial-runtime-sa@hoppwhistle.iam.gserviceaccount.com"
CLOUD_SQL_INSTANCE="hoppwhistle:us-central1:facial-pg-instance"

# Database connection (Cloud SQL Auth Proxy must be running for local execution)
DB_USER="facial_app"
DB_PASS="Fv966468!Sec5677"
DB_NAME="facial_db"
DB_HOST="127.0.0.1"
DB_PORT="5432"

echo -e ""
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GOLD}  ██████╗ ███████╗██████╗ ██╗      ██████╗ ██╗   ██╗            ${RESET}"
echo -e "${GOLD}  ██╔══██╗██╔════╝██╔══██╗██║     ██╔═══██╗╚██╗ ██╔╝            ${RESET}"
echo -e "${GOLD}  ██║  ██║█████╗  ██████╔╝██║     ██║   ██║ ╚████╔╝             ${RESET}"
echo -e "${GOLD}  ██║  ██║██╔══╝  ██╔═══╝ ██║     ██║   ██║  ╚██╔╝              ${RESET}"
echo -e "${GOLD}  ██████╔╝███████╗██║     ███████╗╚██████╔╝   ██║               ${RESET}"
echo -e "${GOLD}  ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝    ╚═╝               ${RESET}"
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  ARCFACE ENGINE — DEPLOYMENT & SCORCHED-EARTH MIGRATION${RESET}"
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e ""
echo -e "  ${CYAN}PROJECT${RESET}      ${PROJECT_ID}"
echo -e "  ${CYAN}REGION${RESET}       ${REGION}"
echo -e "  ${CYAN}SERVICE${RESET}      ${SERVICE_NAME}"
echo -e "  ${CYAN}IMAGE${RESET}        ${BACKEND_IMAGE}"
echo -e "  ${CYAN}MEMORY${RESET}       8Gi (TF/ArcFace)"
echo -e "  ${CYAN}CPU${RESET}          4 vCPUs"
echo -e ""
echo -e "${DIM}──────────────────────────────────────────────────────────────────${RESET}"

# ============================================================================
# PHASE 1: INFRASTRUCTURE SCALING & DEPLOYMENT
# ============================================================================
echo -e ""
echo -e "${YELLOW}  ▶ PHASE 1: DEPLOYING ARCFACE-SCALED BACKEND TO CLOUD RUN${RESET}"
echo -e "${DIM}    Memory: 4Gi → 8Gi  |  CPU: 2 → 4  |  TensorFlow + ArcFace${RESET}"
echo -e ""

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
  --set-env-vars="KMS_KEY_NAME=projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek,BUCKET_NAME=hoppwhistle-facial-uploads,DB_USER=${DB_USER},DB_PASS=${DB_PASS},DB_NAME=${DB_NAME},CLOUD_SQL_CONNECTION_NAME=${CLOUD_SQL_INSTANCE}"

echo -e ""
echo -e "${GREEN}  ✓ DEPLOYMENT SUCCESSFUL${RESET}"
echo -e "${DIM}──────────────────────────────────────────────────────────────────${RESET}"

# ============================================================================
# PHASE 2: SAFETY GUARDRAIL — OPERATOR CONFIRMATION
# ============================================================================
echo -e ""
echo -e "${RED}  ╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${RED}  ║  ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████   ║${RESET}"
echo -e "${RED}  ║  ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██        ║${RESET}"
echo -e "${RED}  ║  ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███  ║${RESET}"
echo -e "${RED}  ║  ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██  ║${RESET}"
echo -e "${RED}  ║   ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████   ║${RESET}"
echo -e "${RED}  ╠══════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${RED}  ║  Proceeding will PERMANENTLY DESTROY the legacy 1404-D      ║${RESET}"
echo -e "${RED}  ║  production vault. This action is IRREVERSIBLE.             ║${RESET}"
echo -e "${RED}  ║                                                              ║${RESET}"
echo -e "${RED}  ║  All existing identity_profiles rows will be TRUNCATED.     ║${RESET}"
echo -e "${RED}  ║  Fresh 512-D ArcFace embeddings will be re-ingested.        ║${RESET}"
echo -e "${RED}  ╚══════════════════════════════════════════════════════════════╝${RESET}"
echo -e ""
echo -ne "${BOLD}  Type 'YES' to continue with scorched-earth migration: ${RESET}"
read -r CONFIRM

if [ "${CONFIRM}" != "YES" ]; then
  echo -e ""
  echo -e "${YELLOW}  ⚠ Migration aborted. Database untouched.${RESET}"
  echo -e "${DIM}  Deployment completed successfully — vault remains in legacy state.${RESET}"
  echo -e ""
  exit 0
fi

# ============================================================================
# PHASE 3: SCORCHED-EARTH DATABASE TRUNCATION
# ============================================================================
echo -e ""
echo -e "${RED}  ▶ PHASE 3: WIPING LEGACY VAULT...${RESET}"
echo -e "${DIM}    Target: ${DB_NAME} → identity_profiles${RESET}"
echo -e ""

# Connect via Cloud SQL Auth Proxy (must be running locally on 127.0.0.1:5432)
PGPASSWORD="${DB_PASS}" psql \
  -h "${DB_HOST}" \
  -p "${DB_PORT}" \
  -U "${DB_USER}" \
  -d "${DB_NAME}" \
  -c "TRUNCATE TABLE identity_profiles;"

echo -e ""
echo -e "${GREEN}  ✓ LEGACY VAULT WIPED — 0 rows remain${RESET}"
echo -e "${DIM}──────────────────────────────────────────────────────────────────${RESET}"

# ============================================================================
# PHASE 4: ARCFACE RE-INGESTION
# ============================================================================
echo -e ""
echo -e "${GOLD}  ▶ PHASE 4: COMMENCING ARCFACE INGESTION...${RESET}"
echo -e "${DIM}    Engine: DeepFace ArcFace 512-D  |  KMS Envelope Encryption${RESET}"
echo -e ""

# Navigate to project root (script is in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

python -m scripts.ingest_vault

echo -e ""
echo -e "${GREEN}  ✓ ARCFACE INGESTION COMPLETE${RESET}"
echo -e "${DIM}──────────────────────────────────────────────────────────────────${RESET}"

# ============================================================================
# FINAL REPORT
# ============================================================================
echo -e ""
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GOLD}  DEPLOYMENT & MIGRATION COMPLETE${RESET}"
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e ""
echo -e "  ${GREEN}✓${RESET} Cloud Run backend deployed (8Gi / 4 CPU)"
echo -e "  ${GREEN}✓${RESET} Legacy 1404-D vault truncated"
echo -e "  ${GREEN}✓${RESET} Fresh 512-D ArcFace embeddings ingested"
echo -e ""
echo -e "  ${CYAN}SERVICE URL${RESET}  $(gcloud run services describe ${SERVICE_NAME} --project=${PROJECT_ID} --region=${REGION} --format='value(status.url)' 2>/dev/null || echo 'Run: gcloud run services describe facial-backend --region=us-east4')"
echo -e ""
echo -e "${GOLD}══════════════════════════════════════════════════════════════════${RESET}"
echo -e ""
