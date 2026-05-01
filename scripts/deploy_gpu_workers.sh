#!/usr/bin/env bash
# Fallen — GPU Worker Fleet Deployment (GCP Batch)
# Spawns preemptible (SPOT) GPU instances to process hyperscale dataset shards.

set -e

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
BUCKET_NAME="hoppwhistle-facial-uploads"
JOB_NAME="Fallen-hyperscale-worker-$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="gcr.io/${PROJECT_ID}/Fallen-worker:latest"

# Shards to process (0 to N). E.g., for 1 million images at 10,000 per shard = 100 shards.
# Adjust TASK_COUNT to match the total number of shards.
TASK_COUNT=100
PARALLELISM=20 # Max VMs to run simultaneously

echo "[DEPLOY] Building and pushing Dockerfile.worker..."
gcloud builds submit --tag ${IMAGE_URI} -f Dockerfile.worker .

echo "[DEPLOY] Submitting GCP Batch Job: ${JOB_NAME}"
echo "[DEPLOY] Provisioning Model: SPOT (Preemptible)"
echo "[DEPLOY] Machine Type: g2-standard-4 (1x L4 GPU)"
echo "[DEPLOY] Tasks: ${TASK_COUNT} (Parallelism: ${PARALLELISM})"

# Write out the JSON config for the Batch job
cat <<EOF > batch-job-config.json
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "${IMAGE_URI}",
              "volumes": [
                "/mnt/gcs:gs://${BUCKET_NAME}"
              ]
            }
          }
        ],
        "computeResource": {
          "cpuMilli": 4000,
          "memoryMib": 16384
        },
        "maxRetryCount": 3
      },
      "taskCount": ${TASK_COUNT},
      "parallelism": ${PARALLELISM}
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "machineType": "g2-standard-4",
          "provisioningModel": "SPOT",
          "accelerators": [
            {
              "type": "nvidia-l4",
              "count": 1
            }
          ]
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

# Submit the Batch job
gcloud batch jobs submit ${JOB_NAME} \
    --location=${REGION} \
    --config=batch-job-config.json

rm batch-job-config.json

echo "[DEPLOY] Batch job submitted successfully."
echo "You can monitor the workers in the Google Cloud Console -> Batch."
