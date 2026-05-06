# V2 Mark Pipeline Deployment Safety

## Executive Summary
The V2 Forensic Mark Pipeline (`USE_MARK_PIPELINE_V2=true`) introduces catastrophic instability when deployed using legacy Cloud Run concurrency configurations (`concurrency=10`). The integration of the V2 pipeline alongside existing heavy ML ensemble models (ArcFace, Facenet512, MediaPipe) causes native library collisions and hard faults under concurrent load. 

**V2 must NOT be deployed to the main production backend service unless strictly configured with `concurrency=1`.**

## Incident Analysis (May 2026)
During a controlled 10-request canary load test of the V2 `/verify/fuse` endpoint:
- **Concurrency Result:** Catastrophic failure. 10 requests were dispatched with `max_workers=3`. Resulted in 10 failures (3 HTTP 503s, 7 timeouts).
- **Crash Logs:** The underlying C/C++ native libraries (likely OpenCV, dlib, or TensorFlow) crashed the entire Cloud Run container. Log telemetry revealed fatal signals:
  - `Uncaught signal: 11 (SIGSEGV)` - Segmentation Fault
  - `Uncaught signal: 6 (SIGABRT)` - Process Abort
- **OOM Status:** The orchestrator did not emit `OOMKilled` or `Memory limit exceeded`. The crashes originated internally within the native Python extensions failing to allocate memory or encountering thread-safety violations during concurrent inference.
- **Isolated Validation:** The V2 pipeline *is* perfectly stable when concurrency is strictly isolated. An isolated revision (`concurrency=1`, `memory=16Gi`) successfully completed sequential tests with 100% reliability, zero 500s, and correct `lr_marks=1.0` extraction.

## Safe Deployment Model
To deploy V2 safely to production, it must run either as an isolated **Dedicated Cloud Run Service** (e.g., `facial-backend-v2-marks`) or a strictly tagged revision with the following immutable parameters:

```yaml
# Required Cloud Run Deployment Configuration
USE_MARK_PIPELINE_V2: "true"
concurrency: 1          # CRITICAL: Prevents native library SIGSEGV
memory: 16Gi            # CRITICAL: Required for legacy + V2 model overhead
cpu: 4                  # (or 8 for faster inference)
timeout: 300            # Allow up to 300s to absorb ArcFace cold starts
min-instances: 1        # Optional but highly recommended to avoid 60s cold starts
```

## Why `/vault/search` Must Wait
`/vault/search` handles massive 1:N comparisons (e.g., thousands of gallery images per search). 
1. The current legacy embedding models (ArcFace/Facenet512) act as massive latency bottlenecks.
2. If `/vault/search` was wired to V2 under the current architecture, it would rapidly saturate the `concurrency=1` V2 instances, causing queue timeouts or requiring massive horizontal scaling (1 instance per search thread).
3. Do not wire `/vault/search` until the legacy models are optimized or the V2 mark pipeline is decoupled into an asynchronous dedicated worker pool.

## Rollback Procedure
If any V2 deployment inadvertently receives excessive concurrent load and triggers `503 Service Unavailable` or `SIGSEGV` container crashes:
1. Immediately execute traffic rollback to the legacy revision (`USE_MARK_PIPELINE_V2=false`).
   ```bash
   gcloud run services update-traffic facial-backend --region=us-east4 --project=hoppwhistle --to-revisions=<LEGACY_REVISION_NAME>=100
   ```
2. Verify frontend stability.
3. Check Cloud Run logs for `Uncaught signal` to confirm the extent of the crash loop.

## Canary Policy
Do not re-enable the 1% or 10% V2 canary on the primary `facial-backend` service. Any canary testing of V2 must be performed by routing traffic directly to a standalone, concurrency-limited Cloud Run backend to protect organic user traffic from native library crashes.
