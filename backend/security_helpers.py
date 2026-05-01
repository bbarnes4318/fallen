import os

def parse_allowed_origins(origins_env: str) -> list[str]:
    """Parse ALLOWED_ORIGINS securely, rejecting wildcards."""
    if not origins_env:
        return [
            "http://localhost:3000",
            "https://scargods.com",
            "https://www.scargods.com"
        ]
    origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
    
    # Reject wildcards and open regexes
    for o in origins:
        if "*" in o or (".run.app" in o and "https://" not in o):
            raise ValueError(f"Insecure origin detected: {o}")
    return origins

def is_mock_crypto_allowed(environment: str, allow_mock: str) -> bool:
    """Ensure mock crypto is only allowed in development."""
    return environment.lower() == "development" and allow_mock.lower() == "true"

def is_safe_image_url(url: str, bucket_name: str) -> bool:
    """Ensure image URLs only point to the authorized GCS bucket."""
    if not url or not bucket_name:
        return False
    return url.startswith(f"gs://{bucket_name}/")

def is_payment_unlocked(session_id: str, payment_status: str, metadata_job_id: str, requested_job_id: str) -> bool:
    """Refuse unpaid/missing sessions."""
    if not session_id:
        return False
    if payment_status != "paid":
        return False
    if metadata_job_id != requested_job_id:
        return False
    return True

def get_safe_error_response(request_id: str) -> dict:
    """Do not expose raw exception text."""
    return {"detail": "Internal Server Error", "request_id": request_id}
