import pytest
from security_helpers import (
    parse_allowed_origins,
    is_mock_crypto_allowed,
    is_safe_image_url,
    is_payment_unlocked,
    get_safe_error_response
)

def test_parse_allowed_origins():
    origins = parse_allowed_origins("https://a.com, https://b.com")
    assert origins == ["https://a.com", "https://b.com"]
    
    with pytest.raises(ValueError):
        parse_allowed_origins("*")
        
    with pytest.raises(ValueError):
        parse_allowed_origins(".*\.run\.app")

def test_is_mock_crypto_allowed():
    assert is_mock_crypto_allowed("development", "true") is True
    assert is_mock_crypto_allowed("production", "true") is False
    assert is_mock_crypto_allowed("development", "false") is False

def test_is_safe_image_url():
    assert is_safe_image_url("gs://my-bucket/test.jpg", "my-bucket") is True
    assert is_safe_image_url("http://evil.com/test.jpg", "my-bucket") is False
    assert is_safe_image_url("gs://other-bucket/test.jpg", "my-bucket") is False

def test_is_payment_unlocked():
    assert is_payment_unlocked("sess_123", "paid", "job_1", "job_1") is True
    assert is_payment_unlocked("", "paid", "job_1", "job_1") is False
    assert is_payment_unlocked("sess_123", "unpaid", "job_1", "job_1") is False
    assert is_payment_unlocked("sess_123", "paid", "job_2", "job_1") is False

def test_get_safe_error_response():
    resp = get_safe_error_response("req_123")
    assert resp["detail"] == "Internal Server Error"
    assert resp["request_id"] == "req_123"
