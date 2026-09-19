"""Tests for security module - XSS, SQLi, path traversal, validation"""
import pytest
from src.ptai.security import (
    sanitize_input, sanitize_bot_name, sanitize_project_name,
    validate_bot_type, validate_task_type, validate_routine_action,
    validate_url, validate_private_key, validate_funder_address,
    check_rate_limit, mask_sensitive_data, security_headers,
    _rate_limit_store
)

class TestSanitizeInput:
    def test_valid_input(self):
        assert sanitize_input("Hello World") == "Hello World"
    
    def test_xss_script_blocked(self):
        with pytest.raises(ValueError, match="malicious"):
            sanitize_input("<script>alert('xss')</script>")
    
    def test_xss_javascript_blocked(self):
        with pytest.raises(ValueError):
            sanitize_input("javascript:alert(1)")
    
    def test_xss_img_onerror_blocked(self):
        with pytest.raises(ValueError):
            sanitize_input("<img src=x onerror=alert(1)>")
    
    def test_sqli_drop_blocked(self):
        with pytest.raises(ValueError, match="SQL injection"):
            sanitize_input("'; DROP TABLE trades; --")
    
    def test_sqli_union_blocked(self):
        with pytest.raises(ValueError):
            sanitize_input("UNION SELECT * FROM users")
    
    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_input("../../etc/passwd")
    
    def test_max_length_truncation(self):
        long_str = "a" * 200
        result = sanitize_input(long_str, max_length=100)
        assert len(result) == 100
    
    def test_html_escaped(self):
        result = sanitize_input("<b>bold</b>")
        assert "&lt;b&gt;" in result

class TestBotValidation:
    def test_valid_bot_types(self):
        for t in ["project", "outbound", "systems", "scout", "researcher", "trader", "custom"]:
            assert validate_bot_type(t) == t
    
    def test_invalid_bot_type(self):
        with pytest.raises(ValueError):
            validate_bot_type("invalid")
    
    def test_valid_task_types(self):
        assert validate_task_type("custom") == "custom"
        assert validate_task_type("scan_markets") == "scan_markets"
    
    def test_invalid_task_type(self):
        with pytest.raises(ValueError):
            validate_task_type("hacking")
    
    def test_valid_routine_actions(self):
        for a in ["navigate", "click", "type", "wait", "extract", "api_call"]:
            assert validate_routine_action(a) == a
    
    def test_invalid_routine_action(self):
        with pytest.raises(ValueError):
            validate_routine_action("rm -rf")

class TestURLValidation:
    def test_valid_polymarket_url(self):
        assert validate_url("https://polymarket.com") == "https://polymarket.com"
        assert validate_url("https://gamma-api.polymarket.com/events") != ""
    
    def test_invalid_url_no_http(self):
        with pytest.raises(ValueError):
            validate_url("ftp://example.com")
    
    def test_ssrf_localhost_blocked(self):
        with pytest.raises(ValueError):
            validate_url("http://127.0.0.1/admin")
    
    def test_ssrf_private_ip_blocked(self):
        with pytest.raises(ValueError):
            validate_url("http://10.0.0.1/secret")
    
    def test_empty_url(self):
        with pytest.raises(ValueError):
            validate_url("")

class TestPrivateKeyValidation:
    def test_valid_private_key(self):
        pk = "0x" + "a" * 64
        assert validate_private_key(pk) == pk
    
    def test_private_key_without_0x(self):
        pk = "a" * 64
        result = validate_private_key(pk)
        assert result.startswith("0x")
    
    def test_invalid_length(self):
        with pytest.raises(ValueError):
            validate_private_key("0x123")
    
    def test_invalid_hex(self):
        with pytest.raises(ValueError):
            validate_private_key("0x" + "z" * 64)

class TestFunderValidation:
    def test_valid_funder(self):
        addr = "0x" + "a" * 40
        assert validate_funder_address(addr) == addr
    
    def test_invalid_funder_length(self):
        with pytest.raises(ValueError):
            validate_funder_address("0x123")
    
    def test_invalid_funder_hex(self):
        with pytest.raises(ValueError):
            validate_funder_address("0x" + "z" * 40)

class TestRateLimiting:
    def setup_method(self):
        _rate_limit_store.clear()
    
    def test_rate_limit_allows_under_limit(self):
        assert check_rate_limit("test_ip", "endpoint") is True
    
    def test_rate_limit_blocks_over_limit(self):
        ip = "test_ip_limit"
        for _ in range(60):
            check_rate_limit(ip, "test")
        assert check_rate_limit(ip, "test") is False
    
    def test_rate_limit_per_second_burst(self):
        ip = "test_ip_burst"
        for _ in range(10):
            check_rate_limit(ip, "burst_test")
        # 11th in same second should be blocked
        assert check_rate_limit(ip, "burst_test") is False

class TestMaskSensitive:
    def test_masks_private_key(self):
        data = {"POLYMARKET_PRIVATE_KEY": "0x" + "a"*64, "BANKROLL": "50"}
        masked = mask_sensitive_data(data)
        assert "a"*10 not in masked["POLYMARKET_PRIVATE_KEY"]
        assert masked["BANKROLL"] == "50"

class TestSecurityHeaders:
    def test_headers_present(self):
        headers = security_headers()
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "Content-Security-Policy" in headers

class TestBotNameSanitization:
    def test_valid_name(self):
        assert sanitize_bot_name("Project Lead") == "Project Lead"
    
    def test_short_name_rejected(self):
        with pytest.raises(ValueError):
            sanitize_bot_name("a")
    
    def test_xss_in_name_blocked(self):
        with pytest.raises(ValueError):
            sanitize_bot_name("<script>alert(1)</script>")
