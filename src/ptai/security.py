"""
PTAI Security Module - Hardening, validation, and attack prevention
"""
import re
import html
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import hashlib
import secrets

# Rate limiting storage
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# Security constants
MAX_REQUESTS_PER_MINUTE = 60
MAX_REQUESTS_PER_SECOND = 10
ALLOWED_BOT_TYPES = {"project", "outbound", "systems", "scout", "researcher", "trader", "custom"}
ALLOWED_TASK_TYPES = {"custom", "scan_markets", "research", "trade", "outbound", "system", "routine", "full_cycle", "scan_only", "sentiment_only", "research_only"}
ALLOWED_ROUTINE_ACTIONS = {"navigate", "click", "type", "wait", "extract", "api_call"}
MAX_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 1000
MAX_BOTS_PER_USER = 20
MAX_PROJECTS_PER_USER = 50
MAX_ROUTINES_PER_USER = 100

# XSS patterns to detect
XSS_PATTERNS = [
    r"<script",
    r"javascript:",
    r"onerror=",
    r"onload=",
    r"onclick=",
    r"<iframe",
    r"eval\(",
    r"document\.cookie",
    r"document\.write",
    r"<img[^>]+onerror",
    r"<svg[^>]+onload",
]

# SQL injection patterns
SQLI_PATTERNS = [
    r"';",
    r"' OR",
    r"' AND",
    r"UNION SELECT",
    r"DROP TABLE",
    r"INSERT INTO",
    r"DELETE FROM",
    r"--",
    r"/\*",
    r"xp_",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e",
    r"%252e",
    r"~\/",
]

def sanitize_input(value: str, max_length: int = MAX_NAME_LENGTH, allow_html: bool = False) -> str:
    """Sanitize user input to prevent XSS and injection"""
    if not isinstance(value, str):
        value = str(value)
    
    # Truncate to max length
    value = value[:max_length]
    
    # Check for XSS
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"Potentially malicious input detected: {pattern}")
    
    # Check for SQLi
    for pattern in SQLI_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"SQL injection attempt detected: {pattern}")
    
    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"Path traversal attempt detected")
    
    # Escape HTML if not allowed
    if not allow_html:
        value = html.escape(value)
    
    return value.strip()

def sanitize_bot_name(name: str) -> str:
    """Validate bot name"""
    if not name or len(name.strip()) < 2:
        raise ValueError("Bot name must be at least 2 characters")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Bot name too long, max {MAX_NAME_LENGTH}")
    return sanitize_input(name, MAX_NAME_LENGTH)

def sanitize_project_name(name: str) -> str:
    """Validate project name"""
    if not name or len(name.strip()) < 2:
        raise ValueError("Project name must be at least 2 characters")
    return sanitize_input(name, MAX_NAME_LENGTH)

def validate_bot_type(bot_type: str) -> str:
    """Validate bot type"""
    if bot_type not in ALLOWED_BOT_TYPES:
        raise ValueError(f"Invalid bot type: {bot_type}, allowed: {ALLOWED_BOT_TYPES}")
    return bot_type

def validate_task_type(task_type: str) -> str:
    """Validate task type"""
    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(f"Invalid task type: {task_type}")
    return task_type

def validate_routine_action(action: str) -> str:
    """Validate routine action"""
    if action not in ALLOWED_ROUTINE_ACTIONS:
        raise ValueError(f"Invalid routine action: {action}")
    return action

def validate_url(url: str) -> str:
    """Validate URL to prevent SSRF"""
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Must be http/https
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("URL must start with http:// or https://")
    
    # Block private IPs and localhost for SSRF prevention (except allowed)
    blocked_patterns = [
        r"http://localhost",
        r"http://127\.",
        r"http://10\.",
        r"http://192\.168\.",
        r"http://172\.(1[6-9]|2[0-9]|3[0-1])\.",
        r"http://0\.0\.0\.0",
        r"file://",
        r"gopher://",
        r"ftp://",
    ]
    
    # Allow polymarket, but block internal
    allowed_domains = ["polymarket.com", "gamma-api.polymarket.com", "clob.polymarket.com", "localhost:1234", "localhost:11434"]
    is_allowed = any(domain in url for domain in allowed_domains)
    
    if not is_allowed:
        for pattern in blocked_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                raise ValueError(f"Blocked URL for security: {url}")
    
    return url[:500]  # limit length

def validate_private_key(pk: str) -> str:
    """Validate private key format without exposing it"""
    if not pk:
        raise ValueError("Private key empty")
    pk = pk.strip()
    if not pk.startswith("0x"):
        pk = "0x" + pk
    if len(pk) != 66:
        raise ValueError(f"Private key must be 66 chars (0x + 64 hex), got {len(pk)}")
    if not re.match(r"^0x[0-9a-fA-F]{64}$", pk):
        raise ValueError("Private key must be hex")
    return pk

def validate_funder_address(addr: str) -> str:
    """Validate funder address"""
    if not addr:
        raise ValueError("Funder address empty")
    addr = addr.strip()
    if not addr.startswith("0x") or len(addr) != 42:
        raise ValueError(f"Funder address must be 42 chars, got {len(addr)}")
    if not re.match(r"^0x[0-9a-fA-F]{40}$", addr):
        raise ValueError("Funder address must be hex")
    return addr

def check_rate_limit(client_ip: str, endpoint: str = "global") -> bool:
    """Simple rate limiting - returns True if allowed, False if blocked"""
    now = time.time()
    key = f"{client_ip}:{endpoint}"
    
    # Clean old entries (older than 60 seconds)
    _rate_limit_store[key] = [t for t in _rate_limit_store[key] if now - t < 60]
    
    # Check per-minute limit
    if len(_rate_limit_store[key]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    # Check per-second burst
    recent = [t for t in _rate_limit_store[key] if now - t < 1]
    if len(recent) >= MAX_REQUESTS_PER_SECOND:
        return False
    
    _rate_limit_store[key].append(now)
    return True

def get_client_ip(request) -> str:
    """Get client IP from request"""
    # Check for forwarded headers (but validate)
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        # Take first IP
        ip = forwarded.split(",")[0].strip()
        # Basic validation
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            return ip
    # Fallback to direct
    if hasattr(request, 'client') and request.client:
        return request.client.host
    return "unknown"

def generate_secure_token() -> str:
    """Generate secure random token for local auth"""
    return secrets.token_urlsafe(32)

def hash_token(token: str) -> str:
    """Hash token for storage"""
    return hashlib.sha256(token.encode()).hexdigest()

def validate_file_path(path: str, base_dir: Path) -> Path:
    """Validate file path to prevent traversal, must be within base_dir"""
    p = Path(path).resolve()
    base = base_dir.resolve()
    try:
        # Check if p is within base
        p.relative_to(base)
    except ValueError:
        raise ValueError(f"Path traversal detected: {path} not within {base}")
    return p

def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive fields in dict"""
    masked = {}
    for k, v in data.items():
        if any(s in k.upper() for s in ["PRIVATE_KEY", "SECRET", "TOKEN", "PASSWORD", "KEY"]):
            if isinstance(v, str) and v:
                masked[k] = v[:6] + "..." + v[-4:] if len(v) > 10 else "***"
            else:
                masked[k] = "***"
        else:
            masked[k] = v
    return masked

def security_headers() -> Dict[str, str]:
    """Return security headers"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; connect-src 'self' http://localhost:1234 http://localhost:11434 https://gamma-api.polymarket.com https://clob.polymarket.com;",
    }

class SecurityAudit:
    """Security audit runner - runs attacks to catch gaps"""
    
    def __init__(self):
        self.results = []
    
    def log(self, test: str, passed: bool, details: str = ""):
        self.results.append({"test": test, "passed": passed, "details": details})
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}: {details}")
    
    def test_xss_injection(self):
        """Test XSS injection attempts"""
        payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
        ]
        for payload in payloads:
            try:
                sanitize_input(payload)
                self.log(f"XSS block: {payload[:20]}", False, "Should have been blocked")
            except ValueError:
                self.log(f"XSS block: {payload[:20]}", True, "Blocked correctly")
    
    def test_sqli(self):
        """Test SQL injection"""
        payloads = [
            "'; DROP TABLE trades; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
        ]
        for payload in payloads:
            try:
                sanitize_input(payload)
                self.log(f"SQLi block: {payload[:20]}", False, "Should have been blocked")
            except ValueError:
                self.log(f"SQLi block: {payload[:20]}", True, "Blocked correctly")
    
    def test_path_traversal(self):
        """Test path traversal"""
        payloads = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e/%2e%2e/etc/passwd",
        ]
        for payload in payloads:
            try:
                sanitize_input(payload)
                self.log(f"Path traversal block: {payload}", False, "Should have been blocked")
            except ValueError:
                self.log(f"Path traversal block: {payload}", True, "Blocked correctly")
    
    def test_bot_type_validation(self):
        """Test bot type validation"""
        try:
            validate_bot_type("invalid_type")
            self.log("Bot type validation", False, "Should block invalid type")
        except ValueError:
            self.log("Bot type validation", True, "Blocked invalid type")
        
        try:
            validate_bot_type("project")
            self.log("Bot type valid", True, "Allowed valid type")
        except ValueError:
            self.log("Bot type valid", False, "Should allow valid type")
    
    def test_private_key_validation(self):
        """Test private key validation"""
        try:
            validate_private_key("0x" + "a"*64)
            self.log("Private key valid", True, "Valid key accepted")
        except ValueError as e:
            self.log("Private key valid", False, str(e))
        
        try:
            validate_private_key("invalid")
            self.log("Private key invalid", False, "Should reject invalid")
        except ValueError:
            self.log("Private key invalid", True, "Rejected invalid key")
    
    def test_url_validation(self):
        """Test URL validation for SSRF"""
        try:
            validate_url("https://polymarket.com")
            self.log("URL polymarket allowed", True, "Allowed")
        except ValueError:
            self.log("URL polymarket allowed", False, "Should allow")
        
        try:
            validate_url("http://127.0.0.1/admin")
            self.log("URL SSRF block", False, "Should block private IP")
        except ValueError:
            self.log("URL SSRF block", True, "Blocked private IP")
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        ip = "test_ip_123"
        # Clear
        _rate_limit_store.clear()
        # Fill up to limit
        for _ in range(MAX_REQUESTS_PER_MINUTE):
            check_rate_limit(ip, "test")
        
        allowed = check_rate_limit(ip, "test")
        self.log("Rate limiting", not allowed, f"Blocked after {MAX_REQUESTS_PER_MINUTE} requests" if not allowed else "Should have blocked")
    
    def run_all(self):
        """Run all security tests"""
        print("\n=== PTAI Security Audit ===\n")
        self.test_xss_injection()
        self.test_sqli()
        self.test_path_traversal()
        self.test_bot_type_validation()
        self.test_private_key_validation()
        self.test_url_validation()
        self.test_rate_limiting()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n=== Results: {passed}/{total} passed ({passed/total*100:.1f}%) ===\n")
        
        if passed < total:
            print("Failures:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['test']}: {r['details']}")
        
        return passed, total
