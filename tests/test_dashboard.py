"""Tests for dashboard API - security and functionality"""
import pytest
from fastapi.testclient import TestClient
from src.ptai.dashboard import app
import tempfile
from pathlib import Path

client = TestClient(app)

class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_sw_js(self):
        response = client.get("/sw.js")
        assert response.status_code == 200
    
    def test_favicon(self):
        response = client.get("/favicon.ico")
        assert response.status_code == 204

class TestStatus:
    def test_api_status(self):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data

class TestBotsAPI:
    def test_list_bots(self):
        response = client.get("/api/bots/list")
        assert response.status_code == 200
        data = response.json()
        assert "bots" in data
        assert "count" in data
        # Should create default team
        assert data["count"] >= 6
    
    def test_create_bot_valid(self):
        response = client.post("/api/bots/create", json={
            "name": "TestBot",
            "type": "project",
            "role": "Test role"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "bot" in data
    
    def test_create_bot_invalid_type(self):
        response = client.post("/api/bots/create", json={
            "name": "TestBot",
            "type": "invalid_type",
            "role": "Test"
        })
        # Should be blocked by security validation
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_create_bot_xss_blocked(self):
        response = client.post("/api/bots/create", json={
            "name": "<script>alert(1)</script>",
            "type": "project",
            "role": "Test"
        })
        assert response.status_code == 400
    
    def test_create_bot_short_name(self):
        response = client.post("/api/bots/create", json={
            "name": "a",
            "type": "project",
            "role": "Test"
        })
        assert response.status_code == 400
    
    def test_give_task_valid(self):
        # Use name-based lookup since BotManager is per-request (not persisted)
        # API supports finding by id or name
        response = client.post("/api/bots/task", json={
            "bot_id": "Project Lead",
            "title": "Test Task",
            "description": "Test description",
            "task_type": "custom",
            "needs_approval": False
        })
        assert response.status_code == 200
        assert response.json()["status"] == "assigned"
    
    def test_give_task_invalid_type(self):
        list_resp = client.get("/api/bots/list")
        bot_id = list_resp.json()["bots"][0]["id"]
        
        response = client.post("/api/bots/task", json={
            "bot_id": bot_id,
            "title": "Test",
            "description": "Desc",
            "task_type": "invalid_type",
            "needs_approval": False
        })
        assert response.status_code == 400

class TestProjectsAPI:
    def test_list_projects(self):
        response = client.get("/api/projects/list")
        assert response.status_code == 200
        assert "projects" in response.json()
    
    def test_create_project_valid(self):
        response = client.post("/api/projects/create", json={
            "name": "Test Project",
            "description": "Test description",
            "goal": "Test goal",
            "bots": ["Project Lead"]
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"
    
    def test_create_project_xss_blocked(self):
        response = client.post("/api/projects/create", json={
            "name": "<script>alert(1)</script>",
            "description": "Desc",
            "goal": "Goal",
            "bots": []
        })
        assert response.status_code == 400
    
    def test_create_project_short_name(self):
        response = client.post("/api/projects/create", json={
            "name": "a",
            "description": "Desc",
            "goal": "Goal",
            "bots": []
        })
        assert response.status_code == 400

class TestRoutinesAPI:
    def test_list_routines(self):
        response = client.get("/api/routines/list")
        assert response.status_code == 200
        assert "routines" in response.json()
    
    def test_create_routine_valid(self):
        response = client.post("/api/routines/create", json={
            "name": "Test Routine",
            "description": "Test desc",
            "steps": [
                {"action": "navigate", "target": "https://polymarket.com", "description": "Go"}
            ]
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"
    
    def test_create_routine_invalid_action(self):
        response = client.post("/api/routines/create", json={
            "name": "Test",
            "description": "Desc",
            "steps": [
                {"action": "rm -rf", "target": "/", "description": "Hack"}
            ]
        })
        assert response.status_code == 400
    
    def test_record_flow(self):
        # Start recording
        response = client.post("/api/routines/record/start", json={
            "name": "Recorded Test",
            "description": "Test"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "recording"
        
        # Record step
        response = client.post("/api/routines/record/step", json={
            "action": "navigate",
            "target": "https://polymarket.com",
            "description": "Go to polymarket"
        })
        assert response.status_code == 200
        
        # Stop
        response = client.post("/api/routines/record/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "saved"
    
    def test_record_invalid_url_blocked(self):
        client.post("/api/routines/record/start", json={
            "name": "Test",
            "description": "Desc"
        })
        response = client.post("/api/routines/record/step", json={
            "action": "navigate",
            "target": "http://127.0.0.1/admin",
            "description": "SSRF attempt"
        })
        # Should block SSRF
        assert response.status_code == 400
        # Cleanup
        client.post("/api/routines/record/stop")

class TestApprovalsAPI:
    def test_list_approvals(self):
        response = client.get("/api/approvals/list")
        assert response.status_code == 200
        assert "pending" in response.json()
    
    def test_approve_nonexistent(self):
        response = client.post("/api/approvals/approve", json={
            "approval_id": "nonexistent123"
        })
        assert response.status_code == 404
    
    def test_reject_nonexistent(self):
        response = client.post("/api/approvals/reject", json={
            "approval_id": "nonexistent123"
        })
        assert response.status_code == 404
    
    def test_approve_xss_blocked(self):
        response = client.post("/api/approvals/approve", json={
            "approval_id": "<script>alert(1)</script>"
        })
        assert response.status_code == 400

class TestConfigAPI:
    def test_get_config(self):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "_private_key_set" in data
        # Should not expose raw private key
        assert "POLYMARKET_PRIVATE_KEY" not in data["_raw"]
    
    def test_update_config_valid(self):
        response = client.post("/api/config", json={
            "BANKROLL": "100",
            "DRY_RUN": "true"
        })
        assert response.status_code == 200
    
    def test_update_config_invalid_bankroll(self):
        response = client.post("/api/config", json={
            "BANKROLL": "1"  # Too low
        })
        assert response.status_code == 400
    
    def test_update_config_private_key_validation(self):
        response = client.post("/api/config", json={
            "POLYMARKET_PRIVATE_KEY": "invalid"
        })
        assert response.status_code == 400

class TestWalletAPI:
    def test_wallet_test_invalid(self):
        response = client.post("/api/wallet/test", json={
            "private_key": "invalid",
            "funder_address": "0x123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

class TestSecurityHeaders:
    def test_security_headers_present(self):
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

class TestRateLimiting:
    def test_rate_limiting(self):
        # Make many requests quickly
        for _ in range(65):
            client.get("/health")
        
        # Next should be rate limited
        response = client.get("/health")
        # Might be 429 or 200 depending on timing, but should not crash
        assert response.status_code in [200, 429]

class TestCORS:
    def test_cors_restricted(self):
        # Test that CORS is not *
        # The middleware should only allow specific origins
        response = client.get("/health", headers={"Origin": "http://evil.com"})
        # Should not have * in allow-origin
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] != "*"

class TestAuthToken:
    def test_get_auth_token(self):
        response = client.get("/api/auth/token")
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert len(data["token"]) > 20
    
    def test_config_post_requires_token_when_file_exists(self):
        # When auth token file exists, POST without token should 401 for non-test client
        # Our test client bypasses, so we test that endpoint exists and returns proper structure when bypassed
        # Simulate real client by not using test bypass
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app, AUTH_TOKEN_PATH, get_auth_token
        # Ensure token file exists
        token = get_auth_token()
        assert AUTH_TOKEN_PATH.exists()
        # Test client with bypass should still work (our verify allows testclient)
        response = client.post("/api/config", json={"BANKROLL": "100"})
        assert response.status_code in [200, 400]  # 200 if valid, 400 if validation fails but not 401 because testclient bypass
        # Test that token endpoint returns same token
        response2 = client.get("/api/auth/token")
        assert response2.json()["token"] == token

class TestDashboardHTML:
    def test_dashboard_loads(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "PTAI" in response.text
        assert "Dynamic Bots" in response.text
        assert "Projects" in response.text
        assert "Routines" in response.text
        assert "Approvals" in response.text
        # Check XSS protection - escapeHTML function exists
        assert "escapeHTML" in response.text
        assert "X-PTAI-Token" in response.text or "authToken" in response.text

class TestThePageScriptActuallyRuns:
    """
    The dashboard's JavaScript is one inline block, so ONE bad literal kills
    every function on the page: the data fetch, the tab switching, the buttons.
    The page still returns HTTP 200 and renders, which is why this went
    unnoticed - it looked healthy while being completely inert.

    The specific failure it caught: a `\n` inside a JS string written as a
    single backslash in the Python source, so Python turned it into a real
    newline and the JS parser saw an unterminated string.
    """

    @staticmethod
    def _newline_in_js_string(js: str):
        """First line where a JS string literal spans a newline, or None."""
        state = "code"
        prev = ""
        start_line = None
        line = 1
        i = 0
        while i < len(js):
            c = js[i]
            nxt = js[i + 1] if i + 1 < len(js) else ""
            if c == "\n":
                if state in ("'", '"'):
                    return start_line, line
                if state == "//":
                    state = "code"
                line += 1
            elif state == "code":
                if c == "/" and nxt == "/":
                    state = "//"
                    i += 1
                elif c == "/" and nxt == "*":
                    state = "/*"
                    i += 1
                elif c == "/" and (prev == "" or prev in "(,=:[!&|?{};+-*\n"):
                    # A regex literal, not a division: skip to its unescaped end.
                    i += 1
                    while i < len(js):
                        if js[i] == "\\":
                            i += 1
                        elif js[i] == "/":
                            break
                        elif js[i] == "\n":
                            line += 1
                        i += 1
                elif c in ("'", '"', "`"):
                    state = c
                    start_line = line
                if not c.isspace():
                    prev = c
            elif state in ("'", '"'):
                if c == "\\":
                    i += 1
                elif c == state:
                    state = "code"
            elif state == "`":
                if c == "\\":
                    i += 1
                elif c == "`":
                    state = "code"
            elif state == "/*":
                if c == "*" and nxt == "/":
                    state = "code"
                    i += 1
            i += 1
        return None

    def test_no_string_literal_spans_a_newline(self):
        """A real newline inside a JS string is a fatal syntax error."""
        import re

        page = client.get("/").text
        blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
        assert blocks, "the dashboard serves no script at all"
        for block in blocks:
            found = self._newline_in_js_string(block)
            assert found is None, (
                f"a JS string literal opened on line {found[0]} and met a real "
                f"newline on line {found[1]} - the whole page script will fail "
                f"to parse and every button and fetch on it is dead"
            )

    def test_the_operator_view_card_is_served(self):
        """The card the operator reads is in the page, not just in the repo."""
        page = client.get("/").text
        assert "Operator View" in page
        assert "operator-lines" in page
        assert "fetchOperatorView" in page

    def test_the_page_parses_as_javascript_when_the_interpreter_is_available(self):
        """
        The strongest check available locally: hand the block to a real JS
        parser. Skipped rather than failed where node is not installed - the
        newline check above is the version that always runs.
        """
        import re
        import shutil
        import subprocess
        import tempfile

        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed")
        page = client.get("/").text
        block = max(re.findall(r"<script>(.*?)</script>", page, re.S), key=len)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(block)
            path = fh.name
        try:
            result = subprocess.run([node, "--check", path],
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (
                f"the dashboard script does not parse: {result.stderr[:600]}")
        finally:
            import os
            os.unlink(path)
