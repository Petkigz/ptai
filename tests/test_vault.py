"""Tests for vault - secure tool sign-in"""
import tempfile
from pathlib import Path
from src.ptai.vault.vault import Vault

class TestVault:
    def test_store_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            vault.store_tool_credentials("Scout", "gamma_api", {"api_key": "test123"})
            creds = vault.get_tool_credentials("Scout", "gamma_api")
            assert creds is not None
            assert creds["api_key"] == "test123"
    
    def test_list_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            vault.store_tool_credentials("Scout", "gamma_api", {"key": "1"})
            vault.store_tool_credentials("Scout", "clob_api", {"key": "2"})
            tools = vault.list_tools_for_teammate("Scout")
            assert len(tools) == 2
            assert "gamma_api" in tools
    
    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            vault.store_tool_credentials("Scout", "gamma_api", {"k": "1"})
            vault.store_tool_credentials("Trader", "polymarket", {"k": "2"})
            all_tools = vault.list_all()
            assert len(all_tools) == 2
            assert "Scout" in all_tools
            assert "Trader" in all_tools
    
    def test_revoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            vault.store_tool_credentials("Scout", "gamma_api", {"k": "1"})
            assert vault.revoke("Scout", "gamma_api") is True
            assert vault.get_tool_credentials("Scout", "gamma_api") is None
    
    def test_encryption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"), master_key="test-master-key")
            sensitive = {"PRIVATE_KEY": "0x" + "a"*64, "public": "visible"}
            vault.store_tool_credentials("Trader", "polymarket", sensitive)
            
            # Check file is encrypted
            import json
            with open(Path(tmpdir) / "vault.json") as f:
                data = json.load(f)
                stored = data["Trader:polymarket"]["credentials"]
                # Private key should be encrypted, not plaintext
                assert stored["PRIVATE_KEY"] != "0x" + "a"*64
                assert stored["public"] == "visible"
            
            # But get should decrypt
            creds = vault.get_tool_credentials("Trader", "polymarket")
            assert creds["PRIVATE_KEY"] == "0x" + "a"*64
