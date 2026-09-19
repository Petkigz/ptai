"""
Vault - Secure storage for tool credentials
Bots sign in to your tools: Polymarket, X, Discord, Telegram, etc.
Local-only, encrypted, per-user
"""
import json
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from loguru import logger
import hashlib

@dataclass
class ToolCredentials:
    teammate: str
    tool: str
    credentials: Dict[str, Any]
    created_at: str
    last_used: Optional[str] = None
    is_encrypted: bool = False

class Vault:
    """
    Secure vault for tool sign-ins - premium product feature
    Each teammate can sign in to tools, credentials stored locally encrypted
    """
    def __init__(self, vault_path: str = "./data/vault.json", master_key: Optional[str] = None):
        self.vault_path = Path(vault_path)
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self.master_key = master_key or os.getenv("VAULT_MASTER_KEY", "ptai-local-vault-key-2024")
        self.credentials: Dict[str, ToolCredentials] = {}
        self.load()
    
    def _encrypt(self, data: str) -> str:
        # Simple XOR encryption for local vault (not for production high security, but prevents plaintext)
        # For production, use Fernet from cryptography
        try:
            from cryptography.fernet import Fernet
            # Derive key from master
            key = base64.urlsafe_b64encode(hashlib.sha256(self.master_key.encode()).digest())
            f = Fernet(key)
            return f.encrypt(data.encode()).decode()
        except ImportError:
            # Fallback XOR
            key = self.master_key
            encrypted = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
            return base64.b64encode(encrypted.encode()).decode()
    
    def _decrypt(self, encrypted_data: str) -> str:
        try:
            from cryptography.fernet import Fernet
            key = base64.urlsafe_b64encode(hashlib.sha256(self.master_key.encode()).digest())
            f = Fernet(key)
            return f.decrypt(encrypted_data.encode()).decode()
        except ImportError:
            try:
                decoded = base64.b64decode(encrypted_data.encode()).decode()
                key = self.master_key
                return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded))
            except:
                return encrypted_data  # assume not encrypted
        except Exception as e:
            logger.warning(f"Decrypt failed: {e}, returning as is")
            return encrypted_data
    
    def load(self):
        if self.vault_path.exists():
            try:
                with open(self.vault_path, "r") as f:
                    data = json.load(f)
                    for key, cred_dict in data.items():
                        # Decrypt credentials if encrypted
                        creds = cred_dict.get("credentials", {})
                        # Try decrypt sensitive fields
                        decrypted_creds = {}
                        for k, v in creds.items():
                            if isinstance(v, str) and ("PRIVATE" in k or "KEY" in k or "TOKEN" in k):
                                try:
                                    decrypted_creds[k] = self._decrypt(v)
                                except:
                                    decrypted_creds[k] = v
                            else:
                                decrypted_creds[k] = v
                        cred_dict["credentials"] = decrypted_creds
                        self.credentials[key] = ToolCredentials(**cred_dict)
                logger.info(f"Vault loaded {len(self.credentials)} credentials from {self.vault_path}")
            except Exception as e:
                logger.warning(f"Vault load failed: {e}")
    
    def save(self):
        try:
            data = {}
            for key, cred in self.credentials.items():
                cred_dict = asdict(cred)
                # Encrypt sensitive fields
                encrypted_creds = {}
                for k, v in cred_dict["credentials"].items():
                    if isinstance(v, str) and ("PRIVATE" in k or "KEY" in k or "TOKEN" in k) and v:
                        encrypted_creds[k] = self._encrypt(v)
                        cred_dict["is_encrypted"] = True
                    else:
                        encrypted_creds[k] = v
                cred_dict["credentials"] = encrypted_creds
                data[key] = cred_dict
            
            with open(self.vault_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Vault saved {len(self.credentials)} credentials")
        except Exception as e:
            logger.error(f"Vault save failed: {e}")
    
    def store_tool_credentials(self, teammate: str, tool: str, credentials: Dict[str, Any]):
        key = f"{teammate}:{tool}"
        cred = ToolCredentials(
            teammate=teammate,
            tool=tool,
            credentials=credentials,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_encrypted=False
        )
        self.credentials[key] = cred
        self.save()
        logger.info(f"Vault stored credentials for {teammate} -> {tool}")
    
    def get_tool_credentials(self, teammate: str, tool: str) -> Optional[Dict[str, Any]]:
        key = f"{teammate}:{tool}"
        cred = self.credentials.get(key)
        if cred:
            cred.last_used = datetime.now(timezone.utc).isoformat()
            self.save()
            return cred.credentials
        return None
    
    def list_tools_for_teammate(self, teammate: str) -> List[str]:
        tools = []
        for key in self.credentials.keys():
            if key.startswith(f"{teammate}:"):
                tools.append(key.split(":", 1)[1])
        return tools
    
    def list_all(self) -> Dict[str, List[str]]:
        result = {}
        for key, cred in self.credentials.items():
            teammate, tool = key.split(":", 1)
            if teammate not in result:
                result[teammate] = []
            result[teammate].append(tool)
        return result
    
    def revoke(self, teammate: str, tool: str) -> bool:
        key = f"{teammate}:{tool}"
        if key in self.credentials:
            del self.credentials[key]
            self.save()
            logger.info(f"Vault revoked {teammate} -> {tool}")
            return True
        return False
