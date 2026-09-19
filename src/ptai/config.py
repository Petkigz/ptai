"""
PTAI Config - Loads env + yaml, provides typed settings
Local-only, no cloud dependencies
Supports: Ollama, LM Studio, any OpenAI-compatible local LLM
"""
import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
ENV_PATH = ROOT / ".env"

class RiskConfig(BaseModel):
    max_position_pct: float = 0.06
    min_edge_pct: float = 0.08
    max_open_positions: int = 8
    kelly_fraction: float = 0.5
    min_liquidity: float = 1000
    min_volume_24h: float = 5000
    max_daily_loss_pct: float = 0.15
    max_total_drawdown_pct: float = 0.30
    stop_loss_pct: float = 0.50

class LoopConfig(BaseModel):
    scan_interval_minutes: int = 10
    scan_markets_count: int = 750
    scan_order_by: str = "volume24hr"
    batch_size: int = 50

class LLMConfig(BaseModel):
    provider: Literal["auto", "ollama", "lm_studio", "openai_compatible"] = "auto"
    model: str = "local-model"
    host: str = "http://localhost:1234"
    ollama_host: str = "http://localhost:11434"
    lm_studio_host: str = "http://localhost:1234"
    temperature: float = 0.2
    max_tokens: int = 1200

class SentimentConfig(BaseModel):
    enabled: bool = True
    method: Literal["snscrape", "api", "browser"] = "snscrape"
    max_tweets_per_market: int = 50
    lookback_hours: int = 24

class BrowserConfig(BaseModel):
    headless: bool = False
    persistent_dir: str = "./browser/profiles/default"
    stealth: bool = True
    timeout: int = 30000
    human_delay: bool = True

class ExecutionConfig(BaseModel):
    mode: Literal["api", "browser", "hybrid"] = "hybrid"
    dry_run: bool = True
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    max_slippage: float = 0.02

class PolymarketConfig(BaseModel):
    private_key: Optional[str] = None
    funder_address: Optional[str] = None
    chain_id: int = 137
    signature_type: int = 1
    host: str = "https://clob.polymarket.com"
    gamma_api: str = "https://gamma-api.polymarket.com"
    dry_run: bool = True

class AgentConfig(BaseModel):
    name: str = "ptai-v1"
    initial_bankroll: float = 50.0
    currency: str = "USD"
    daily_cost: float = 5.0
    max_unprofitable_days: int = 3
    self_preservation: bool = True

class Settings(BaseSettings):
    # Bankroll
    bankroll: float = Field(default=50.0, alias="BANKROLL")
    currency: str = Field(default="USD", alias="CURRENCY")
    daily_cost_to_cover: float = Field(default=5.0, alias="DAILY_COST_TO_COVER")
    shutdown_if_unprofitable_days: int = Field(default=3, alias="SHUTDOWN_IF_UNPROFITABLE_DAYS")
    max_daily_loss_pct: float = Field(default=0.15, alias="MAX_DAILY_LOSS_PCT")
    max_total_drawdown_pct: float = Field(default=0.30, alias="MAX_TOTAL_DRAWDOWN_PCT")

    # Polymarket
    polymarket_private_key: Optional[str] = Field(default=None, alias="POLYMARKET_PRIVATE_KEY")
    polymarket_funder_address: Optional[str] = Field(default=None, alias="POLYMARKET_FUNDER_ADDRESS")
    polymarket_chain_id: int = Field(default=137, alias="POLYMARKET_CHAIN_ID")
    polymarket_signature_type: int = Field(default=1, alias="POLYMARKET_SIGNATURE_TYPE")
    polymarket_host: str = Field(default="https://clob.polymarket.com", alias="POLYMARKET_HOST")
    gamma_api: str = Field(default="https://gamma-api.polymarket.com", alias="GAMMA_API")
    dry_run: bool = Field(default=True, alias="DRY_RUN")

    # LLM - Ollama
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")

    # LLM - LM Studio (PRIMARY FOR THIS USER)
    lm_studio_host: str = Field(default="http://localhost:1234", alias="LM_STUDIO_HOST")
    lm_studio_model: str = Field(default="local-model", alias="LM_STUDIO_MODEL")
    lm_studio_api_key: str = Field(default="lm-studio", alias="LM_STUDIO_API_KEY")

    # LLM - Generic
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")  # auto, ollama, lm_studio, openai_compatible
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="http://localhost:1234/v1", alias="OPENAI_BASE_URL")
    use_local_llm: bool = Field(default=True, alias="USE_LOCAL_LLM")

    # X
    sentiment_use_x: bool = Field(default=True, alias="SENTIMENT_USE_X")
    x_use_snscrape: bool = Field(default=True, alias="X_USE_SNSCRAPE")
    x_bearer_token: Optional[str] = Field(default=None, alias="X_BEARER_TOKEN")
    x_api_key: Optional[str] = Field(default=None, alias="X_API_KEY")
    x_api_secret: Optional[str] = Field(default=None, alias="X_API_SECRET")
    x_access_token: Optional[str] = Field(default=None, alias="X_ACCESS_TOKEN")
    x_access_secret: Optional[str] = Field(default=None, alias="X_ACCESS_SECRET")
    x_use_browser: bool = Field(default=False, alias="X_USE_BROWSER")

    # Browser
    browser_headless: bool = Field(default=False, alias="BROWSER_HEADLESS")
    browser_persistent_dir: str = Field(default="./browser/profiles/default", alias="BROWSER_PERSISTENT_DIR")
    browser_stealth: bool = Field(default=True, alias="BROWSER_STEALTH")
    browser_timeout: int = Field(default=30000, alias="BROWSER_TIMEOUT")

    # Risk
    max_position_pct: float = Field(default=0.06, alias="MAX_POSITION_PCT")
    min_edge_pct: float = Field(default=0.08, alias="MIN_EDGE_PCT")
    max_open_positions: int = Field(default=8, alias="MAX_OPEN_POSITIONS")
    kelly_fraction: float = Field(default=0.5, alias="KELLY_FRACTION")
    min_liquidity: float = Field(default=1000, alias="MIN_LIQUIDITY")
    min_volume_24h: float = Field(default=5000, alias="MIN_VOLUME_24H")

    # Loop
    scan_interval_minutes: int = Field(default=10, alias="SCAN_INTERVAL_MINUTES")
    scan_markets_count: int = Field(default=750, alias="SCAN_MARKETS_COUNT")
    scan_order_by: str = Field(default="volume24hr", alias="SCAN_ORDER_BY")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        extra = "ignore"

    def to_agent_config(self) -> AgentConfig:
        return AgentConfig(
            initial_bankroll=self.bankroll,
            currency=self.currency,
            daily_cost=self.daily_cost_to_cover,
            max_unprofitable_days=self.shutdown_if_unprofitable_days,
            self_preservation=True
        )

    def to_risk_config(self) -> RiskConfig:
        return RiskConfig(
            max_position_pct=self.max_position_pct,
            min_edge_pct=self.min_edge_pct,
            max_open_positions=self.max_open_positions,
            kelly_fraction=self.kelly_fraction,
            min_liquidity=self.min_liquidity,
            min_volume_24h=self.min_volume_24h,
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_total_drawdown_pct=self.max_total_drawdown_pct
        )

    def to_llm_config(self) -> LLMConfig:
        # Auto-detect: prefer LM Studio if user said they use it
        provider = self.llm_provider
        if provider == "auto":
            # Default to auto which will try LM Studio first, then Ollama
            pass
        model = self.lm_studio_model if provider in ["auto", "lm_studio"] else self.ollama_model
        host = self.lm_studio_host if provider in ["auto", "lm_studio"] else self.ollama_host
        return LLMConfig(
            provider=provider,
            model=model,
            host=host,
            ollama_host=self.ollama_host,
            lm_studio_host=self.lm_studio_host
        )

    def to_browser_config(self) -> BrowserConfig:
        return BrowserConfig(
            headless=self.browser_headless,
            persistent_dir=self.browser_persistent_dir,
            stealth=self.browser_stealth,
            timeout=self.browser_timeout
        )

    def to_execution_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            dry_run=self.dry_run,
            browser=self.to_browser_config()
        )

    def to_polymarket_config(self) -> PolymarketConfig:
        return PolymarketConfig(
            private_key=self.polymarket_private_key,
            funder_address=self.polymarket_funder_address,
            chain_id=self.polymarket_chain_id,
            signature_type=self.polymarket_signature_type,
            host=self.polymarket_host,
            gamma_api=self.gamma_api,
            dry_run=self.dry_run
        )

def load_yaml_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}

# Singleton
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_root() -> Path:
    return ROOT
