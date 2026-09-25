"""
LLM Provider abstraction - supports Ollama, LM Studio, and any OpenAI-compatible local LLM
All local, no cloud. LM Studio is primary for this user.
"""
import re
import json
from typing import Optional, Dict, Any, Literal, List
from dataclasses import dataclass
import requests
from loguru import logger

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    parsed_json: Optional[Dict] = None


# The placeholder values that mean "no model was pinned". Kept in one place
# because three parts of the system ask the same question: the provider that
# makes the call, the console that reports which model is in use, and the
# diagnostics dashboard.
UNPINNED_MODELS = ("local-model", "", "auto", None)


def is_slow_reasoning_model(model_id: Optional[str]) -> bool:
    """
    Is this SPECIFIC model an R1-style reasoning model?

    They think for thousands of tokens before answering - measured on the
    operator's own machine at ~9 minutes for ONE market - so a 10-minute cycle
    cannot use one. "r1" is matched as a name SEGMENT, so deepseek-r1,
    deepseek-r1-distill-qwen-32b and similar count, while qwen3.8-27b,
    qwen/qwen3-32b and the like do not.

    Judging the whole downloaded list is how a fast model got reported as slow;
    judging only the name of the model actually being called is the rule here
    and everywhere else.
    """
    if not model_id:
        return False
    segments = set(re.split(r"[-/._:\s]+", str(model_id).lower()))
    return "r1" in segments


def choose_loaded_model(models: List[str], configured: Optional[str] = None):
    """
    Which loaded model the agent will actually call. THE ONE DECISION.

    Returns ``(model, reason)`` where reason is written for the operator.

    Rules, in order:

    1. a PINNED model that is loaded wins - the operator asked for it;
    2. otherwise the first loaded model that is not R1-style;
    3. only if every loaded model is R1-style does it take one, and it says so.

    Rule 2 is the fix for what happened on the operator's PC: LM Studio listed
    a reasoning model first, `local-model` meant "auto", and the auto pick took
    it - so every market cost ~9 minutes and a cycle that should take minutes
    could never finish. Whatever the list order happens to be, a 10-minute cycle
    cannot be spent on a model that needs 9 minutes per market, and the operator
    should never have to discover that from a timestamp gap in a log.
    """
    models = [m for m in (models or []) if m]
    if not models:
        return None, "no model is loaded in LM Studio"
    if configured and configured not in UNPINNED_MODELS:
        if configured in models:
            return configured, f"pinned: {configured} is loaded"
        return (models[0],
                f"pinned model {configured} is NOT loaded, so this fell back "
                f"to the first loaded model ({models[0]})")
    fast = [m for m in models if not is_slow_reasoning_model(m)]
    if fast:
        if fast[0] != models[0]:
            return fast[0], (
                f"auto: picked {fast[0]} because it is a fast model, and "
                f"{models[0]} is R1-style (~minutes per market). Pin a model in "
                f"Setup if you want a different one.")
        return fast[0], f"auto: first loaded model is {fast[0]}"
    return models[0], (
        f"auto: EVERY loaded model is R1-style; using {models[0]}, which is "
        f"slow enough that a cycle may not finish in its interval")


# The longest the agent will wait for ONE market's forecast. Not 600 s, which
# is what the OpenAI client does by default and which let a single slow model
# call run for ~9 minutes inside a 10-minute cycle.
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0


class BaseLLMProvider:
    def __init__(self, model: str, host: str, temperature: float = 0.2,
                 max_tokens: int = 1200,
                 timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS):
        self.model = model or "local-model"
        self.host = (host or "http://localhost:1234").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = float(timeout_seconds or DEFAULT_LLM_TIMEOUT_SECONDS)

    def is_available(self) -> bool:
        raise NotImplementedError

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        raise NotImplementedError

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response - handles R1 <think> tags"""
        if not text:
            return None
        # Strip <think>...</think> for R1 models
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except:
            pass
        try:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"JSON extract failed: {e}")
        return None

class OllamaProvider(BaseLLMProvider):
    """Ollama - http://localhost:11434"""
    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except:
            return False

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        try:
            import ollama
            client = ollama.Client(host=self.host)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature, "num_predict": self.max_tokens}
            )
            content = response["message"]["content"]
            return LLMResponse(
                content=content,
                model=self.model,
                provider="ollama",
                parsed_json=self.extract_json(content)
            )
        except ImportError:
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    return LLMResponse(content=content, model=self.model, provider="ollama", parsed_json=self.extract_json(content))
            except Exception as e:
                logger.error(f"Ollama HTTP fallback failed: {e}")
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
        return None

class LMStudioProvider(BaseLLMProvider):
    """LM Studio - OpenAI compatible at http://localhost:1234/v1 (default)"""
    def __init__(self, model: str = "local-model", host: str = "http://localhost:1234", temperature: float = 0.2, max_tokens: int = 1200, api_key: str = "lm-studio", timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS):
        super().__init__(model, host, temperature, max_tokens, timeout_seconds)
        self.api_key = api_key or "lm-studio"
        if not self.host.endswith("/v1"):
            self.base_url = f"{self.host}/v1"
        else:
            self.base_url = self.host
            self.host = self.host.replace("/v1", "")

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=3, headers={"Authorization": f"Bearer {self.api_key}"})
            return resp.status_code == 200
        except:
            try:
                resp = requests.get(f"{self.host}/v1/models", timeout=3)
                return resp.status_code == 200
            except:
                return False

    def list_models(self) -> list:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5, headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning(f"LM Studio list models failed: {e}")
        return []

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        try:
            from openai import OpenAI
            # A bounded wait. Without it the client's own 600 s default applies
            # and one market can hold the whole cycle.
            client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=self.timeout_seconds)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            model_to_use = self.model
            if model_to_use in UNPINNED_MODELS:
                models = self.list_models()
                if models:
                    model_to_use, reason = choose_loaded_model(models, self.model)
                    logger.info(f"LM Studio model choice: {reason}")
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            return LLMResponse(
                content=content,
                model=model_to_use,
                provider="lm_studio",
                parsed_json=self.extract_json(content)
            )
        except ImportError:
            logger.error("openai package not installed, needed for LM Studio")
        except Exception as e:
            logger.error(f"LM Studio chat failed: {e}")
        return None

class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic OpenAI compatible (for any local server)"""
    def __init__(self, model: str, host: str, api_key: str = "not-needed", temperature: float = 0.2, max_tokens: int = 1200):
        safe_host = host or "http://localhost:1234"
        super().__init__(model, safe_host, temperature, max_tokens)
        self.api_key = api_key or "not-needed"
        self.base_url = self.host if self.host.endswith("/v1") else f"{self.host}/v1"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=3, headers={"Authorization": f"Bearer {self.api_key}"})
            return resp.status_code in [200, 401]
        except:
            return False

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            return LLMResponse(content=content, model=self.model, provider="openai_compatible", parsed_json=self.extract_json(content))
        except Exception as e:
            logger.error(f"OpenAI compatible chat failed: {e}")
        return None

class LLMRouter:
    """
    Auto-detects available local LLM:
    1. LM Studio (http://localhost:1234) - PRIORITY for this user
    2. Ollama (http://localhost:11434)
    3. Any OpenAI compatible
    4. Fallback heuristic
    """
    def __init__(self, preferred: str = "auto", ollama_host: str = "http://localhost:11434", lm_studio_host: str = "http://localhost:1234", model: str = "local-model", temperature: float = 0.2, max_tokens: int = 1200, timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS):
        self.timeout_seconds = float(timeout_seconds or DEFAULT_LLM_TIMEOUT_SECONDS)
        self.preferred = preferred or "auto"
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.lm_studio_host = lm_studio_host or "http://localhost:1234"
        self.model = model or "local-model"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider: Optional[BaseLLMProvider] = None
        self._last_detected_models: List[str] = []
        self._detect()

    def _detect(self):
        logger.info(f"LLM Router detecting, preferred={self.preferred}, lm_studio={self.lm_studio_host}, ollama={self.ollama_host}, model={self.model}")

        if self.preferred in ["auto", "lm_studio", "lmstudio"]:
            lm = LMStudioProvider(model=self.model, host=self.lm_studio_host, temperature=self.temperature, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds)
            if lm.is_available():
                models = lm.list_models()
                self._last_detected_models = models
                logger.success(f"LM Studio detected at {self.lm_studio_host}, models: {models[:5]}")
                self.provider = lm
                return
            else:
                logger.info(f"LM Studio not available at {self.lm_studio_host}")

        if self.preferred in ["auto", "ollama"]:
            ollama = OllamaProvider(model=self.model if self.model != "local-model" else "llama3.1:8b", host=self.ollama_host, temperature=self.temperature, max_tokens=self.max_tokens)
            if ollama.is_available():
                logger.success(f"Ollama detected at {self.ollama_host}")
                self.provider = ollama
                return
            else:
                logger.info(f"Ollama not available at {self.ollama_host}")

        if self.preferred == "auto":
            generic = OpenAICompatibleProvider(model=self.model, host=self.lm_studio_host, temperature=self.temperature, max_tokens=self.max_tokens)
            if generic.is_available():
                logger.success(f"Generic OpenAI compatible at {self.lm_studio_host}")
                self.provider = generic
                return

        logger.warning("No local LLM detected, will use heuristic fallback")

    def is_available(self) -> bool:
        return self.provider is not None and self.provider.is_available()

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        if not self.provider:
            self._detect()
        if self.provider:
            return self.provider.chat(prompt, system)
        return None

    def get_provider_name(self) -> str:
        return self.provider.__class__.__name__ if self.provider else "heuristic_fallback"

    def get_provider(self):
        return self.provider

    def get_detected_models(self):
        return self._last_detected_models

_router: Optional[LLMRouter] = None

def get_llm_router(preferred: str = "auto", ollama_host: str = None, lm_studio_host: str = None, model: str = None) -> LLMRouter:
    global _router
    if _router is None:
        from ..config import get_settings
        settings = get_settings()
        _router = LLMRouter(
            preferred=preferred or settings.llm_provider,
            ollama_host=ollama_host or settings.ollama_host,
            lm_studio_host=lm_studio_host or settings.lm_studio_host,
            model=model or settings.lm_studio_model or settings.ollama_model,
            temperature=0.2,
            max_tokens=1200
        )
    return _router
