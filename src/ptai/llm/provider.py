"""
LLM Provider abstraction - supports Ollama, LM Studio, and any OpenAI-compatible local LLM
All local, no cloud. LM Studio is primary for this user.
"""
import re
import json
from datetime import datetime, timezone
import os
import pathlib
import time
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


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The first BALANCED {...} in `text` that actually parses as an object."""
    start = None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start:i + 1])
                except Exception:  # noqa: BLE001 - try the next object, if any
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    The answer, whatever the model wrapped around it.

    A model with thinking ON writes its reasoning first - in `<think>` tags, in
    a bare thinking block, or as plain prose - and a model cut off by
    `max_tokens` mid-thought writes reasoning and NO JSON. Both were parse
    failures before: the old extractor only understood a closed
    `<think>...</think>` pair followed by clean JSON, and the Brain's fallback
    used a greedy `\{.*\}` over raw text, which happily matched from the first
    brace of the reasoning to the last brace of the answer and failed to parse.

    This strips reasoning wrappers and code fences, tries the whole remainder,
    then scans for the first balanced object that parses. `None` means the
    answer really had no JSON in it (truncated, or an error page).
    """
    if not text:
        return None
    cleaned = re.sub(r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>", "",
                     text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001 - fall through to the balanced scan
        pass
    return _first_json_object(cleaned)


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


# One line per detected model per process, not one per detection.
_DETECTIONS_SAID: set = set()


def _selection_file():
    """Where this install remembers which loaded model it settled on."""
    try:
        from ..storage.db import default_data_dir
        return pathlib.Path(default_data_dir()) / "llm_selection.json"
    except Exception:  # noqa: BLE001 - remembering is an optimisation, not a need
        return None


def remembered_model() -> str:
    """The model this install used last time, if it was an auto pick."""
    path = _selection_file()
    if path is None:
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable: no memory, no crash
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("model") or "")


def remember_model(model: str, models: Optional[List[str]] = None) -> None:
    """Record the auto pick, atomically, next to the database."""
    path = _selection_file()
    if path is None or not model:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model,
                   "chosen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "loaded": list(models or [])[:20]}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - a bookkeeping write must never stop a run
        pass


def choose_loaded_model(models: List[str], configured: Optional[str] = None,
                        remembered: Optional[str] = None):
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
        # ONE MODEL PER INSTALL. LM Studio lists its loaded models in an order
        # that is not stable, and this decision is made by every process that
        # builds a router - the agent, the console, the CLI. Two of them could
        # therefore call two different models, which is exactly what the
        # operator's log showed: `LOCAL MODEL: qwen/qwen3-14b` from one process
        # and `Local model: qwen3.8-27b-...` from another, in the same minute.
        # The first auto pick is remembered beside the database and reused while
        # it is still loaded, so the whole install speaks to one model. Pinning
        # a model in Setup still overrides it, and the reason says so.
        if remembered and remembered in fast:
            return remembered, (
                f"auto: {remembered}, the model this install already uses (chosen "
                f"on an earlier run and still loaded). Pin a model in Setup to "
                f"change it.")
        if fast[0] != models[0]:
            return fast[0], (
                f"auto: picked {fast[0]} because it is a fast model, and "
                f"{models[0]} is R1-style (~minutes per market). Pin a model in "
                f"Setup if you want a different one.")
        return fast[0], f"auto: first loaded model is {fast[0]}"
    return models[0], (
        f"auto: EVERY loaded model is R1-style; using {models[0]}, which is "
        f"slow enough that a cycle may not finish in its interval")


def probe_local_model(host: str = "http://localhost:1234",
                      configured_model: Optional[str] = None,
                      timeout: float = 3.0) -> Dict[str, Any]:
    """
    Which model a local server would answer with right now, and why.

    For callers that are not the agent's router yet - the startup panel, the
    doctor - and it uses the SAME `choose_loaded_model` decision the router
    uses, so the panel cannot name a different model than the one the agent
    will call. It never raises: an unreachable server is an answer.

    Returns {"connected", "model", "reason", "models", "where"}.
    """
    provider = LMStudioProvider(model=configured_model or "local-model",
                                host=host, timeout_seconds=timeout)
    where = provider.base_url
    if not provider.is_available():
        return {"connected": False, "model": None, "models": [], "where": where,
                "reason": (f"LM Studio is not answering at {where} - start the "
                           f"server in LM Studio (Developer -> Start Server)")}
    models = provider.list_models()
    _unpinned = not configured_model or configured_model in UNPINNED_MODELS
    _remembered = remembered_model() if _unpinned else ""
    model, reason = choose_loaded_model(models, configured_model,
                                        remembered=_remembered)
    if _unpinned and model and model != _remembered:
        # The panel is allowed to MAKE the choice, and when it does it records
        # it, so the agent's router that starts a minute later calls the same
        # model. Two processes, two auto picks, two names in one log was the
        # operator's 16:53 report.
        remember_model(model, models)
    return {"connected": True, "model": model, "reason": reason,
            "models": models, "where": where}


# The longest the agent will wait for ONE market's forecast. Not 600 s, which
# is what the OpenAI client does by default and which let a single slow model
# call run for ~9 minutes inside a 10-minute cycle.
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0


class BaseLLMProvider:
    def __init__(self, model: str, host: str, temperature: float = 0.2,
                 max_tokens: int = 1200,
                 timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS,
                 thinking: bool = False):
        self.model = model or "local-model"
        # Whether this provider asks the server to let the model reason first.
        # LM Studio exposes Qwen3-style hybrid thinking as the chat-template
        # kwarg `enable_thinking`, and nothing in this codebase ever sent it, so
        # whatever the loaded model's own default was is what happened - with
        # PTAI's prompt telling the model not to think on top of it.
        self.thinking = bool(thinking)
        self.host = (host or "http://localhost:1234").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = float(timeout_seconds or DEFAULT_LLM_TIMEOUT_SECONDS)
        # The outcome of the last call this provider made: which model, how
        # long, and what went wrong if anything. The router reads it to count
        # the cycle's model work, and the operator's log gets the same facts.
        # Before this, `LLMResponse.model` - the id the SERVER says answered -
        # was read nowhere in the codebase, so no surface could name the model.
        self.last_call: Dict[str, Any] = {}

    def is_available(self) -> bool:
        raise NotImplementedError

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        raise NotImplementedError

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from a model's answer, reasoning wrappers and all."""
        return extract_json_object(text)

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
            self.last_call = {"model": self.model, "ok": True, "error": "",
                              "seconds": None}
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
                    self.last_call = {"model": self.model, "ok": True, "error": "",
                                      "seconds": None}
                    return LLMResponse(content=content, model=self.model, provider="ollama", parsed_json=self.extract_json(content))
            except Exception as e:
                self.last_call = {"model": self.model, "ok": False,
                                  "error": f"{type(e).__name__}: {e}",
                                  "seconds": None}
                logger.error(f"Ollama HTTP fallback failed: {type(e).__name__}: {e}")
        except Exception as e:
            self.last_call = {"model": self.model, "ok": False,
                              "error": f"{type(e).__name__}: {e}", "seconds": None}
            logger.error(f"Ollama chat failed: {type(e).__name__}: {e}")
        return None

class LMStudioProvider(BaseLLMProvider):
    """LM Studio - OpenAI compatible at http://localhost:1234/v1 (default)"""
    def __init__(self, model: str = "local-model", host: str = "http://localhost:1234", temperature: float = 0.2, max_tokens: int = 1200, api_key: str = "lm-studio", timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS, thinking: bool = False):
        super().__init__(model, host, temperature, max_tokens, timeout_seconds,
                         thinking=thinking)
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
        started = time.time()
        model_to_use = self.model
        requested = self.model
        try:
            from openai import OpenAI
            # A bounded wait. Without it the client's own 600 s default applies
            # and one market can hold the whole cycle.
            #
            # `max_retries=0` is not a detail. The OpenAI client retries twice by
            # default, and each retry re-sends the whole prompt, so a "180 second"
            # limit was really 3 x 180 = ~540 seconds of the user's cycle. Their
            # log shows exactly that: 22:12:01 -> 22:21:08 = 547 s per market,
            # and 848 s on the first one. A limit that the client is allowed to
            # multiply by three is not a limit.
            client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=self.timeout_seconds, max_retries=0)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            if model_to_use in UNPINNED_MODELS:
                models = self.list_models()
                if models:
                    model_to_use, reason = choose_loaded_model(
                        models, self.model, remembered=remembered_model())
                    # Resolved once, then remembered: the id this provider calls
                    # from now on is the one it logged, so a later call cannot
                    # silently pick a different model.
                    self.model = model_to_use
                    logger.info(f"LM Studio model choice: {reason}")

            # THINKING IS A REQUEST-SIDE FLAG, NOT A PROMPT.
            #
            # The operator asked why thinking was off. Nothing in PTAI ever sent
            # `enable_thinking`, so the loaded model's own default applied - and
            # PTAI's prompt then told it "no chain-of-thought, no <think> tag".
            # Now the one setting `llm_thinking` decides, and the kwarg goes in
            # the request so LM Studio's template actually honours it. Sent only
            # when True: a server that does not understand the kwarg must not
            # reject every call because of a feature it was not using anyway.
            extra_body = {}
            if self.thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}

            started = time.time()
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra_body or None,
            )
            elapsed = time.time() - started
            # The id the SERVER reports, which is the only one that is a fact
            # about the answer. LM Studio echoes the requested id; a proxy may
            # not, and then the log should show what actually answered.
            answered_model = (getattr(response, "model", None) or model_to_use)
            if elapsed > max(20.0, self.timeout_seconds * 0.5):
                logger.warning(
                    f"LM Studio took {elapsed:.0f}s for one forecast with "
                    f"{model_to_use} (limit {self.timeout_seconds:.0f}s)")
            content = response.choices[0].message.content
            logger.info(
                f"LM Studio answered with '{answered_model}' in {elapsed:.1f}s"
                + (f" (requested '{requested}')" if requested != answered_model
                   else ""))
            self.last_call = {"model": answered_model, "seconds": round(elapsed, 2),
                              "ok": True, "error": ""}
            return LLMResponse(
                content=content,
                model=answered_model,
                provider="lm_studio",
                parsed_json=self.extract_json(content)
            )
        except ImportError:
            self.last_call = {"model": model_to_use, "ok": False,
                              "seconds": round(time.time() - started, 2),
                              "error": "the openai package is not installed"}
            logger.error("openai package not installed, needed for LM Studio")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.last_call = {"model": model_to_use, "ok": False,
                              "seconds": round(time.time() - started, 2),
                              "error": error}
            logger.error(
                f"LM Studio chat failed after {time.time() - started:.0f}s with "
                f"{model_to_use} (limit {self.timeout_seconds:.0f}s, no retry): {error}")
        return None

class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic OpenAI compatible (for any local server)"""
    def __init__(self, model: str, host: str, api_key: str = "not-needed", temperature: float = 0.2, max_tokens: int = 1200, timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS, thinking: bool = False):
        safe_host = host or "http://localhost:1234"
        super().__init__(model, safe_host, temperature, max_tokens, timeout_seconds,
                         thinking=thinking)
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
            # Same bound as the LM Studio path: this class had no timeout at all,
            # so any OpenAI-compatible server inherited the client's 600 s default
            # and could hold a whole cycle on one market.
            client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=self.timeout_seconds, max_retries=0)
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
            answered = getattr(response, "model", None) or self.model
            self.last_call = {"model": answered, "ok": True, "error": "",
                              "seconds": None}
            return LLMResponse(content=content, model=answered, provider="openai_compatible", parsed_json=self.extract_json(content))
        except Exception as e:
            self.last_call = {"model": self.model, "ok": False,
                              "error": f"{type(e).__name__}: {e}", "seconds": None}
            logger.error(f"OpenAI compatible chat failed: {type(e).__name__}: {e}")
        return None

class LLMRouter:
    """
    Auto-detects available local LLM:
    1. LM Studio (http://localhost:1234) - PRIORITY for this user
    2. Ollama (http://localhost:11434)
    3. Any OpenAI compatible
    4. Fallback heuristic
    """
    def __init__(self, preferred: str = "auto", ollama_host: str = "http://localhost:11434", lm_studio_host: str = "http://localhost:1234", model: str = "local-model", temperature: float = 0.2, max_tokens: int = 1200, timeout_seconds: Optional[float] = DEFAULT_LLM_TIMEOUT_SECONDS, thinking: bool = False):
        self.timeout_seconds = float(timeout_seconds or DEFAULT_LLM_TIMEOUT_SECONDS)
        # `thinking` reaches every provider this router builds, so the whole
        # install answers "is the model reasoning first?" the same way.
        self.thinking = bool(thinking)
        self.preferred = preferred or "auto"
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.lm_studio_host = lm_studio_host or "http://localhost:1234"
        self.model = model or "local-model"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider: Optional[BaseLLMProvider] = None
        self._last_detected_models: List[str] = []
        # Detection is re-run when a call is attempted and no provider was
        # found (the operator may start LM Studio mid-run), but NOT once per
        # market: the honest reason a whole run had no model is one line, and
        # repeating it per market is how it gets lost.
        self._last_detect_at = 0.0
        self._detect_retry_seconds = 60.0
        self._detect_failures = 0
        # WHICH MODEL the agent calls, decided at detection time and named from
        # then on. It used to be decided lazily inside the first chat() call, so
        # every surface could only report the CONFIGURED value - the
        # "local-model" placeholder - while the real id was never printed
        # anywhere, and `LLMResponse.model` was never read at all. The operator
        # went looking for their model id in the log and it was nowhere.
        self.active_model: Optional[str] = None
        self.active_model_reason: str = ""
        # What this process's model actually did: calls, answers, failures, and
        # which ids answered. Reset per cycle by the agent, so a cycle can be
        # described in its own words rather than in the process's lifetime.
        self.usage: Dict[str, Any] = {
            "calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
            "models": {}, "last_model": "", "last_error": "",
        }
        self._detect()

    def _detect(self, quiet: bool = False):
        """
        Find a local model server and decide WHICH model will be called.

        `quiet` is for the retries after a failure: the first miss is reported
        in full, later ones at DEBUG (a success is always reported, quietly or
        not - a server that came up mid-run is news).
        """
        self._last_detect_at = time.monotonic()
        self._detect_failures += 1

        def _say(level: str, message: str) -> None:
            if quiet and level in ("info", "warning"):
                logger.debug(message)
            else:
                getattr(logger, level)(message)

        _say("info", f"LLM Router detecting, preferred={self.preferred}, lm_studio={self.lm_studio_host}, ollama={self.ollama_host}, model={self.model}")

        if self.preferred in ["auto", "lm_studio", "lmstudio"]:
            lm = LMStudioProvider(model=self.model, host=self.lm_studio_host, temperature=self.temperature, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds, thinking=self.thinking)
            if lm.is_available():
                models = lm.list_models()
                self._last_detected_models = models
                _unpinned = not self.model or self.model in UNPINNED_MODELS
                _remembered = remembered_model() if _unpinned else ""
                chosen, reason = choose_loaded_model(models, self.model,
                                                     remembered=_remembered)
                self.active_model = chosen
                self.active_model_reason = reason
                if chosen:
                    # The provider calls THIS model from now on, so the id in
                    # the log is the id being called rather than a placeholder.
                    lm.model = chosen
                    if _unpinned and chosen != _remembered:
                        # Remember the AUTO pick so the next process - the
                        # console, the CLI, the agent loop - calls the same
                        # model instead of whatever LM Studio happens to list
                        # first. A pinned model is never remembered over.
                        remember_model(chosen, models)
                # SAID ONCE PER PROCESS. The console builds a router (and so runs
                # a detection) per request, and the log filled with the same
                # "LOCAL MODEL:" line - sometimes naming a different model than
                # the agent's own line, because each process made its own auto
                # pick.
                # Keyed on the MODEL, not on the sentence around it: the fact
                # is "this process calls X". A second detection that picks the
                # same model says nothing, whatever its reason now reads.
                _key = chosen or "none"
                if _key in _DETECTIONS_SAID:
                    logger.debug(f"LOCAL MODEL: {chosen or 'none'} (lm_studio at "
                                 f"{lm.base_url}) - {reason} (already reported)")
                else:
                    _DETECTIONS_SAID.add(_key)
                    _say("success",
                         f"LOCAL MODEL: {chosen or 'none'} (lm_studio at {lm.base_url})"
                         f" - {reason}")
                self.provider = lm
                return
            else:
                _say("info",
                     f"LOCAL MODEL: none - LM Studio is not answering at "
                     f"{self.lm_studio_host}")

        if self.preferred in ["auto", "ollama"]:
            ollama = OllamaProvider(model=self.model if self.model != "local-model" else "llama3.1:8b", host=self.ollama_host, temperature=self.temperature, max_tokens=self.max_tokens, thinking=self.thinking)
            if ollama.is_available():
                self.active_model = ollama.model
                self.active_model_reason = "ollama is the detected local model server"
                _say("success", f"LOCAL MODEL: {ollama.model} (ollama at {ollama.host})")
                self.provider = ollama
                return
            else:
                _say("info", f"LOCAL MODEL: none - Ollama is not answering at {self.ollama_host}")

        if self.preferred == "auto":
            generic = OpenAICompatibleProvider(model=self.model, host=self.lm_studio_host, temperature=self.temperature, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds, thinking=self.thinking)
            if generic.is_available():
                self.active_model = generic.model
                self.active_model_reason = "an OpenAI-compatible server answered"
                _say("success", f"LOCAL MODEL: {generic.model} (openai_compatible at {generic.base_url})")
                self.provider = generic
                return

        self.active_model = None
        self.active_model_reason = (
            f"no local model server answered (lm_studio {self.lm_studio_host}, "
            f"ollama {self.ollama_host})")
        _say("warning",
             f"LOCAL MODEL: none - {self.active_model_reason}; every market is "
             f"priced without a model, and the log says so on each market")

    def is_available(self) -> bool:
        return self.provider is not None and self.provider.is_available()

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        if not self.provider:
            # Retry the detection, but on a leash: once a minute, and only the
            # first miss is printed in full. Without this a server that is down
            # produced a detect line (and a "no model" warning) for every market
            # in the scan.
            if time.monotonic() - self._last_detect_at >= self._detect_retry_seconds:
                self._detect(quiet=self._detect_failures > 0)
        if not self.provider:
            self.note_call(model=None, ok=False,
                           error=self.active_model_reason or "no model answered")
            return None
        response = self.provider.chat(prompt, system)
        last = dict(getattr(self.provider, "last_call", None) or {})
        model = (last.get("model")
                 or (response.model if response else None)
                 or self.active_model or getattr(self.provider, "model", None))
        self.note_call(model=model, ok=response is not None,
                       error=last.get("error", ""),
                       seconds=last.get("seconds"))
        return response

    def note_call(self, model: Optional[str] = None, ok: bool = False,
                  error: str = "", seconds: Optional[float] = None) -> None:
        """Count one model call, and remember what it did. Never raises."""
        try:
            self.usage["calls"] = int(self.usage.get("calls", 0)) + 1
            self.usage["answered" if ok else "failed"] = (
                int(self.usage.get("answered" if ok else "failed", 0)) + 1)
            if seconds:
                self.usage["seconds"] = round(
                    float(self.usage.get("seconds", 0.0)) + float(seconds), 2)
            if model:
                models = self.usage.setdefault("models", {})
                models[str(model)] = int(models.get(str(model), 0)) + 1
                self.usage["last_model"] = str(model)
            if error:
                self.usage["last_error"] = str(error)[:300]
        except Exception as e:  # noqa: BLE001 - counting must never break a call
            logger.debug(f"Could not record the model call: {e}")

    def reset_usage(self) -> None:
        """Start a fresh count - the agent calls this at the top of a cycle."""
        self.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                      "models": {}, "last_model": "", "last_error": ""}

    def usage_report(self) -> Dict[str, Any]:
        return {"calls": int(self.usage.get("calls", 0)),
                "answered": int(self.usage.get("answered", 0)),
                "failed": int(self.usage.get("failed", 0)),
                "seconds": float(self.usage.get("seconds", 0.0)),
                "models": dict(self.usage.get("models", {})),
                "last_model": self.usage.get("last_model", ""),
                "last_error": self.usage.get("last_error", "")}

    def describe(self) -> str:
        """
        One factual sentence: which model the agent calls, where it is, and why
        that one. Used by the startup panel and the per-cycle line, so the two
        cannot disagree with the provider that actually makes the calls.
        """
        if not self.provider:
            reason = self.active_model_reason or "no local model server answered"
            return f"none - {reason}; markets are priced without a model"
        model = (self.active_model or getattr(self.provider, "model", None)
                 or "unresolved")
        where = (getattr(self.provider, "base_url", None)
                 or getattr(self.provider, "host", ""))
        reason = self.active_model_reason or "no reason recorded"
        return f"{model} ({self.get_provider_name()} at {where}) - {reason}"

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
            max_tokens=1200,
            thinking=bool(getattr(settings, "llm_thinking", False)),
        )
    return _router
