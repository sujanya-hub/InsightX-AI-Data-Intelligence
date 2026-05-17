from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

# ── safe metrics import ───────────────────────────────────────────────────────
try:
    from metrics_logger import log_metric
    _metrics_ok = True
except Exception:
    _metrics_ok = False

def _log(event: str, data: dict) -> None:
    if not _metrics_ok:
        return
    try:
        log_metric(event, data)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Safe serialization
# ─────────────────────────────────────────────────────────────
def _safe_serialise(obj: Any, max_chars: int = 4000) -> str:
    def _default(o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            if isinstance(o, np.bool_):    return bool(o)
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(o, pd.Series):    return o.tolist()
            if isinstance(o, pd.DataFrame): return o.to_dict(orient="list")
        except ImportError:
            pass
        return str(o)

    try:
        serialised = json.dumps(obj, indent=2, default=_default)
    except Exception:
        serialised = str(obj)

    if len(serialised) > max_chars:
        serialised = serialised[:max_chars] + "\n... [truncated]"
    return serialised


# ─────────────────────────────────────────────────────────────
# Key resolver
# ─────────────────────────────────────────────────────────────
def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key.strip()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return secret.strip()
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────────────────────
class AIEngine:
    _BASE_URL    = "https://api.groq.com/openai/v1/chat/completions"
    _MODEL       = "llama-3.1-8b-instant"
    _TIMEOUT     = 30
    _MAX_TOKENS  = 300
    _TEMPERATURE = 0.7

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = _resolve_api_key(api_key)
        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it as a Render environment variable or in .streamlit/secrets.toml."
            )
        self.model    = model or self._MODEL
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def chat(self, prompt: str, max_tokens: int | None = None, **_kwargs) -> str:
        """Alias for generate(). Accepts same arguments."""
        return self.generate(prompt, max_tokens=max_tokens)

    def generate(self, prompt: str, max_tokens: int | None = None, **_kwargs) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt string.
            max_tokens: Optional override for max tokens. Defaults to _MAX_TOKENS (300).
            **_kwargs: Absorbs any unexpected keyword arguments for forward compatibility.
        """
        return self._complete(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self._MAX_TOKENS,
        )

    def explain_quality(self, metrics: dict) -> str:
        prompt = f"""
You are a senior data quality analyst.

Dataset metrics:
{_safe_serialise(metrics)}

Explain:
- Overall quality
- Key issues
- Actionable fixes
"""
        return self.generate(prompt)

    def explain_suitability(self, results: dict) -> str:
        prompt = f"""
You are a data science advisor.

Dataset suitability:
{_safe_serialise(results)}

Explain readiness for ML and next steps.
"""
        return self.generate(prompt)

    def generate_cleaning_summary(self, before: dict, after: dict) -> str:
        prompt = f"""
Before cleaning:
{_safe_serialise(before, 1500)}

After cleaning:
{_safe_serialise(after, 1500)}

Explain improvements and remaining issues.
"""
        return self.generate(prompt)

    def generate_executive_summary(self, context: dict) -> str:
        prompt = f"""
You are a business analyst.

Context:
{_safe_serialise(context)}

Give executive summary with:
- key insights
- risks
- recommendation
"""
        return self.generate(prompt)

    # ─────────────────────────────────────────────────────────
    # Core API call
    # ─────────────────────────────────────────────────────────
    def _complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens or self._MAX_TOKENS,
            "temperature": self._TEMPERATURE,
        }

        t_start = time.perf_counter()
        http_status = 0
        response_chars = 0
        success = False

        try:
            response = requests.post(
                self._BASE_URL,
                headers=self._headers,
                json=payload,
                timeout=self._TIMEOUT,
            )
            http_status = response.status_code
        except requests.exceptions.RequestException as e:
            _log("llm_call", {
                "llm_time_s":     round(time.perf_counter() - t_start, 3),
                "success":        False,
                "error":          "network_error",
                "model":          self.model,
                "prompt_chars":   len(messages[0]["content"]) if messages else 0,
                "response_chars": 0,
            })
            return f"Network error: {e}"

        if response.status_code != 200:
            _log("llm_call", {
                "llm_time_s":     round(time.perf_counter() - t_start, 3),
                "success":        False,
                "http_status":    http_status,
                "error":          "bad_status",
                "model":          self.model,
                "prompt_chars":   len(messages[0]["content"]) if messages else 0,
                "response_chars": 0,
            })
            return f"API Error {response.status_code}: {response.text}"

        try:
            result = response.json()["choices"][0]["message"]["content"].strip()
            response_chars = len(result)
            success = True
        except Exception:
            result = "Invalid response from AI model."

        _log("llm_call", {
            "llm_time_s":     round(time.perf_counter() - t_start, 3),
            "success":        success,
            "http_status":    http_status,
            "model":          self.model,
            "prompt_chars":   len(messages[0]["content"]) if messages else 0,
            "response_chars": response_chars,
        })

        return result