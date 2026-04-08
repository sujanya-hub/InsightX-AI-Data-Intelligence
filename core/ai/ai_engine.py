from __future__ import annotations

import json
import os
from typing import Any

import requests


# ─────────────────────────────────────────────────────────────
# Safe serialization
# ─────────────────────────────────────────────────────────────
def _safe_serialise(obj: Any, max_chars: int = 4000) -> str:
    def _default(o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
        except ImportError:
            pass

        try:
            import pandas as pd
            if isinstance(o, pd.Series):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="list")
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
# Key resolver — never imports streamlit at module level
# ─────────────────────────────────────────────────────────────
def _resolve_api_key(api_key: str | None) -> str:
    """
    Resolution order (first non-empty value wins):
      1. Explicitly passed api_key argument
      2. GROQ_API_KEY environment variable   ← works on Render
      3. st.secrets["GROQ_API_KEY"]          ← works locally
    Returns "" if nothing is found; never raises.
    """
    if api_key:
        return api_key.strip()

    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key

    # Attempt Streamlit secrets — may not exist on Render
    try:
        import streamlit as st
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return secret.strip()
    except Exception:
        pass  # secrets.toml absent — totally fine on Render

    return ""


# ─────────────────────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────────────────────
class AIEngine:
    _BASE_URL   = "https://api.groq.com/openai/v1/chat/completions"
    _MODEL      = "llama-3.1-8b-instant"
    _TIMEOUT    = 30
    _MAX_TOKENS = 300
    _TEMPERATURE = 0.7

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = _resolve_api_key(api_key)

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it as a Render environment variable or in .streamlit/secrets.toml."
            )

        self.model = model or self._MODEL
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        return self._complete([{"role": "user", "content": prompt}])

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

    # ─────────────────────────────────────────────────────────────
    # Core API call
    # ─────────────────────────────────────────────────────────────
    def _complete(self, messages: list[dict]) -> str:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self._MAX_TOKENS,
            "temperature": self._TEMPERATURE,
        }

        try:
            response = requests.post(
                self._BASE_URL,
                headers=self._headers,
                json=payload,
                timeout=self._TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            return f"Network error: {e}"

        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return "Invalid response from AI model."