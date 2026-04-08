from __future__ import annotations

import json
import time
import os
from typing import Any

import requests
import streamlit as st


# ─────────────────────────────────────────────────────────────
# Safe serialization (unchanged, but cleaned)
# ─────────────────────────────────────────────────────────────
def _safe_serialise(obj: Any, max_chars: int = 4000) -> str:
    def _default(o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, (np.bool_,)):
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
# AI ENGINE
# ─────────────────────────────────────────────────────────────
class AIEngine:
    _BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    _MODEL = "llama-3.1-8b-instant"
    _TIMEOUT = 30
    _MAX_TOKENS = 300
    _TEMPERATURE = 0.7

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """
        Automatically loads API key from:
        1. Passed parameter
        2. Streamlit secrets
        3. Environment variables (Render)
        """

        self._api_key = (
            api_key
            or st.secrets.get("GROQ_API_KEY", None)
            or os.getenv("GROQ_API_KEY", "")
        ).strip()

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Set it in Streamlit secrets or Render environment variables."
            )

        self.model = model or self._MODEL

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    # ─────────────────────────────────────────────────────────────
    # Public methods
    # ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        return self._complete(
            [{"role": "user", "content": prompt}]
        )

    def explain_quality(self, metrics: dict) -> str:
        safe = _safe_serialise(metrics)
        prompt = f"""
You are a senior data quality analyst.

Dataset metrics:
{safe}

Explain:
- Overall quality
- Key issues
- Actionable fixes
"""
        return self.generate(prompt)

    def explain_suitability(self, results: dict) -> str:
        safe = _safe_serialise(results)
        prompt = f"""
You are a data science advisor.

Dataset suitability:
{safe}

Explain readiness for ML and next steps.
"""
        return self.generate(prompt)

    def generate_cleaning_summary(self, before: dict, after: dict) -> str:
        b = _safe_serialise(before, 1500)
        a = _safe_serialise(after, 1500)

        prompt = f"""
Before cleaning:
{b}

After cleaning:
{a}

Explain improvements and remaining issues.
"""
        return self.generate(prompt)

    def generate_executive_summary(self, context: dict) -> str:
        safe = _safe_serialise(context)

        prompt = f"""
You are a business analyst.

Context:
{safe}

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
            "model": self.model,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS,
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
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return "Invalid response from AI model."