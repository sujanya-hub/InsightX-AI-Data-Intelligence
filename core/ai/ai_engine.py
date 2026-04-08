from __future__ import annotations

import json
import time
from typing import Any

import requests


def _safe_serialise(obj: Any, max_chars: int = 4000) -> str:
    """
    Convert any object to a compact, truncated string safe for prompt injection.
    Handles numpy scalars, pandas objects, and other non-standard types.
    """
    def _default(o: Any) -> Any:
        # numpy scalar types
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
        # pandas types
        try:
            import pandas as pd
            if isinstance(o, pd.Series):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="list")
        except ImportError:
            pass
        # fallback
        return str(o)

    try:
        serialised = json.dumps(obj, indent=2, default=_default)
    except Exception:
        serialised = str(obj)

    if len(serialised) > max_chars:
        serialised = serialised[:max_chars] + "\n... [truncated]"

    return serialised


class AIEngine:
    _BASE_URL  = "https://api.groq.com/openai/v1/chat/completions"
    _MODEL = "llama-3.1-8b-instant"
    _TIMEOUT   = 30
    _MAX_TOKENS_DEFAULT = 300
    _TEMPERATURE        = 0.7

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
    ) -> None:
        self._api_key = api_key.strip()
        self.model    = model
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core completion primitives
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = _MAX_TOKENS_DEFAULT) -> str:
        """Send a single user prompt and return the assistant reply as a string."""
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(max_tokens, self._MAX_TOKENS_DEFAULT),
        )

    def chat(self, prompt: str, max_tokens: int = _MAX_TOKENS_DEFAULT) -> str:
        """Alias for generate — preserves backward compatibility with all callers."""
        return self.generate(prompt, max_tokens=max_tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Domain-specific public methods (called by tab modules)
    # ──────────────────────────────────────────────────────────────────────────

    def explain_quality(self, metrics: dict) -> str:
        """Called by tab_data_readiness. Receives quality metrics, returns explanation."""
        safe_metrics = _safe_serialise(metrics)
        prompt = (
            "You are a senior data quality analyst.\n\n"
            "Dataset quality metrics:\n"
            f"{safe_metrics}\n\n"
            "Write 3-5 sentences covering: overall quality assessment, "
            "critical issues (missing values, duplicates, outliers), "
            "and specific actionable next steps. Be precise and data-driven."
        )
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._MAX_TOKENS_DEFAULT,
        )

    def explain_suitability(self, results: dict) -> str:
        """Called by tab_data_readiness. Receives suitability results, returns recommendation."""
        safe_results = _safe_serialise(results)
        prompt = (
            "You are a senior data science advisor.\n\n"
            "Dataset suitability assessment:\n"
            f"{safe_results}\n\n"
            "Write 3-5 sentences covering: suitability for ML or analytics, "
            "key strengths and limitations, and concrete next steps before modelling. "
            "Be direct and practical."
        )
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._MAX_TOKENS_DEFAULT,
        )

    def generate_cleaning_summary(self, before: dict, after: dict) -> str:
        """Called by tab_data_readiness. Receives before/after snapshots, returns summary."""
        safe_before = _safe_serialise(before, max_chars=1800)
        safe_after  = _safe_serialise(after,  max_chars=1800)
        prompt = (
            "You are a senior data engineer reviewing an automated cleaning pipeline.\n\n"
            f"BEFORE cleaning:\n{safe_before}\n\n"
            f"AFTER cleaning:\n{safe_after}\n\n"
            "Write 3-5 sentences covering: what was fixed or removed, "
            "measurable quality improvement, and any remaining concerns. "
            "Use specific numbers where available."
        )
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._MAX_TOKENS_DEFAULT,
        )

    def generate_executive_summary(self, context: dict) -> str:
        """Called by tab_dashboard and tab_decision. Receives analytics context, returns executive brief."""
        safe_context = _safe_serialise(context)
        prompt = (
            "You are a senior business analyst preparing an executive briefing.\n\n"
            f"ANALYTICS CONTEXT:\n{safe_context}\n\n"
            "Write 4-6 sentences covering: key findings and trends, "
            "notable risks or anomalies, strategic implications, "
            "and one clear prioritised recommended action. "
            "Use boardroom-appropriate language. No bullet points."
        )
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._MAX_TOKENS_DEFAULT,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal HTTP layer
    # ──────────────────────────────────────────────────────────────────────────

    def _complete(
        self,
        messages: list[dict],
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        _retry: bool = True,
    ) -> str:
        safe_max_tokens = min(max(1, max_tokens), self._MAX_TOKENS_DEFAULT)

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  safe_max_tokens,
            "temperature": self._TEMPERATURE,
        }

        # Validate payload is JSON-serialisable before sending
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            return f"Payload serialisation error — could not build a valid request: {exc}"

        try:
            response = requests.post(
                self._BASE_URL,
                headers=self._headers,
                json=payload,
                timeout=self._TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return "Request timed out. The Groq API did not respond in time. Please try again."
        except requests.exceptions.ConnectionError:
            return "Network error. Unable to reach the Groq API. Check your internet connection."
        except requests.exceptions.RequestException as exc:
            return f"Unexpected request error: {exc}"

        if response.status_code == 400:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            return f"Bad request (HTTP 400): {detail}"

        if response.status_code == 401:
            return (
                "Authentication failed (HTTP 401). "
                "Your Groq API key is invalid or has not been activated."
            )

        if response.status_code == 429:
            if _retry:
                time.sleep(2.0)
                return self._complete(messages, max_tokens=safe_max_tokens, _retry=False)
            return (
                "Rate limit exceeded (HTTP 429). "
                "Too many requests have been sent to the Groq API. Please wait and try again."
            )

        if response.status_code >= 500:
            if _retry:
                time.sleep(1.5)
                return self._complete(messages, max_tokens=safe_max_tokens, _retry=False)
            return (
                f"Groq API server error (HTTP {response.status_code}). "
                "The service is temporarily unavailable. Please try again later."
            )

        if not response.ok:
            return (
                f"Request failed (HTTP {response.status_code}). "
                "An unexpected error was returned by the Groq API."
            )

        try:
            data    = response.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip() if content else "The model returned an empty response."
        except (KeyError, IndexError):
            return "Unexpected response structure returned by the Groq API."
        except ValueError:
            return "Failed to parse the response from the Groq API."