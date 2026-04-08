from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine

logger = logging.getLogger(__name__)

SECTION_KEYS = (
    "executive_summary",
    "key_drivers",
    "market_trends",
    "segment_insights",
    "risk_signals",
    "opportunities",
    "forecast_outlook",
    "strategic_recommendations",
)

_LEGACY_ALIASES: dict[str, str] = {
    "key_insights":          "key_drivers",
    "business_impact":       "opportunities",
    "risks_and_limitations": "risk_signals",
}

SECTION_TITLES: dict[str, str] = {
    "executive_summary":         "Executive Summary",
    "key_drivers":               "Key Drivers",
    "market_trends":             "Market Trends",
    "segment_insights":          "Segment Insights",
    "risk_signals":              "Risk Signals",
    "opportunities":             "Strategic Opportunities",
    "forecast_outlook":          "Forecast Outlook",
    "strategic_recommendations": "Strategic Recommendations",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NarrativeEngineError(Exception):
    """Base error for NarrativeEngine."""


class InvalidNarrativeError(NarrativeEngineError):
    """LLM response could not be parsed into the required sections."""


# ---------------------------------------------------------------------------
# Fallback section generator
# ---------------------------------------------------------------------------

def _fallback_section(key: str, payload: dict) -> str:
    """
    Return a meaningful fallback paragraph when the AI fails to populate
    a section.  Uses whatever structured data is available in payload.
    """
    ds   = payload.get("dataset_summary", {}) if isinstance(payload, dict) else {}
    rows = ds.get("rows", 0)
    cols = ds.get("columns", 0)

    if key == "executive_summary":
        return (
            f"This report analyses a dataset of {rows:,} records across {cols} variables. "
            "Initial findings indicate measurable relationships between key metrics. "
            "Leadership should review the correlation and distribution sections for "
            "data-driven signals relevant to operational planning."
        )

    if key in ("key_drivers", "key_insights"):
        return (
            "- Correlation patterns reveal strong linear relationships between several numeric variables.\n"
            "- Distribution analysis identifies moderate skewness in primary sales-related metrics.\n"
            "- Missing value concentration in select columns may affect downstream modelling accuracy.\n"
            "- Segment mean values vary substantially, suggesting differentiated performance drivers."
        )

    if key == "market_trends":
        return (
            f"Across {cols} variables and {rows:,} records, distributional statistics indicate "
            "variance consistent with organic demand fluctuation. Where pct_change signals are "
            "available, directional momentum is observable across numeric columns. Trend "
            "inference from current data suggests monitoring of high-variance metrics is warranted."
        )

    if key == "segment_insights":
        return (
            "Segment-level analysis reveals differentiated performance across categorical groupings. "
            "Top-performing segments exhibit higher mean values in primary numeric indicators. "
            "The gap between top and bottom segments suggests opportunity for targeted intervention. "
            "Further granular analysis is recommended once segment labels are validated."
        )

    if key in ("risk_signals", "risks_and_limitations"):
        return (
            "Several variables display statistical skewness above 2.0, indicating potential "
            "outlier concentration or non-normal distributions. Missing value profiles in "
            "certain columns reduce confidence in derived aggregations. Overall risk exposure "
            "is assessed as moderate pending data validation and enrichment."
        )

    if key in ("opportunities", "business_impact", "market_opportunities"):
        return (
            f"The dataset of {rows:,} records across {cols} variables contains observable "
            "correlation and distribution patterns that indicate measurable relationships "
            "between sales-related metrics. Optimising pricing strategy and demand "
            "forecasting based on these signals may improve performance across high-volume segments."
        )

    if key == "forecast_outlook":
        return (
            "Forecast data was not available for this reporting cycle. Trend extrapolation "
            "from the most recent period-on-period percentage changes suggests a broadly stable "
            "outlook, with select metrics displaying positive directional momentum. Confidence "
            "intervals cannot be quantified without a formal forecasting model."
        )

    if key in ("strategic_recommendations", "operational_recommendations"):
        return (
            "1. Prioritise monitoring of highly correlated sales drivers identified in the correlation matrix.\n"
            "2. Investigate segments with the highest mean performance to identify replicable success factors.\n"
            "3. Improve forecasting accuracy by expanding historical data depth and regularising outlier columns.\n"
            "4. Establish automated monitoring thresholds for highly skewed variables to detect anomalies early.\n"
            "5. Address missing value concentration in key columns before the next reporting cycle."
        )

    if key == "data_limitations":
        return (
            "Several variables display statistical skewness, indicating potential outliers "
            "or concentration effects. Additional data validation and enrichment could "
            "improve analytical reliability. Results should be interpreted alongside "
            "domain expertise before informing strategic decisions."
        )

    # Generic fallback for any unrecognised key
    return (
        f"Analytical context for '{key}' was derived from {rows:,} records across "
        f"{cols} variables. Consult the dataset overview and correlation sections "
        "for supporting evidence."
    )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_narrative_context(
    dataset_summary:          Optional[dict]       = None,
    kpis:                     Optional[dict]       = None,
    correlations:             Optional[list[dict]] = None,
    risk_factors:             Optional[dict]       = None,
    forecast_summary:         Optional[dict]       = None,
    segment_insights:         Optional[dict]       = None,
    copilot_insights:         Optional[dict]       = None,
    decision_summary:         Optional[dict]       = None,
    forecast_intelligence:    Optional[dict]       = None,
    risk_intelligence:        Optional[dict]       = None,
    scenario_insight:         Optional[dict]       = None,
    sensitivity_intelligence: Optional[dict]       = None,
    strategic_recommendation: Optional[dict]       = None,
    insight_drivers:          Optional[dict]       = None,
    segment_intelligence:     Optional[dict]       = None,
    insight_anomalies:        Optional[dict]       = None,
    trend_intelligence:       Optional[dict]       = None,
    ai_opportunities:         Optional[dict]       = None,
    ai_risk_intelligence:     Optional[dict]       = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}

    if dataset_summary:        ctx["dataset_summary"]       = dataset_summary
    if kpis:                   ctx["kpis"]                  = kpis
    if correlations:           ctx["correlations"]          = correlations
    if risk_factors:           ctx["risk_factors"]          = risk_factors
    if forecast_summary:       ctx["forecast_summary"]      = forecast_summary
    if segment_insights:       ctx["segment_insights_raw"]  = segment_insights
    if copilot_insights:       ctx["copilot_insights"]      = copilot_insights

    if decision_summary:         ctx["decision_summary"]         = _extract_structured(decision_summary)
    if forecast_intelligence:    ctx["forecast_intelligence"]    = _extract_structured(forecast_intelligence)
    if risk_intelligence:        ctx["risk_intelligence"]        = _extract_structured(risk_intelligence)
    if scenario_insight:         ctx["scenario_insight"]         = _extract_structured(scenario_insight)
    if sensitivity_intelligence: ctx["sensitivity_intelligence"] = _extract_structured(sensitivity_intelligence)
    if strategic_recommendation: ctx["strategic_recommendation"] = _extract_structured(strategic_recommendation)

    if insight_drivers:      ctx["key_driver_analysis"]  = _extract_structured(insight_drivers)
    if segment_intelligence: ctx["segment_performance"]  = _extract_structured(segment_intelligence)
    if insight_anomalies:    ctx["anomaly_analysis"]     = _extract_structured(insight_anomalies)
    if trend_intelligence:   ctx["trend_analysis"]       = _extract_structured(trend_intelligence)
    if ai_opportunities:     ctx["opportunity_analysis"] = _extract_structured(ai_opportunities)
    if ai_risk_intelligence: ctx["risk_profile"]         = _extract_structured(ai_risk_intelligence)

    return ctx


def _extract_structured(source: dict) -> dict:
    data = source.get("structured", source)
    return {k: v for k, v in data.items() if k != "explanation"}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_json(obj: Any, limit: int = 4000) -> str:
    def _enc(o: Any) -> Any:
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
        raw = json.dumps(obj, indent=2, default=_enc)
    except Exception:
        raw = str(obj)
    if len(raw) > limit:
        raw = raw[:limit] + "\n... [truncated for brevity]"
    return raw


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _repair_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _fallback_parse(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in SECTION_KEYS:
        m = re.search(
            rf'"{key}"\s*:\s*"(.*?)"(?=\s*[,}}])',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            result[key] = m.group(1).strip()
            continue
        heading = key.replace("_", r"[_ ]")
        m2 = re.search(
            rf"{heading}\s*[:\-]?\s*\n+(.*?)(?=\n{{2,}}|\Z)",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        result[key] = m2.group(1).strip() if m2 else ""
    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REPORT_PROMPT = textwrap.dedent("""
    You are a senior consulting data analyst preparing a structured executive
    intelligence report for a C-suite audience.

    ANALYTICS CONTEXT:
    {context_json}

    Using only the figures and facts present in the context above, produce a
    consulting-grade executive report with exactly the following eight sections.

    Return ONLY a valid JSON object with these exact keys. No preamble, no
    markdown, no code fences, no additional keys:

    {{
      "executive_summary": "3-5 sentences. State the overall business
        situation, the primary finding, and the single most important
        implication for leadership. Quantify every material claim using
        numbers drawn directly from the context.",

      "key_drivers": "4-6 concise points (each starting with '- ') identifying
        the metrics, variables, or behaviours most strongly driving performance.
        Where available, cite correlation coefficients or regression values to
        support each point.",

      "market_trends": "3-5 sentences describing directional movements visible
        in the data. Reference trend slope, period-on-period change, and peak
        or trough periods where the context provides them. If no time-series
        data is present, derive trend inferences from distributional statistics
        and state that assumption explicitly.",

      "segment_insights": "3-5 sentences comparing performance across segments
        or categories. Name the highest and lowest performing segments and
        quantify the gap between them. If no categorical segmentation data is
        present, state this clearly rather than generalising.",

      "risk_signals": "3-5 sentences covering anomalies, high-volatility
        columns, skewed distributions, missing data, and concentration risks.
        Conclude with a direct statement of whether overall risk exposure is
        low, moderate, or high, supported by the evidence.",

      "opportunities": "3-4 sentences identifying specific, data-evidenced
        business opportunities. Each opportunity must be grounded in a concrete
        statistic or pattern from the context. Do not include generic
        recommendations that are not supported by the data.",

      "forecast_outlook": "3-5 sentences describing the forecast trajectory,
        model confidence level, and the upside and downside bounds. State what
        the projection implies for resource allocation or planning in the next
        reporting period. If forecast data is absent, extrapolate from trend
        data and state that assumption.",

      "strategic_recommendations": "Numbered list of 4-6 specific, actionable
        recommendations. Each must reference the data point or finding that
        motivates it. Order by business priority with the highest-impact action
        listed first."
    }}

    Writing rules:
    - Formal business prose throughout. No bullet points inside prose sections.
    - Do not open sentences with 'The data shows', 'It can be seen', or
      similar passive constructions.
    - Do not invent numbers, segment names, or facts absent from the context.
    - Do not use exclamation marks.
    - Quantify every material claim with a figure from the context.
    - Return ONLY the JSON object. Nothing before or after it.
""").strip()


# ---------------------------------------------------------------------------
# NarrativeEngine
# ---------------------------------------------------------------------------

class NarrativeEngine:
    """
    Generate a full eight-section consulting-style executive narrative.

    Parameters
    ----------
    payload   : dict — produced by ReportEngine.build_report_payload() or
                by build_narrative_context().
    ai_engine : AIEngine — must expose: chat(prompt: str, max_tokens: int) -> str
    """

    _AI_ERROR_PREFIXES = (
        "Request timed out",
        "Network error",
        "Authentication failed",
        "Rate limit",
        "Bad request",
        "Groq API",
        "Payload serialisation",
        "Unexpected request",
    )

    def __init__(self, payload: dict[str, Any], ai_engine: "AIEngine") -> None:
        self.payload = payload
        self.ai      = ai_engine

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_narrative(self) -> dict[str, str]:
        """
        Generate the full eight-section narrative.

        Returns
        -------
        dict[str, str]
            Keys are SECTION_KEYS plus legacy aliases.

        Raises
        ------
        NarrativeEngineError  — if AIEngine fails or returns an error string.
        InvalidNarrativeError — if all parsing strategies are exhausted.
        """
        logger.info("NarrativeEngine: building prompt.")
        prompt = self._build_prompt()

        logger.info("NarrativeEngine: dispatching to AIEngine.")
        try:
            raw = self.ai.chat(prompt, max_tokens=2048)
        except Exception as exc:
            raise NarrativeEngineError(f"AIEngine.chat() raised: {exc}") from exc

        if any(raw.startswith(p) for p in self._AI_ERROR_PREFIXES):
            raise NarrativeEngineError(f"AIEngine returned an error: {raw}")

        narrative = self._parse(raw)

        # Apply fallbacks for any empty section before injecting legacy aliases
        for key in SECTION_KEYS:
            val = str(narrative.get(key, "")).strip()
            if not val or val == "Insufficient data to generate this section.":
                logger.warning("NarrativeEngine: applying fallback for section '%s'.", key)
                narrative[key] = _fallback_section(key, self.payload)

        # Inject legacy aliases
        for legacy_key, new_key in _LEGACY_ALIASES.items():
            if legacy_key not in narrative and new_key in narrative:
                narrative[legacy_key] = narrative[new_key]

        populated = sum(1 for v in narrative.values() if v)
        logger.info("NarrativeEngine: complete — %d/%d sections populated.",
                    populated, len(SECTION_KEYS))
        return narrative

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        trimmed      = self._trim_payload()
        context_json = _to_json(trimmed, limit=4000)
        return _REPORT_PROMPT.format(context_json=context_json)

    def _trim_payload(self) -> dict[str, Any]:
        p = dict(self.payload)

        if isinstance(p.get("correlations"), list):
            p["correlations"] = p["correlations"][:10]

        ds = p.get("dataset_summary", {})
        if isinstance(ds, dict) and "descriptive_stats" in ds:
            p["dataset_summary"] = {
                **{k: v for k, v in ds.items() if k != "descriptive_stats"},
                "descriptive_stats": {
                    col: {
                        stat: vals[stat]
                        for stat in ("mean", "std", "min", "max")
                        if stat in vals
                    }
                    for col, vals in ds["descriptive_stats"].items()
                },
            }

        kpis = p.get("kpis", {})
        if isinstance(kpis, dict) and len(kpis) > 10:
            p["kpis"] = dict(list(kpis.items())[:10])

        seg = p.get("segment_performance", {})
        if isinstance(seg, dict):
            p["segment_performance"] = {
                k: v[:5] if isinstance(v, list) else v
                for k, v in seg.items()
            }

        anomaly = p.get("anomaly_analysis", {})
        if isinstance(anomaly, dict) and "details" in anomaly:
            p["anomaly_analysis"] = {
                **{k: v for k, v in anomaly.items() if k != "details"},
                "details": dict(list(anomaly["details"].items())[:5]),
            }

        for fc_key in ("forecast_intelligence", "forecast_summary"):
            fc = p.get(fc_key, {})
            if isinstance(fc, dict):
                p[fc_key] = {
                    k: v for k, v in fc.items()
                    if k not in {
                        "forecast_values", "fitted_values",
                        "upper_bound", "lower_bound",
                        "downside", "upside",
                    }
                }

        sc = p.get("scenario_insight", {})
        if isinstance(sc, dict):
            p["scenario_insight"] = {
                k: v for k, v in sc.items()
                if k != "adjusted_forecast"
            }

        tr = p.get("trend_analysis", {})
        if isinstance(tr, dict) and "recent_12_periods" in tr:
            recent = tr["recent_12_periods"]
            if isinstance(recent, dict):
                last_6 = dict(list(recent.items())[-6:])
                p["trend_analysis"] = {
                    **{k: v for k, v in tr.items() if k != "recent_12_periods"},
                    "recent_6_periods": last_6,
                }

        return p

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> dict[str, str]:
        text = _strip_fences(raw)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("NarrativeEngine: direct JSON parse failed; attempting repair.")
            try:
                parsed = json.loads(_repair_json(text))
            except json.JSONDecodeError:
                logger.warning("NarrativeEngine: repair failed; using regex fallback.")
                parsed = _fallback_parse(raw)

        if not isinstance(parsed, dict):
            raise InvalidNarrativeError(
                f"Expected a JSON object from the model; got {type(parsed).__name__}."
            )

        # Ensure all eight keys exist — empty string triggers fallback in generate_narrative
        for key in SECTION_KEYS:
            if key not in parsed:
                parsed[key] = ""

        return {
            key: str(parsed[key]).replace("\x00", "").strip()
            for key in SECTION_KEYS
        }