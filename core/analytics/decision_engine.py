"""
Decision Intelligence Engine
Pure analytics layer — no Streamlit dependencies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    model_name: str
    fitted_values: np.ndarray
    forecast_values: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    rmse: float
    confidence_score: float          # 0–100
    all_rmse: dict[str, float] = field(default_factory=dict)
    horizon: int = 6


@dataclass
class RiskProjection:
    downside: np.ndarray
    upside: np.ndarray
    risk_adjusted: np.ndarray
    volatility: float
    concentration_risk: float
    anomaly_pressure: float
    stability_score: float           # 0–10


@dataclass
class ScenarioResult:
    adjusted_forecast: np.ndarray
    risk_impact: float
    revenue_delta: float
    params: dict[str, float]


@dataclass
class SensitivityResult:
    elasticities: dict[str, float]
    ranked: list[tuple[str, float]]  # sorted descending by |elasticity|


@dataclass
class DecisionSignal:
    signal: str
    strategy: str
    rationale: str
    severity: str                    # "positive" | "warning" | "critical"


@dataclass
class DecisionScorecard:
    growth_strength: float           # 0–10
    risk_exposure: float             # 0–10
    stability_score: float           # 0–10
    forecast_confidence: float       # 0–100 %
    strategic_readiness: float       # 0–10
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_target_series(df: pd.DataFrame) -> pd.Series:
    """Pick the most variance-rich numeric column as analysis target."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("Dataset has no numeric columns.")
    variances = df[num_cols].var().fillna(0)
    return df[variances.idxmax()].dropna().reset_index(drop=True)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    n = min(len(actual), len(predicted))
    return float(np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2)))


def _linear_trend(series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series, 1)
    fitted = np.polyval(coeffs, x)
    future_x = np.arange(len(series), len(series) + horizon)
    forecast = np.polyval(coeffs, future_x)
    return fitted, forecast


def _exponential_smoothing(series: np.ndarray, horizon: int,
                            alpha: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    fitted = np.zeros(len(series))
    fitted[0] = series[0]
    for i in range(1, len(series)):
        fitted[i] = alpha * series[i] + (1 - alpha) * fitted[i - 1]
    last = fitted[-1]
    forecast = np.array([last] * horizon)
    return fitted, forecast


def _moving_average(series: np.ndarray, horizon: int,
                    window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    w = min(window, len(series))
    fitted = pd.Series(series).rolling(w, min_periods=1).mean().values
    forecast = np.array([fitted[-1]] * horizon)
    return fitted, forecast


# ---------------------------------------------------------------------------
# Section 1 — Multi-Model Forecast Engine
# ---------------------------------------------------------------------------

def run_forecast(df: pd.DataFrame, horizon: int = 6) -> ForecastResult:
    series = _detect_target_series(df).values.astype(float)

    lin_fit, lin_fc = _linear_trend(series, horizon)
    exp_fit, exp_fc = _exponential_smoothing(series, horizon)
    ma_fit, ma_fc = _moving_average(series, horizon)

    rmse_lin = _rmse(series, lin_fit)
    rmse_exp = _rmse(series, exp_fit)
    rmse_ma = _rmse(series, ma_fit)

    all_rmse = {"Linear Trend": rmse_lin,
                "Exponential Smoothing": rmse_exp,
                "Moving Average": rmse_ma}

    best_name = min(all_rmse, key=all_rmse.get)
    best_fc_map = {
        "Linear Trend": (lin_fit, lin_fc),
        "Exponential Smoothing": (exp_fit, exp_fc),
        "Moving Average": (ma_fit, ma_fc),
    }
    best_fit, best_fc = best_fc_map[best_name]
    best_rmse = all_rmse[best_name]

    # Confidence interval — residual std
    residuals = series - best_fit
    std = np.std(residuals) if np.std(residuals) > 0 else 1.0
    z = 1.645  # 90 % CI
    lower = best_fc - z * std
    upper = best_fc + z * std

    # Confidence score: inversely related to CV of forecast
    mean_fc = np.mean(np.abs(best_fc)) if np.mean(np.abs(best_fc)) > 0 else 1
    cv = std / mean_fc
    confidence_score = float(np.clip(100 * (1 - cv), 10, 99))

    return ForecastResult(
        model_name=best_name,
        fitted_values=best_fit,
        forecast_values=best_fc,
        lower_bound=lower,
        upper_bound=upper,
        rmse=best_rmse,
        confidence_score=confidence_score,
        all_rmse=all_rmse,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Section 2 — Risk Projection
# ---------------------------------------------------------------------------

def run_risk_projection(df: pd.DataFrame,
                        forecast: ForecastResult) -> RiskProjection:
    series = _detect_target_series(df).values.astype(float)

    # Volatility = rolling std / mean
    roll_std = pd.Series(series).rolling(min(5, len(series)), min_periods=1).std().fillna(0)
    volatility = float(roll_std.mean() / (np.mean(np.abs(series)) + 1e-9))

    # Concentration risk = Herfindahl-like index on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        col_vars = df[num_cols].var().fillna(0)
        total = col_vars.sum() + 1e-9
        weights = col_vars / total
        concentration_risk = float((weights ** 2).sum())  # 0–1
    else:
        concentration_risk = 1.0

    # Anomaly pressure = fraction of values beyond 2 std
    mu, sigma = np.mean(series), np.std(series) + 1e-9
    anomaly_pressure = float(np.mean(np.abs(series - mu) > 2 * sigma))

    fc = forecast.forecast_values
    shock = volatility * np.std(fc) if np.std(fc) > 0 else np.std(series)

    downside = fc - 1.5 * shock
    upside = fc + 1.0 * shock
    risk_adjusted = fc - (concentration_risk * shock * 0.5
                          + anomaly_pressure * shock * 0.3)

    # Stability score 0–10
    raw_instability = volatility + concentration_risk + anomaly_pressure
    stability_score = float(np.clip(10 * (1 - raw_instability / 3), 0, 10))

    return RiskProjection(
        downside=downside,
        upside=upside,
        risk_adjusted=risk_adjusted,
        volatility=volatility,
        concentration_risk=concentration_risk,
        anomaly_pressure=anomaly_pressure,
        stability_score=stability_score,
    )


# ---------------------------------------------------------------------------
# Section 3 — Scenario Simulator
# ---------------------------------------------------------------------------

def run_scenario(df: pd.DataFrame,
                 forecast: ForecastResult,
                 risk: RiskProjection,
                 growth_pct: float = 0.0,
                 volatility_adj_pct: float = 0.0,
                 concentration_reduction_pct: float = 0.0,
                 demand_shock_pct: float = 0.0) -> ScenarioResult:

    fc = forecast.forecast_values.copy()
    base_mean = np.mean(np.abs(fc)) + 1e-9

    # Apply adjustments
    fc_adj = fc * (1 + growth_pct / 100)
    vol_factor = 1 + volatility_adj_pct / 100
    conc_factor = max(0.0, 1 - concentration_reduction_pct / 100)
    demand_factor = 1 + demand_shock_pct / 100
    fc_adj = fc_adj * demand_factor

    adj_vol = risk.volatility * vol_factor
    adj_conc = risk.concentration_risk * conc_factor
    shock = adj_vol * np.std(fc_adj) if np.std(fc_adj) > 0 else 1.0
    risk_impact = float(adj_conc * shock + risk.anomaly_pressure * shock * 0.3)

    revenue_delta = float(np.sum(fc_adj) - np.sum(fc))

    params = {
        "growth_pct": growth_pct,
        "volatility_adj_pct": volatility_adj_pct,
        "concentration_reduction_pct": concentration_reduction_pct,
        "demand_shock_pct": demand_shock_pct,
    }

    return ScenarioResult(
        adjusted_forecast=fc_adj,
        risk_impact=risk_impact,
        revenue_delta=revenue_delta,
        params=params,
    )


# ---------------------------------------------------------------------------
# Section 4 — Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_sensitivity(df: pd.DataFrame,
                    forecast: ForecastResult,
                    risk: RiskProjection,
                    delta: float = 10.0) -> SensitivityResult:
    """Compute elasticity of forecast sum w.r.t. each lever."""

    base_scenario = run_scenario(df, forecast, risk)
    base_sum = float(np.sum(base_scenario.adjusted_forecast))

    def _elasticity(param: str) -> float:
        kwargs = {"growth_pct": 0.0, "volatility_adj_pct": 0.0,
                  "concentration_reduction_pct": 0.0, "demand_shock_pct": 0.0}
        kwargs[param] = delta
        sc = run_scenario(df, forecast, risk, **kwargs)
        new_sum = float(np.sum(sc.adjusted_forecast))
        if base_sum == 0:
            return 0.0
        return ((new_sum - base_sum) / base_sum) / (delta / 100)

    elasticities = {
        "Growth": _elasticity("growth_pct"),
        "Volatility": _elasticity("volatility_adj_pct"),
        "Concentration": _elasticity("concentration_reduction_pct"),
        "Demand Shock": _elasticity("demand_shock_pct"),
    }

    ranked = sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True)
    return SensitivityResult(elasticities=elasticities, ranked=ranked)


# ---------------------------------------------------------------------------
# Section 5 — Decision Signal Engine
# ---------------------------------------------------------------------------

def run_decision_signals(forecast: ForecastResult,
                         risk: RiskProjection) -> list[DecisionSignal]:
    signals: list[DecisionSignal] = []

    fc = forecast.forecast_values
    if len(fc) >= 2:
        growth_rate = (fc[-1] - fc[0]) / (abs(fc[0]) + 1e-9)
    else:
        growth_rate = 0.0

    strong_growth = growth_rate > 0.05
    moderate_growth = 0.0 < growth_rate <= 0.05
    declining = growth_rate < 0.0
    low_risk = risk.volatility < 0.15 and risk.concentration_risk < 0.4
    high_concentration = risk.concentration_risk > 0.5
    rising_volatility = risk.volatility > 0.2
    high_anomaly = risk.anomaly_pressure > 0.1

    if strong_growth and low_risk:
        signals.append(DecisionSignal(
            signal="Expansion Opportunity",
            strategy="Scale operations and increase capacity investment.",
            rationale=f"Forecast growth of {growth_rate:.1%} with low risk profile supports expansion.",
            severity="positive",
        ))

    if strong_growth and high_concentration:
        signals.append(DecisionSignal(
            signal="Diversification Required",
            strategy="Reduce concentration risk before scaling further.",
            rationale=f"High concentration index ({risk.concentration_risk:.2f}) creates fragility despite growth.",
            severity="warning",
        ))

    if moderate_growth and not low_risk:
        signals.append(DecisionSignal(
            signal="Cautious Optimism",
            strategy="Pursue measured growth with risk controls in place.",
            rationale="Moderate growth trajectory offset by elevated risk factors.",
            severity="warning",
        ))

    if declining:
        signals.append(DecisionSignal(
            signal="Contraction Alert",
            strategy="Protect core margins and defer discretionary investment.",
            rationale=f"Forecast projects {abs(growth_rate):.1%} decline. Prioritize cash preservation.",
            severity="critical",
        ))

    if rising_volatility:
        signals.append(DecisionSignal(
            signal="Hedge Exposure",
            strategy="Introduce volatility-dampening instruments or operational buffers.",
            rationale=f"Volatility index at {risk.volatility:.2f} exceeds safe threshold.",
            severity="warning",
        ))

    if high_anomaly:
        signals.append(DecisionSignal(
            signal="Operational Risk Flag",
            strategy="Conduct root-cause analysis on anomalous data patterns.",
            rationale=f"Anomaly pressure of {risk.anomaly_pressure:.1%} signals irregular operational dynamics.",
            severity="critical",
        ))

    if not signals:
        signals.append(DecisionSignal(
            signal="Stable State",
            strategy="Maintain current trajectory with routine monitoring.",
            rationale="No significant risk or growth triggers detected.",
            severity="positive",
        ))

    return signals


# ---------------------------------------------------------------------------
# Section 6 — AI Strategic Advisor
# ---------------------------------------------------------------------------

def build_ai_context(forecast: ForecastResult,
                     risk: RiskProjection,
                     scenario: ScenarioResult,
                     sensitivity: SensitivityResult) -> dict[str, Any]:
    fc = forecast.forecast_values
    growth = float((fc[-1] - fc[0]) / (abs(fc[0]) + 1e-9)) if len(fc) >= 2 else 0.0

    return {
        "forecast_growth_pct": round(growth * 100, 2),
        "forecast_confidence_pct": round(forecast.confidence_score, 1),
        "best_model": forecast.model_name,
        "model_rmse": round(forecast.rmse, 4),
        "risk_score": round(risk.volatility * 10, 2),
        "concentration_risk": round(risk.concentration_risk, 3),
        "anomaly_pressure_pct": round(risk.anomaly_pressure * 100, 2),
        "stability_score": round(risk.stability_score, 2),
        "scenario_revenue_delta": round(scenario.revenue_delta, 2),
        "scenario_risk_impact": round(scenario.risk_impact, 4),
        "top_sensitivity_driver": sensitivity.ranked[0][0] if sensitivity.ranked else "N/A",
        "sensitivity_ranking": [
            {"variable": k, "elasticity": round(v, 4)}
            for k, v in sensitivity.ranked
        ],
    }


def run_ai_advisor(ai_engine,
                   context: dict[str, Any]) -> Optional[str]:
    """Call the existing AIEngine with structured context. Returns None if unavailable."""
    if ai_engine is None:
        return None

    prompt = f"""You are a senior strategic advisor reviewing a business intelligence report.

ANALYTICAL CONTEXT:
{context}

Provide a concise executive briefing covering:
1. STRATEGIC RECOMMENDATION — One clear action the business should prioritize.
2. RISK MITIGATION PLAN — Two to three specific risk controls.
3. RESOURCE ALLOCATION GUIDANCE — Where to concentrate investment or cut exposure.

Be specific, decisive, and data-driven. Avoid generic statements. Maximum 250 words."""

    try:
        response = ai_engine.generate(prompt)
        return response
    except Exception:
        try:
            response = ai_engine.chat(prompt)
            return response
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Section 7 — Decision Scorecard
# ---------------------------------------------------------------------------

def run_scorecard(forecast: ForecastResult,
                  risk: RiskProjection,
                  signals: list[DecisionSignal]) -> DecisionScorecard:
    fc = forecast.forecast_values
    growth_rate = float((fc[-1] - fc[0]) / (abs(fc[0]) + 1e-9)) if len(fc) >= 2 else 0.0

    # Growth strength 0–10
    growth_strength = float(np.clip((growth_rate + 0.3) / 0.6 * 10, 0, 10))

    # Risk exposure 0–10 (higher = more risk)
    risk_exposure = float(np.clip(
        (risk.volatility * 4 + risk.concentration_risk * 4 + risk.anomaly_pressure * 2) * 10,
        0, 10
    ))

    stability_score = risk.stability_score

    forecast_confidence = forecast.confidence_score

    # Penalty for critical signals
    critical_count = sum(1 for s in signals if s.severity == "critical")
    warning_count = sum(1 for s in signals if s.severity == "warning")
    penalty = critical_count * 1.5 + warning_count * 0.5

    raw_readiness = (growth_strength * 0.35
                     + (10 - risk_exposure) * 0.25
                     + stability_score * 0.20
                     + (forecast_confidence / 10) * 0.20
                     - penalty)

    strategic_readiness = float(np.clip(raw_readiness, 0, 10))

    # Summary
    if strategic_readiness >= 7:
        summary = "Strong strategic position. Conditions support bold moves."
    elif strategic_readiness >= 4:
        summary = "Moderate readiness. Proceed with selective investments and risk controls."
    else:
        summary = "Elevated risk environment. Defensive posture recommended."

    return DecisionScorecard(
        growth_strength=round(growth_strength, 2),
        risk_exposure=round(risk_exposure, 2),
        stability_score=round(stability_score, 2),
        forecast_confidence=round(forecast_confidence, 1),
        strategic_readiness=round(strategic_readiness, 2),
        summary=summary,
    )