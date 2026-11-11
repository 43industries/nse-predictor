"""Classical time-series helpers (ARIMA / GARCH) per stock."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def fit_arima_forecast(series: pd.Series, steps: int = 5) -> dict:
    """Fit a simple ARIMA(1,1,1) and forecast `steps` ahead.

    Returns point forecast of terminal return vs last price, or NaNs on failure.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    y = series.dropna().astype(float)
    if len(y) < 50:
        return {"ok": False, "error": "insufficient history"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=(1, 1, 1))
            res = model.fit()
            fc = res.forecast(steps=steps)
        last = float(y.iloc[-1])
        terminal = float(fc.iloc[-1])
        return {
            "ok": True,
            "pred_price": terminal,
            "pred_ret": terminal / last - 1,
            "path": [float(x) for x in fc],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def fit_prophet_forecast(series: pd.Series, steps: int = 5) -> dict:
    """Optional Prophet baseline (skipped if prophet / cmdstanpy unavailable)."""
    try:
        from prophet import Prophet
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"prophet_unavailable: {exc}", "method": "prophet"}

    y = series.dropna().astype(float)
    if len(y) < 60:
        return {"ok": False, "error": "insufficient history", "method": "prophet"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = pd.DataFrame({"ds": y.index if isinstance(y.index, pd.DatetimeIndex) else pd.RangeIndex(len(y)), "y": y.values})
            if not isinstance(y.index, pd.DatetimeIndex):
                # Build synthetic business dates ending today
                hist["ds"] = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(y))
            else:
                hist["ds"] = pd.to_datetime(y.index)
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(hist)
            future = m.make_future_dataframe(periods=steps, freq="B")
            fc = m.predict(future).tail(steps)
        last = float(y.iloc[-1])
        terminal = float(fc["yhat"].iloc[-1])
        return {
            "ok": True,
            "pred_price": terminal,
            "pred_ret": terminal / last - 1 if last else float("nan"),
            "method": "prophet",
            "path": [float(x) for x in fc["yhat"]],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "method": "prophet"}


def fit_garch_vol(returns: pd.Series, horizon: int = 20) -> dict:
    """GARCH(1,1) forecast of average daily vol over horizon."""
    try:
        from arch import arch_model
    except Exception as exc:  # noqa: BLE001
        # Fallback: rolling std
        vol = float(returns.dropna().tail(horizon).std())
        return {"ok": True, "pred_vol": vol, "method": "rolling_std_fallback", "error": str(exc)}

    r = returns.dropna().astype(float) * 100  # percent for numerical stability
    if len(r) < 100:
        return {"ok": False, "error": "insufficient history"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(r, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off")
            f = res.forecast(horizon=horizon)
        var = f.variance.values[-1]
        daily = np.sqrt(np.mean(var)) / 100.0
        return {"ok": True, "pred_vol": float(daily), "method": "garch11"}
    except Exception as exc:  # noqa: BLE001
        vol = float(returns.dropna().tail(horizon).std())
        return {"ok": True, "pred_vol": vol, "method": "rolling_std_fallback", "error": str(exc)}
