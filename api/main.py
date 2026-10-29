"""FastAPI service for NSE decision-support forecasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history
from nse_predictor.serving.score import score_latest_forecasts

app = FastAPI(
    title="NSE Predictor API",
    description="Research decision-support forecasts for NSE. Not investment advice.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastOut(BaseModel):
    stock_code: str
    as_of_date: str
    price: float | None = None
    direction_5d: str
    prob_up: float = Field(..., ge=0, le=1)
    ret5d_point: float | None = None
    ret5d_low: float | None = None
    ret5d_high: float | None = None
    vol20d: float | None = None
    disclaimer: str


def _load_forecasts() -> pd.DataFrame:
    path = settings.data_processed_dir / "latest_forecasts.parquet"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def _load_eval() -> dict:
    path = ROOT / "reports" / "evaluation_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_error_history() -> pd.DataFrame:
    path = settings.data_processed_dir / "error_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_collection_status() -> dict:
    path = settings.data_processed_dir / "collection_status.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": settings.disclaimer}


@app.get("/disclaimer")
def disclaimer():
    return {"disclaimer": settings.disclaimer}


@app.get("/stocks")
def list_stocks():
    try:
        df = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    return {"stocks": sorted(df["stock_code"].unique().tolist())}


@app.get("/forecasts", response_model=list[ForecastOut])
def all_forecasts():
    try:
        df = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    rows = []
    for _, r in df.iterrows():
        rows.append(
            ForecastOut(
                stock_code=r["stock_code"],
                as_of_date=str(pd.to_datetime(r["trade_date"]).date()),
                price=float(r["price"]) if pd.notna(r.get("price")) else None,
                direction_5d=str(r["pred_label"]),
                prob_up=float(r["prob_up"]),
                ret5d_point=float(r["ret5d"]) if pd.notna(r.get("ret5d")) else None,
                ret5d_low=float(r["ret5d_lo"]) if pd.notna(r.get("ret5d_lo")) else None,
                ret5d_high=float(r["ret5d_hi"]) if pd.notna(r.get("ret5d_hi")) else None,
                vol20d=float(r["vol20d"]) if pd.notna(r.get("vol20d")) else None,
                disclaimer=settings.disclaimer,
            )
        )
    return rows


@app.get("/forecasts/{stock_code}", response_model=ForecastOut)
def forecast_one(stock_code: str):
    try:
        df = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    hit = df[df["stock_code"].str.upper() == stock_code.upper()]
    if hit.empty:
        raise HTTPException(404, f"No forecast for {stock_code}")
    r = hit.iloc[0]
    return ForecastOut(
        stock_code=r["stock_code"],
        as_of_date=str(pd.to_datetime(r["trade_date"]).date()),
        price=float(r["price"]) if pd.notna(r.get("price")) else None,
        direction_5d=str(r["pred_label"]),
        prob_up=float(r["prob_up"]),
        ret5d_point=float(r["ret5d"]) if pd.notna(r.get("ret5d")) else None,
        ret5d_low=float(r["ret5d_lo"]) if pd.notna(r.get("ret5d_lo")) else None,
        ret5d_high=float(r["ret5d_hi"]) if pd.notna(r.get("ret5d_hi")) else None,
        vol20d=float(r["vol20d"]) if pd.notna(r.get("vol20d")) else None,
        disclaimer=settings.disclaimer,
    )


@app.get("/history/{stock_code}")
def history(stock_code: str, limit: int = 252):
    path = settings.data_processed_dir / "prices.parquet"
    if not path.exists():
        raise HTTPException(503, "Price history not loaded")
    prices = pd.read_parquet(path)
    hit = prices[prices["stock_code"].str.upper() == stock_code.upper()].sort_values("trade_date")
    if hit.empty:
        raise HTTPException(404, f"Unknown stock {stock_code}")
    tail = hit.tail(limit)
    return {
        "stock_code": stock_code.upper(),
        "points": [
            {
                "date": str(pd.to_datetime(r.trade_date).date()),
                "price": float(r.price),
                "volume": float(r.volume) if pd.notna(r.volume) else 0.0,
            }
            for r in tail.itertuples()
        ],
    }


@app.get("/evaluation")
def evaluation():
    return _load_eval()


@app.get("/drivers")
def drivers():
    path = ROOT / "reports" / "shap_direction.csv"
    if not path.exists():
        return {"drivers": [], "note": "Train models to generate SHAP drivers"}
    df = pd.read_csv(path)
    return {"drivers": df.head(15).to_dict(orient="records")}


@app.get("/accuracy/{stock_code}")
def accuracy(stock_code: str, lookback_n: int = 60):
    """Per-stock hit rate on labelled 5-day direction forecasts (research only)."""
    hist = _load_error_history()
    summary = summarize_error_history(hist, stock_code, lookback_n=lookback_n)
    summary["disclaimer"] = settings.disclaimer
    if summary["n"] == 0 and hist.empty:
        raise HTTPException(
            503,
            "Error history not ready — run scripts/train_evaluate.py",
        )
    return summary


@app.get("/collection/status")
def collection_status():
    """Last drop-folder collection run metadata."""
    status = _load_collection_status()
    status["disclaimer"] = settings.disclaimer
    return status


@app.post("/score/latest")
def score_latest():
    """Re-score latest stored trading day with saved models (no retrain)."""
    try:
        summary = score_latest_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Scoring failed: {exc}") from exc
    summary["disclaimer"] = settings.disclaimer
    return summary


@app.post("/admin/refresh")
def admin_refresh(run_collect: bool = True, run_ingest: bool = True):
    """Optional collect + ingest + score. Intended for local/ops use."""
    import subprocess

    steps: dict = {}
    py = sys.executable
    try:
        if run_collect:
            proc = subprocess.run(
                [py, str(ROOT / "scripts" / "collect_daily.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            steps["collect_exit"] = proc.returncode
            try:
                steps["collect"] = json.loads(proc.stdout.strip().split("\n\n")[-1])
            except Exception:  # noqa: BLE001
                steps["collect_stdout"] = proc.stdout[-2000:]
        if run_ingest:
            proc = subprocess.run(
                [py, str(ROOT / "scripts" / "run_ingest.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            steps["ingest_exit"] = proc.returncode
            if proc.returncode != 0:
                raise HTTPException(500, f"Ingest failed: {proc.stderr[-1000:]}")
        summary = score_latest_forecasts()
        steps["score"] = summary
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc
    steps["disclaimer"] = settings.disclaimer
    return steps
