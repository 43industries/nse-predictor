"""FastAPI service for NSE decision-support forecasts."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.companion.desk_brief import build_desk_brief
from nse_predictor.companion.market_board import build_market_board
from nse_predictor.companion.movers import build_movers
from nse_predictor.companion.paper_journal import append_entry, load_journal
from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history
from nse_predictor.ingest.banks import BANK_NAMES, BANK_TICKERS
from nse_predictor.ingest.fundamentals import load_fundamentals
from nse_predictor.monitoring.status import load_monitor_history, load_monitor_status
from nse_predictor.portfolio import (
    build_nav_history,
    build_suggestions,
    confirm_ticket,
    create_ticket,
    create_tickets_batch,
    get_book_mtm,
    load_fills,
    load_tickets,
    reject_ticket,
    reset_book,
    seed_position,
)
from nse_predictor.serving.score import score_latest_forecasts
from nse_predictor.training.meta import load_model_meta

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


class JournalIn(BaseModel):
    stock_code: str
    stance: str = Field(..., pattern="^(long|flat|skip)$")
    note: str = ""
    as_of_date: str | None = None
    prob_up: float | None = None
    pred_label: str | None = None
    price: float | None = None


class PositionSeedIn(BaseModel):
    stock_code: str
    shares: int = Field(..., ge=0)
    avg_cost: float = Field(..., gt=0)


class PortfolioResetIn(BaseModel):
    cash_kes: float | None = Field(None, gt=0)


class TicketIn(BaseModel):
    stock_code: str
    side: str = Field(..., pattern="^(buy|sell)$")
    shares: int = Field(..., gt=0)
    reason: str = ""
    limit_price: float | None = None
    suggestion_action: str | None = None


class TicketConfirmIn(BaseModel):
    price: float | None = Field(None, gt=0)


class TicketBatchIn(BaseModel):
    tickets: list[TicketIn]


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


def _forecast_row(r) -> ForecastOut:
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


def _require_forecasts() -> pd.DataFrame:
    try:
        return _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc


@app.get("/stocks")
def list_stocks():
    df = _require_forecasts()
    return {"stocks": sorted(df["stock_code"].unique().tolist())}


@app.get("/forecasts", response_model=list[ForecastOut])
def all_forecasts():
    df = _require_forecasts()
    return [_forecast_row(r) for _, r in df.iterrows()]


@app.get("/market/movers")
def market_movers(top_n: int = 5):
    """Session gainers / losers / absolute movers (price moves, not model signals)."""
    return build_movers(top_n=max(1, min(int(top_n), 20)))


@app.get("/market/board")
def market_board():
    """Full forecast board with hit rate, book status, and paper ticket hints."""
    try:
        forecasts = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    return build_market_board(
        forecasts,
        evaluation=_load_eval(),
        error_hist=_load_error_history(),
        model_meta=load_model_meta(settings.models_dir),
    )


@app.get("/forecasts/{stock_code}", response_model=ForecastOut)
def forecast_one(stock_code: str):
    df = _require_forecasts()
    hit = df[df["stock_code"].str.upper() == stock_code.upper()]
    if hit.empty:
        raise HTTPException(404, f"No forecast for {stock_code}")
    return _forecast_row(hit.iloc[0])


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


@app.get("/monitor")
def monitor_status():
    """Latest monitor_status.json (empty object if not run yet)."""
    payload = load_monitor_status(settings.data_processed_dir) or {}
    if not isinstance(payload, dict):
        payload = {}
    out = dict(payload)
    out["disclaimer"] = settings.disclaimer
    return out


@app.get("/monitor/history")
def monitor_history(tail: int = 14):
    """Recent monitor_history.parquet rows."""
    n = max(1, min(int(tail), 90))
    df = load_monitor_history(settings.data_processed_dir, tail=n)
    if df.empty:
        return {"rows": [], "disclaimer": settings.disclaimer}
    if "checked_at" in df.columns:
        df = df.sort_values("checked_at", ascending=False)
    # JSON-safe: NaN/NaT → null
    clean = df.astype(object).where(pd.notna(df), None)
    rows = clean.to_dict(orient="records")
    return {"rows": rows, "disclaimer": settings.disclaimer}


LOGOS_DIR = ROOT / "data" / "external" / "logos"
_LOGO_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif")
_CODE_RE = re.compile(r"^[A-Za-z0-9._-]{1,16}$")


def _logo_path(stock_code: str) -> Path | None:
    code = str(stock_code).strip().upper()
    if not _CODE_RE.match(code):
        return None
    for ext in _LOGO_EXTS:
        path = LOGOS_DIR / f"{code}{ext}"
        if path.is_file():
            return path
    return None


@app.get("/logos/{stock_code}")
def company_logo(stock_code: str):
    """Serve a local company logo from data/external/logos/{CODE}.*"""
    path = _logo_path(stock_code)
    if path is None:
        raise HTTPException(404, "Logo not found")
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime or "application/octet-stream")


@app.get("/companion/desk-brief")
def companion_desk_brief(
    top_n: int = 10,
    min_prob_up: float | None = None,
    min_hit_rate: float | None = None,
):
    """Today's companion setups / skips (book-aware, paper ticket hints)."""
    try:
        forecasts = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    brief = build_desk_brief(
        forecasts,
        evaluation=_load_eval(),
        error_hist=_load_error_history(),
        model_meta=load_model_meta(settings.models_dir),
        top_n=max(1, min(int(top_n), 20)),
        min_prob_up=min_prob_up,
        min_hit_rate=min_hit_rate,
    )
    return brief


@app.get("/companion/journal")
def companion_journal():
    df = load_journal()
    if df.empty:
        return {"entries": [], "disclaimer": settings.disclaimer}
    rows = df.sort_values("logged_at", ascending=False).head(100)
    return {
        "entries": rows.to_dict(orient="records"),
        "disclaimer": settings.disclaimer,
    }


@app.post("/companion/journal")
def companion_journal_append(body: JournalIn):
    try:
        df = append_entry(
            stock_code=body.stock_code,
            stance=body.stance,
            as_of_date=body.as_of_date,
            prob_up=body.prob_up,
            pred_label=body.pred_label,
            price=body.price,
            note=body.note or "",
            source="desk_brief",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        "n_entries": int(len(df)),
        "disclaimer": settings.disclaimer,
    }


@app.get("/fundamentals/banks")
def fundamentals_banks():
    """Latest FY (prefer FY2025) bank fundamentals rows for comparison table."""
    fund = load_fundamentals()
    if fund.empty:
        return {"banks": [], "disclaimer": settings.disclaimer}
    hit = fund[fund["stock_code"].astype(str).str.upper().isin(BANK_TICKERS)].copy()
    if hit.empty:
        return {"banks": [], "disclaimer": settings.disclaimer}
    hit["stock_code"] = hit["stock_code"].astype(str).str.upper()
    hit["_pub"] = pd.to_datetime(hit.get("publish_date", hit.get("report_date")), errors="coerce")
    fy = hit[hit["period"].astype(str).str.upper().str.startswith("FY")].copy()
    if fy.empty:
        fy = hit
    fy["_fy_rank"] = fy["period"].astype(str).str.upper().eq("FY2025").astype(int)
    fy = fy.sort_values(["stock_code", "_fy_rank", "_pub"], ascending=[True, False, False])
    latest = fy.groupby("stock_code", as_index=False).head(1)
    rows = []
    for _, r in latest.iterrows():
        code = str(r["stock_code"])
        rows.append(
            {
                "stock_code": code,
                "bank_name": BANK_NAMES.get(code, code),
                "period": str(r.get("period") or ""),
                "publish_date": str(pd.to_datetime(r.get("publish_date")).date())
                if pd.notna(r.get("publish_date"))
                else None,
                "pat": float(r["pat"]) if pd.notna(r.get("pat")) else None,
                "pat_yoy": float(r["pat_yoy"]) if pd.notna(r.get("pat_yoy")) else None,
                "roe": float(r["roe"]) if pd.notna(r.get("roe")) else None,
                "roa": float(r["roa"]) if pd.notna(r.get("roa")) else None,
                "nim": float(r["nim"]) if pd.notna(r.get("nim")) else None,
                "cir": float(r["cir"]) if pd.notna(r.get("cir")) else None,
                "npl": float(r["npl"]) if pd.notna(r.get("npl")) else None,
                "assets": float(r["assets"]) if pd.notna(r.get("assets")) else None,
                "loans": float(r["loans"]) if pd.notna(r.get("loans")) else None,
                "deposits": float(r["deposits"]) if pd.notna(r.get("deposits")) else None,
                "nii": float(r["nii"]) if pd.notna(r.get("nii")) else None,
                "nfi": float(r["nfi"]) if pd.notna(r.get("nfi")) else None,
                "eps": float(r["eps"]) if pd.notna(r.get("eps")) else None,
                "dividend": float(r["dividend"]) if pd.notna(r.get("dividend")) else None,
            }
        )
    rows.sort(key=lambda x: (x.get("pat") is None, -(x.get("pat") or 0)))
    return {"banks": rows, "disclaimer": settings.disclaimer}


@app.get("/portfolio")
def portfolio_book():
    """Paper book with mark-to-market (research only — not a live broker account)."""
    return get_book_mtm()


@app.post("/portfolio/reset")
def portfolio_reset(body: PortfolioResetIn | None = None):
    cash = body.cash_kes if body else None
    reset_book(cash_kes=cash)
    return get_book_mtm()


@app.post("/portfolio/positions")
def portfolio_seed_position(body: PositionSeedIn):
    """Ops seed/replace a position without cash accounting."""
    try:
        seed_position(body.stock_code, body.shares, body.avg_cost)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_book_mtm()


@app.get("/portfolio/suggestions")
def portfolio_suggestions(
    min_prob_up: float | None = None,
    min_hit_rate: float | None = None,
):
    """Score-aware paper suggestions vs current book. Does not place orders."""
    try:
        forecasts = _load_forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Forecasts not ready: {exc}") from exc
    return build_suggestions(
        forecasts,
        evaluation=_load_eval(),
        model_meta=load_model_meta(settings.models_dir),
        error_hist=_load_error_history(),
        min_prob_up=min_prob_up,
        min_hit_rate=min_hit_rate,
    )


@app.get("/portfolio/nav-history")
def portfolio_nav_history(days: int = 90):
    """Daily paper NAV series from fill replay + mark-to-market."""
    n = max(1, min(int(days), 730))
    return build_nav_history(days=n)


@app.get("/portfolio/tickets")
def portfolio_tickets(status: str | None = None):
    tickets = load_tickets()
    if status:
        want = status.strip().lower()
        tickets = [t for t in tickets if str(t.get("status")).lower() == want]
    return {"tickets": tickets, "disclaimer": settings.disclaimer, "mode": "paper"}


@app.post("/portfolio/tickets")
def portfolio_create_ticket(body: TicketIn):
    try:
        ticket = create_ticket(
            stock_code=body.stock_code,
            side=body.side,
            shares=body.shares,
            reason=body.reason or "",
            limit_price=body.limit_price,
            suggestion_action=body.suggestion_action,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ticket": ticket, "disclaimer": settings.disclaimer, "mode": "paper"}


@app.post("/portfolio/tickets/batch")
def portfolio_create_tickets_batch(body: TicketBatchIn):
    """Create many pending tickets; partial success OK."""
    items = [t.model_dump() for t in (body.tickets or [])]
    return create_tickets_batch(items)


@app.post("/portfolio/tickets/{ticket_id}/confirm")
def portfolio_confirm_ticket(ticket_id: str, body: TicketConfirmIn | None = None):
    """Human-confirmed paper fill at last close (or provided price)."""
    try:
        result = confirm_ticket(
            ticket_id,
            price=body.price if body else None,
            source="manual",
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["disclaimer"] = settings.disclaimer
    return result


@app.post("/portfolio/tickets/{ticket_id}/reject")
def portfolio_reject_ticket(ticket_id: str):
    try:
        ticket = reject_ticket(ticket_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ticket": ticket, "disclaimer": settings.disclaimer, "mode": "paper"}


@app.get("/portfolio/fills")
def portfolio_fills(limit: int = 50):
    fills = load_fills()
    n = max(1, min(int(limit), 500))
    return {
        "fills": list(reversed(fills[-n:])),
        "disclaimer": settings.disclaimer,
        "mode": "paper",
    }


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
