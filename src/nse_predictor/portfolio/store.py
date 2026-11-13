"""JSON portfolio book, tickets, and fill ledger (paper only)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nse_predictor.config import settings

DISCLAIMER = settings.disclaimer


def _portfolio_dir(processed: Path | None = None) -> Path:
    root = Path(processed or settings.data_processed_dir) / "portfolio"
    root.mkdir(parents=True, exist_ok=True)
    return root


def book_path(processed: Path | None = None) -> Path:
    return _portfolio_dir(processed) / "book.json"


def tickets_path(processed: Path | None = None) -> Path:
    return _portfolio_dir(processed) / "tickets.json"


def fills_path(processed: Path | None = None) -> Path:
    return _portfolio_dir(processed) / "fills.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_book() -> dict[str, Any]:
    return {
        "cash_kes": float(settings.portfolio_starting_cash),
        "currency": "KES",
        "lot_size": int(settings.portfolio_lot_size),
        "cost_bps": float(settings.portfolio_cost_bps),
        "updated_at": _now(),
        "positions": [],
        "disclaimer": DISCLAIMER,
        "mode": "paper",
    }


def load_book(processed: Path | None = None) -> dict[str, Any]:
    path = book_path(processed)
    if not path.exists():
        book = default_book()
        save_book(book, processed)
        return book
    book = _read_json(path, default_book())
    book.setdefault("positions", [])
    book.setdefault("currency", "KES")
    book.setdefault("lot_size", int(settings.portfolio_lot_size))
    book.setdefault("cost_bps", float(settings.portfolio_cost_bps))
    book.setdefault("mode", "paper")
    book["disclaimer"] = DISCLAIMER
    return book


def save_book(book: dict[str, Any], processed: Path | None = None) -> dict[str, Any]:
    book = dict(book)
    book["updated_at"] = _now()
    book["disclaimer"] = DISCLAIMER
    book["mode"] = "paper"
    save = {k: v for k, v in book.items() if k != "mtm"}
    _write_json(book_path(processed), save)
    return book


def reset_book(processed: Path | None = None, cash_kes: float | None = None) -> dict[str, Any]:
    book = default_book()
    if cash_kes is not None:
        book["cash_kes"] = float(cash_kes)
    save_book(book, processed)
    # Clear ledger so NAV history matches a fresh paper book
    save_tickets([], processed)
    save_fills([], processed)
    return load_book(processed)


def seed_position(
    stock_code: str,
    shares: int,
    avg_cost: float,
    *,
    processed: Path | None = None,
    opened_at: str | None = None,
) -> dict[str, Any]:
    """Set or replace a position without cash accounting (ops seed)."""
    code = str(stock_code).strip().upper()
    shares_i = int(shares)
    if shares_i < 0:
        raise ValueError("shares must be >= 0")
    book = load_book(processed)
    positions = [p for p in book["positions"] if str(p.get("stock_code")).upper() != code]
    if shares_i > 0:
        positions.append(
            {
                "stock_code": code,
                "shares": shares_i,
                "avg_cost": float(avg_cost),
                "opened_at": opened_at or _now(),
            }
        )
    book["positions"] = sorted(positions, key=lambda p: p["stock_code"])
    return save_book(book, processed)


def load_tickets(processed: Path | None = None) -> list[dict[str, Any]]:
    return list(_read_json(tickets_path(processed), []))


def save_tickets(tickets: list[dict[str, Any]], processed: Path | None = None) -> None:
    _write_json(tickets_path(processed), tickets)


def load_fills(processed: Path | None = None) -> list[dict[str, Any]]:
    return list(_read_json(fills_path(processed), []))


def save_fills(fills: list[dict[str, Any]], processed: Path | None = None) -> None:
    _write_json(fills_path(processed), fills)


def _latest_prices(processed: Path | None = None) -> dict[str, float]:
    path = Path(processed or settings.data_processed_dir) / "prices.parquet"
    if not path.exists():
        return {}
    prices = pd.read_parquet(path, columns=["trade_date", "stock_code", "price"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.upper()
    prices = prices.dropna(subset=["price"]).sort_values("trade_date")
    last = prices.groupby("stock_code", as_index=False).tail(1)
    return {str(r.stock_code): float(r.price) for r in last.itertuples()}


def risk_snapshot(book_mtm: dict[str, Any]) -> dict[str, Any]:
    """Policy vs utilization for the paper desk."""
    max_names = int(settings.portfolio_max_names)
    max_w = float(settings.portfolio_max_weight_pct)
    min_cash = float(settings.portfolio_min_cash_pct)
    lot = int(book_mtm.get("lot_size") or settings.portfolio_lot_size)
    cost_bps = float(book_mtm.get("cost_bps") or settings.portfolio_cost_bps)
    nav = float(book_mtm.get("nav_kes") or 0.0)
    cash = float(book_mtm.get("cash_kes") or 0.0)
    n_names = int(book_mtm.get("n_names") or 0)
    cash_pct = (cash / nav) if nav > 0 else 1.0
    weights = [
        float(p["weight"])
        for p in (book_mtm.get("positions") or [])
        if p.get("weight") is not None
    ]
    max_pos_w = max(weights) if weights else 0.0
    breaches: list[str] = []
    if n_names > max_names:
        breaches.append(f"n_names {n_names} > max_names {max_names}")
    if max_pos_w > max_w + 1e-9:
        breaches.append(f"max_position_weight {max_pos_w:.1%} > {max_w:.0%}")
    if cash_pct + 1e-9 < min_cash:
        breaches.append(f"cash_pct {cash_pct:.1%} < min_cash_pct {min_cash:.0%}")
    return {
        "policy": {
            "max_names": max_names,
            "max_weight_pct": max_w,
            "min_cash_pct": min_cash,
            "lot_size": lot,
            "cost_bps": cost_bps,
        },
        "utilization": {
            "n_names": n_names,
            "cash_pct": cash_pct,
            "max_position_weight": max_pos_w,
            "nav_kes": nav,
            "cash_kes": cash,
        },
        "breaches": breaches,
    }


def get_book_mtm(processed: Path | None = None) -> dict[str, Any]:
    book = load_book(processed)
    px = _latest_prices(processed)
    positions_out: list[dict[str, Any]] = []
    equity_positions = 0.0
    cost_basis = 0.0
    for p in book.get("positions") or []:
        code = str(p["stock_code"]).upper()
        shares = int(p["shares"])
        avg = float(p["avg_cost"])
        last = px.get(code)
        mtm = (last * shares) if last is not None else None
        pnl = (mtm - avg * shares) if mtm is not None else None
        row = {
            "stock_code": code,
            "shares": shares,
            "avg_cost": avg,
            "last_price": last,
            "market_value": mtm,
            "unrealized_pnl": pnl,
            "opened_at": p.get("opened_at"),
        }
        if mtm is not None:
            equity_positions += mtm
            cost_basis += avg * shares
        positions_out.append(row)

    cash = float(book.get("cash_kes") or 0.0)
    nav = cash + equity_positions
    for row in positions_out:
        if row["market_value"] is not None and nav > 0:
            row["weight"] = float(row["market_value"]) / nav
        else:
            row["weight"] = None

    out = {
        "cash_kes": cash,
        "currency": book.get("currency", "KES"),
        "lot_size": int(book.get("lot_size") or settings.portfolio_lot_size),
        "cost_bps": float(book.get("cost_bps") or settings.portfolio_cost_bps),
        "mode": "paper",
        "updated_at": book.get("updated_at"),
        "positions": positions_out,
        "equity_kes": equity_positions,
        "nav_kes": nav,
        "cost_basis_kes": cost_basis,
        "unrealized_pnl_kes": equity_positions - cost_basis,
        "n_names": len(positions_out),
        "disclaimer": DISCLAIMER,
    }
    out["risk"] = risk_snapshot(out)
    return out


def validate_ticket_against_book(
    *,
    stock_code: str,
    side: str,
    shares: int,
    processed: Path | None = None,
    price: float | None = None,
) -> None:
    """Raise ValueError if ticket would breach lots, cash, holdings, or soft risk caps."""
    side_n = str(side).strip().lower()
    if side_n not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    shares_i = int(shares)
    if shares_i <= 0:
        raise ValueError("shares must be > 0")
    lot = int(settings.portfolio_lot_size)
    code = str(stock_code).strip().upper()
    book = get_book_mtm(processed)
    pos_map = {str(p["stock_code"]).upper(): p for p in book.get("positions") or []}
    held = int(pos_map.get(code, {}).get("shares") or 0)

    # Full exit of odd lots allowed when selling entire holding
    if side_n == "sell" and held > 0 and shares_i == held:
        pass
    elif shares_i % lot != 0:
        raise ValueError(f"shares must be a multiple of lot_size={lot}")

    if side_n == "sell":
        if shares_i > held:
            raise ValueError(f"insufficient shares to sell: have {held}, need {shares_i}")
        return

    px_map = _latest_prices(processed)
    fill_price = float(price) if price is not None else px_map.get(code)
    if fill_price is None or fill_price <= 0:
        raise ValueError(f"no price for {code}; cannot validate buy ticket")

    cost_bps = float(book.get("cost_bps") or settings.portfolio_cost_bps)
    notional = fill_price * shares_i
    fees = notional * (cost_bps / 10000.0)
    total = notional + fees
    cash = float(book.get("cash_kes") or 0.0)
    if total > cash + 1e-6:
        raise ValueError(f"insufficient cash: need {total:.2f}, have {cash:.2f}")

    nav = float(book.get("nav_kes") or 0.0)
    min_cash = float(settings.portfolio_min_cash_pct)
    cash_after = cash - total
    if nav > 0 and (cash_after / nav) + 1e-9 < min_cash:
        raise ValueError(
            f"cash buffer breach: after buy cash_pct would be "
            f"{cash_after / nav:.1%} < min_cash_pct {min_cash:.0%}"
        )

    max_names = int(settings.portfolio_max_names)
    if held == 0 and int(book.get("n_names") or 0) >= max_names:
        raise ValueError(f"max_names={max_names} already reached")

    max_w = float(settings.portfolio_max_weight_pct)
    current_mv = float(pos_map.get(code, {}).get("market_value") or 0.0)
    new_mv = current_mv + notional
    if nav > 0 and (new_mv / nav) > max_w + 1e-6:
        raise ValueError(
            f"max_weight breach: {code} would be {new_mv / nav:.1%} > {max_w:.0%}"
        )


def create_ticket(
    *,
    stock_code: str,
    side: str,
    shares: int,
    reason: str = "",
    limit_price: float | None = None,
    suggestion_action: str | None = None,
    processed: Path | None = None,
) -> dict[str, Any]:
    side_n = str(side).strip().lower()
    shares_i = int(shares)
    validate_ticket_against_book(
        stock_code=stock_code,
        side=side_n,
        shares=shares_i,
        processed=processed,
        price=limit_price,
    )

    ticket = {
        "id": f"t_{uuid.uuid4().hex[:12]}",
        "status": "pending",
        "stock_code": str(stock_code).strip().upper(),
        "side": side_n,
        "shares": shares_i,
        "limit_price": float(limit_price) if limit_price is not None else None,
        "reason": str(reason or ""),
        "suggestion_action": suggestion_action,
        "created_at": _now(),
        "resolved_at": None,
    }
    tickets = load_tickets(processed)
    tickets.append(ticket)
    save_tickets(tickets, processed)
    return ticket


def create_tickets_batch(
    items: list[dict[str, Any]],
    *,
    processed: Path | None = None,
) -> dict[str, Any]:
    """Create many pending tickets; partial success OK."""
    created: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, raw in enumerate(items or []):
        try:
            ticket = create_ticket(
                stock_code=str(raw.get("stock_code") or ""),
                side=str(raw.get("side") or ""),
                shares=int(raw.get("shares") or 0),
                reason=str(raw.get("reason") or ""),
                limit_price=raw.get("limit_price"),
                suggestion_action=raw.get("suggestion_action"),
                processed=processed,
            )
            created.append(ticket)
        except (ValueError, TypeError) as exc:
            errors.append(
                {
                    "index": i,
                    "stock_code": raw.get("stock_code"),
                    "side": raw.get("side"),
                    "shares": raw.get("shares"),
                    "error": str(exc),
                }
            )
    return {
        "created": created,
        "errors": errors,
        "n_created": len(created),
        "n_errors": len(errors),
        "disclaimer": DISCLAIMER,
        "mode": "paper",
    }


def reject_ticket(ticket_id: str, processed: Path | None = None) -> dict[str, Any]:
    tickets = load_tickets(processed)
    hit = None
    for t in tickets:
        if t.get("id") == ticket_id:
            hit = t
            break
    if hit is None:
        raise KeyError(f"ticket not found: {ticket_id}")
    if hit.get("status") != "pending":
        raise ValueError(f"ticket is {hit.get('status')}, not pending")
    hit["status"] = "rejected"
    hit["resolved_at"] = _now()
    save_tickets(tickets, processed)
    return hit


def confirm_ticket(
    ticket_id: str,
    *,
    price: float | None = None,
    processed: Path | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """Apply a pending ticket as a paper fill at given or last close price."""
    tickets = load_tickets(processed)
    hit = None
    for t in tickets:
        if t.get("id") == ticket_id:
            hit = t
            break
    if hit is None:
        raise KeyError(f"ticket not found: {ticket_id}")
    if hit.get("status") != "pending":
        raise ValueError(f"ticket is {hit.get('status')}, not pending")

    code = str(hit["stock_code"]).upper()
    side = str(hit["side"]).lower()
    shares = int(hit["shares"])
    px_map = _latest_prices(processed)
    fill_price = float(price) if price is not None else px_map.get(code)
    if fill_price is None or fill_price <= 0:
        raise ValueError(f"no fill price for {code}; pass price= explicitly")

    book = load_book(processed)
    cost_bps = float(book.get("cost_bps") or settings.portfolio_cost_bps)
    notional = fill_price * shares
    fees = notional * (cost_bps / 10000.0)
    cash = float(book.get("cash_kes") or 0.0)
    positions = {str(p["stock_code"]).upper(): dict(p) for p in book.get("positions") or []}

    if side == "buy":
        total = notional + fees
        if total > cash + 1e-6:
            raise ValueError(f"insufficient cash: need {total:.2f}, have {cash:.2f}")
        cash -= total
        if code in positions:
            prev = positions[code]
            prev_shares = int(prev["shares"])
            prev_cost = float(prev["avg_cost"]) * prev_shares
            new_shares = prev_shares + shares
            prev["avg_cost"] = (prev_cost + notional) / new_shares
            prev["shares"] = new_shares
        else:
            positions[code] = {
                "stock_code": code,
                "shares": shares,
                "avg_cost": fill_price,
                "opened_at": _now(),
            }
    else:
        if code not in positions or int(positions[code]["shares"]) < shares:
            have = int(positions.get(code, {}).get("shares") or 0)
            raise ValueError(f"insufficient shares to sell: have {have}, need {shares}")
        cash += notional - fees
        left = int(positions[code]["shares"]) - shares
        if left == 0:
            del positions[code]
        else:
            positions[code]["shares"] = left

    book["cash_kes"] = cash
    book["positions"] = sorted(positions.values(), key=lambda p: p["stock_code"])
    save_book(book, processed)

    fill = {
        "ticket_id": ticket_id,
        "stock_code": code,
        "side": side,
        "shares": shares,
        "price": fill_price,
        "fees_bps": cost_bps,
        "fees_kes": fees,
        "notional_kes": notional,
        "ts": _now(),
        "source": source or "manual",
    }
    fills = load_fills(processed)
    fills.append(fill)
    save_fills(fills, processed)

    hit["status"] = "filled"
    hit["resolved_at"] = fill["ts"]
    hit["fill_price"] = fill_price
    save_tickets(tickets, processed)

    return {"ticket": hit, "fill": fill, "book": get_book_mtm(processed)}
