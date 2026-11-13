"""Paper portfolio brain and human-confirmed execution (research only)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BrokerAdapter",
    "PaperBrokerStub",
    "build_nav_history",
    "build_suggestions",
    "confirm_ticket",
    "create_ticket",
    "create_tickets_batch",
    "get_book_mtm",
    "load_book",
    "load_fills",
    "load_tickets",
    "reject_ticket",
    "reset_book",
    "save_book",
    "seed_position",
    "validate_ticket_against_book",
]


def __getattr__(name: str) -> Any:
    if name in ("BrokerAdapter", "PaperBrokerStub"):
        from nse_predictor.portfolio.broker import BrokerAdapter, PaperBrokerStub

        return BrokerAdapter if name == "BrokerAdapter" else PaperBrokerStub
    if name == "build_suggestions":
        from nse_predictor.portfolio.brain import build_suggestions

        return build_suggestions
    if name == "build_nav_history":
        from nse_predictor.portfolio.nav_history import build_nav_history

        return build_nav_history
    if name in {
        "confirm_ticket",
        "create_ticket",
        "create_tickets_batch",
        "get_book_mtm",
        "load_book",
        "load_fills",
        "load_tickets",
        "reject_ticket",
        "reset_book",
        "save_book",
        "seed_position",
        "validate_ticket_against_book",
    }:
        from nse_predictor.portfolio import store

        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
