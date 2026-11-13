"""Human trading-companion helpers (desk brief, paper journal, market board)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_desk_brief",
    "build_market_board",
    "build_movers",
    "append_entry",
    "load_journal",
    "journal_path",
]


def __getattr__(name: str) -> Any:
    if name == "build_desk_brief":
        from nse_predictor.companion.desk_brief import build_desk_brief

        return build_desk_brief
    if name == "build_market_board":
        from nse_predictor.companion.market_board import build_market_board

        return build_market_board
    if name == "build_movers":
        from nse_predictor.companion.movers import build_movers

        return build_movers
    if name in ("append_entry", "load_journal", "journal_path"):
        from nse_predictor.companion import paper_journal

        return getattr(paper_journal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
