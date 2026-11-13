"""Broker adapter protocol — paper stub only until a licensed broker is configured."""

from __future__ import annotations

from typing import Any, Protocol


class BrokerAdapter(Protocol):
    def submit_order(
        self,
        *,
        stock_code: str,
        side: str,
        shares: int,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        ...


class PaperBrokerStub:
    """Does not route live orders. Use portfolio confirm_ticket for paper fills."""

    name = "paper_stub"

    def submit_order(
        self,
        *,
        stock_code: str,
        side: str,
        shares: int,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Live broker routing is not configured. "
            "Use POST /portfolio/tickets/{id}/confirm for paper fills, "
            "or implement a licensed BrokerAdapter for your dealing member."
        )
