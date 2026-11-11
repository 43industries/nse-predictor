"""NSE banking ticker map for peer features and dashboard context."""

from __future__ import annotations

# Ordinary share codes present in Wanjawa / processed prices.
BANK_TICKERS: frozenset[str] = frozenset(
    {
        "EQTY",
        "KCB",
        "COOP",
        "ABSA",
        "NCBA",
        "HFCK",
        "SCBK",
        "DTK",
        "IMH",
        "SBIC",
    }
)

# kenyanbanks.com / listed-bank display names (docs only).
BANK_NAMES: dict[str, str] = {
    "EQTY": "Equity Group",
    "KCB": "KCB Group",
    "COOP": "Co-operative Bank",
    "ABSA": "Absa Bank Kenya",
    "NCBA": "NCBA Group",
    "HFCK": "HF Group",
    "SCBK": "Standard Chartered Kenya",
    "DTK": "Diamond Trust Bank",
    "IMH": "I&M Group",
    "SBIC": "Stanbic Holdings",
}


def is_bank(code: str) -> bool:
    return str(code).strip().upper() in BANK_TICKERS
