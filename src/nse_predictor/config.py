"""Project configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = f"sqlite:///{ROOT / 'data' / 'processed' / 'nse_predictor.db'}"
    data_raw_dir: Path = ROOT / "data" / "raw" / "nse"
    data_processed_dir: Path = ROOT / "data" / "processed"
    data_external_dir: Path = ROOT / "data" / "external"
    models_dir: Path = ROOT / "models" / "artifacts"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    disclaimer: str = (
        "Forecasts are for internal research and decision support only. "
        "They are not investment advice. Past performance does not guarantee future results. "
        "Any real investment decision should involve a licensed adviser regulated by the Capital Markets Authority."
    )

    # Time splits (calendar years)
    train_end: str = "2019-12-31"
    valid_end: str = "2022-12-31"
    test_end: str = "2025-12-31"

    # Default liquid universe (refined after EDA)
    liquid_tickers: list[str] = [
        "SCOM",
        "EQTY",
        "KCB",
        "COOP",
        "ABSA",
        "SCBK",
        "NCBA",
        "SBIC",
        "EABL",
        "BAT",
        "BAMB",
        "KQ",
        "NMG",
        "JHL",
        "SAF",
    ]
    index_tickers: list[str] = ["NASI", "NSE20", "NSE 20", "NSE-20"]


settings = Settings()
