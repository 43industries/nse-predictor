"""Database engine and schema bootstrap (Postgres or SQLite)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

from nse_predictor.config import ROOT, settings

_SCHEMA_SQL = (ROOT / "scripts" / "sql" / "init.sql").read_text(encoding="utf-8")


def get_engine(url: str | None = None) -> Engine:
    db_url = url or settings.database_url
    if db_url.startswith("sqlite"):
        Path(db_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(db_url, future=True)
    return create_engine(db_url, future=True, pool_pre_ping=True)


def _sqlite_compatible(sql: str) -> str:
    """Strip Postgres-only clauses for SQLite."""
    lines = []
    for line in sql.splitlines():
        u = line.upper()
        if "DOUBLE PRECISION" in u:
            line = line.replace("DOUBLE PRECISION", "REAL").replace("double precision", "REAL")
        if "TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in u:
            line = line.replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT CURRENT_TIMESTAMP")
        lines.append(line)
    return "\n".join(lines)


def init_db(engine: Engine | None = None) -> Engine:
    eng = engine or get_engine()
    sql = _SCHEMA_SQL
    if eng.dialect.name == "sqlite":
        sql = _sqlite_compatible(sql)
    with eng.begin() as conn:
        for stmt in sql.split(";"):
            chunk = stmt.strip()
            if chunk:
                conn.execute(text(chunk))
    return eng


def get_connection(engine: Engine | None = None) -> Connection:
    eng = engine or get_engine()
    return eng.connect()
