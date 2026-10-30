"""Streamlit dashboard for NSE decision-support forecasts.

Visual language inspired by https://www.nse.co.ke/ (lime green, navy ticker, clean white).
Not affiliated with the Nairobi Securities Exchange.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history

LOGOS_DIR = ROOT / "data" / "external" / "logos"
LOGOS_CSV = LOGOS_DIR / "companies.csv"

st.set_page_config(
    page_title="NSE Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = settings.disclaimer
PREFS_PATH = settings.data_processed_dir / "dashboard_prefs.json"

# Palette inspired by nse.co.ke (not official brand assets)
NSE_GREEN = "#8DC442"
NSE_GREEN_DARK = "#61A229"
NSE_NAVY = "#0B1F3A"
NSE_NAVY_MID = "#16325A"
UP = "#2E7D32"
DOWN = "#C62828"
INK = "#222222"
MUTED = "#676767"
BG = "#F5F6F7"
LINE = NSE_NAVY
VOLUME = "#9AA3B2"
WHITE = "#FFFFFF"

FEATURE_LABELS: dict[str, str] = {
    "ret_1d": "1-day price change",
    "ma5_ratio": "Price vs 5-day average",
    "ma20_ratio": "Price vs 20-day average",
    "ma50_ratio": "Price vs 50-day average",
    "vol_20": "20-day volatility",
    "rsi_14": "RSI momentum (14)",
    "macd": "MACD trend",
    "macd_signal": "MACD signal line",
    "dist_52w_high": "Distance from 52-week high",
    "dist_52w_low": "Distance from 52-week low",
    "volume_chg": "Volume change",
    "days_since_trade": "Days since last trade",
    "nasi_ret": "NSE index return",
    "sector_ret": "Sector return",
    "cbk_rate": "CBK policy rate",
    "cbk_chg": "CBK rate change",
    "tbill_91d": "91-day T-bill",
    "usd_kes": "USD/KES level",
    "usd_kes_chg": "USD/KES change",
    "inflation_yoy": "Inflation (YoY)",
    "inflation_trend": "Inflation trend",
    "lending_rate": "Lending rate",
    "dow": "Day of week",
    "month": "Calendar month",
    "results_season": "Results season flag",
}

CHART_WINDOWS = {"1M": 22, "3M": 66, "1Y": 252, "All": None}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Montserrat:wght@600;700;800&display=swap');

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
  color: {INK};
}}

.stApp {{
  background: {BG};
}}

[data-testid="stSidebar"] {{
  background: {WHITE};
  border-right: 1px solid #DEDFE0;
}}

h1, h2, h3, .brand-wordmark {{
  font-family: 'Montserrat', 'DM Sans', sans-serif !important;
  color: {NSE_NAVY} !important;
  letter-spacing: -0.02em;
}}

.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  background: {WHITE};
  border: 1px solid #DEDFE0;
  border-radius: 4px;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.85rem;
}}

.brand-lockup {{
  display: flex;
  align-items: baseline;
  gap: 0.45rem;
}}

.brand-wordmark {{
  font-size: 1.55rem;
  font-weight: 800;
  margin: 0;
  line-height: 1;
}}

.brand-wordmark span {{
  color: {NSE_GREEN_DARK};
}}

.brand-tag {{
  font-size: 0.78rem;
  color: {MUTED};
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}

.market-status {{
  font-size: 0.92rem;
  color: {INK};
  text-align: right;
}}

.market-status strong.open {{
  color: {NSE_GREEN_DARK};
}}

.market-status .date {{
  display: block;
  color: {MUTED};
  font-size: 0.8rem;
  margin-top: 0.15rem;
}}

.ticker-bar {{
  overflow: hidden;
  background: {NSE_NAVY};
  color: {WHITE};
  border-radius: 4px;
  margin: 0 0 0.65rem 0;
  padding: 0.55rem 0;
  white-space: nowrap;
}}

.ticker-track {{
  display: inline-flex;
  width: max-content;
  animation: ticker-scroll 45s linear infinite;
}}

.ticker-bar:hover .ticker-track {{
  animation-play-state: paused;
}}

@keyframes ticker-scroll {{
  from {{ transform: translateX(0); }}
  to {{ transform: translateX(-50%); }}
}}

.ticker-item {{
  flex: 0 0 auto;
  padding: 0.15rem 1.1rem;
  border-right: 1px solid rgba(255,255,255,0.12);
  font-size: 0.82rem;
  white-space: nowrap;
}}

.ticker-code {{
  font-weight: 700;
  margin-right: 0.4rem;
}}

.ticker-px {{
  color: #DEDFE0;
  margin-right: 0.45rem;
}}

.ticker-up {{ color: {NSE_GREEN}; font-weight: 600; }}
.ticker-down {{ color: #FF6B6B; font-weight: 600; }}
.ticker-flat {{ color: #B0B7C3; font-weight: 600; }}

.section-label {{
  font-family: 'Montserrat', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {MUTED};
  margin: 0.25rem 0 0.55rem 0;
}}

.jump-nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0 0 0.85rem 0;
}}

.jump-chip {{
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border: 1px solid #DEDFE0;
  background: {WHITE};
  color: {NSE_NAVY};
  font-size: 0.78rem;
  font-weight: 600;
  border-radius: 2px;
}}

.stance-hero {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem 1.1rem;
  background: {WHITE};
  border: 1px solid #DEDFE0;
  border-left: 5px solid {NSE_NAVY_MID};
  border-radius: 4px;
  padding: 0.95rem 1.1rem;
  margin: 0.35rem 0 0.85rem 0;
}}

.stance-hero.lean-up {{ border-left-color: {UP}; }}
.stance-hero.lean-down {{ border-left-color: {DOWN}; }}
.stance-hero.lean-flat {{ border-left-color: {MUTED}; }}

.stance-pill {{
  display: inline-block;
  padding: 0.4rem 0.9rem;
  border-radius: 2px;
  font-weight: 800;
  font-size: 0.95rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}

.stance-watch {{ background: {NSE_GREEN}; color: {NSE_NAVY}; }}
.stance-avoid {{ background: #FDECEA; color: {DOWN}; }}
.stance-neutral {{ background: #ECEFF3; color: {MUTED}; }}

.stance-prob {{
  font-family: 'Montserrat', sans-serif;
  font-size: 1.35rem;
  font-weight: 800;
  color: {NSE_NAVY};
  line-height: 1;
}}

.stance-sub {{
  color: {MUTED};
  font-size: 0.85rem;
}}

.track-badge {{
  display: inline-block;
  margin-left: auto;
  padding: 0.35rem 0.7rem;
  background: #F0F4F8;
  border: 1px solid #DEDFE0;
  border-radius: 2px;
  font-size: 0.78rem;
  color: {INK};
}}

.track-badge strong {{
  color: {NSE_NAVY};
}}

.signal-card {{
  background: {WHITE};
  border: 1px solid #DEDFE0;
  border-radius: 4px;
  padding: 0.75rem 0.85rem 0.55rem 0.85rem;
  margin-bottom: 0.35rem;
  min-height: 148px;
}}

.signal-card.active {{
  border-color: {NSE_GREEN_DARK};
  box-shadow: inset 3px 0 0 {NSE_GREEN_DARK};
}}

.signal-card .card-head {{
  display: flex;
  align-items: center;
  gap: 0.65rem;
}}

.signal-card .code {{
  font-family: 'Montserrat', sans-serif;
  font-weight: 800;
  font-size: 1.05rem;
  color: {NSE_NAVY};
}}

.logo-wrap {{
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  flex-shrink: 0;
}}

.logo-wrap.lg {{
  width: 52px;
  height: 52px;
}}

.stock-logo {{
  width: 36px;
  height: 36px;
  border-radius: 6px;
  object-fit: contain;
  background: #F0F4F8;
  border: 1px solid #DEDFE0;
  flex-shrink: 0;
}}

.stock-logo.lg {{
  width: 52px;
  height: 52px;
  border-radius: 8px;
}}

.logo-mono {{
  width: 36px;
  height: 36px;
  border-radius: 6px;
  background: {NSE_NAVY};
  color: {WHITE};
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: 'Montserrat', sans-serif;
  font-weight: 800;
  font-size: 0.72rem;
  flex-shrink: 0;
  letter-spacing: 0.02em;
}}

.logo-mono.lg {{
  width: 52px;
  height: 52px;
  font-size: 0.95rem;
}}

.detail-title {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.15rem 0 0.55rem 0;
}}

.detail-title h3 {{
  margin: 0;
}}

.detail-title .co-name {{
  display: block;
  font-size: 0.82rem;
  color: {MUTED};
  font-weight: 500;
  margin-top: 0.15rem;
}}

.signal-card .meta {{
  color: {MUTED};
  font-size: 0.78rem;
  margin-top: 0.15rem;
}}

.signal-card .chg-up {{ color: {UP}; font-weight: 700; font-size: 0.82rem; }}
.signal-card .chg-down {{ color: {DOWN}; font-weight: 700; font-size: 0.82rem; }}
.signal-card .chg-flat {{ color: {MUTED}; font-weight: 700; font-size: 0.82rem; }}

.prob-label {{
  font-size: 0.72rem;
  color: {MUTED};
  margin: 0.15rem 0 0 0;
}}

.mover-box {{
  background: {WHITE};
  border: 1px solid #DEDFE0;
  border-radius: 4px;
  padding: 0.65rem 0.75rem 0.45rem 0.75rem;
  min-height: 210px;
}}

.mover-box .box-title {{
  font-family: 'Montserrat', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {MUTED};
  margin: 0 0 0.55rem 0;
}}

.mover-row {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 0;
  border-bottom: 1px solid #EEF0F2;
  font-size: 0.88rem;
}}

.mover-row:last-child {{
  border-bottom: none;
}}

.mover-row .sym {{
  font-family: 'Montserrat', sans-serif;
  font-weight: 700;
  color: {NSE_NAVY};
  min-width: 3.2rem;
}}

.mover-row .px {{
  color: {MUTED};
  font-size: 0.78rem;
  margin-left: auto;
  margin-right: 0.45rem;
}}

.mover-row .pct {{
  font-weight: 700;
  min-width: 3.6rem;
  text-align: right;
}}

.mover-asof {{
  color: {MUTED};
  font-size: 0.75rem;
  margin: 0.15rem 0 0.65rem 0;
}}

.one-liner {{
  color: {MUTED};
  font-size: 0.9rem;
  margin: 0.35rem 0 0.85rem 0;
}}

div[data-testid="stMetric"] {{
  background: {WHITE};
  border: 1px solid #DEDFE0;
  border-radius: 4px;
  padding: 0.65rem 0.85rem;
}}

div[data-testid="stMetric"] label {{
  color: {MUTED} !important;
}}

.block-container {{
  padding-top: 1.1rem;
  max-width: 1200px;
}}

section[data-testid="stSidebar"] .block-container {{
  padding-top: 1.25rem;
}}

.stProgress > div > div > div > div {{
  background-color: {NSE_GREEN_DARK};
}}
</style>
"""


def _default_prefs() -> dict:
    return {
        "favourites": [],
        "last_stock": None,
        "recent": [],
        "chart_window": "1Y",
        "min_prob_up": 0.0,
        "min_hit_rate": 0.0,
        "favourites_only": False,
        "main_tab": "My market",
    }


def load_prefs() -> dict:
    base = _default_prefs()
    if PREFS_PATH.exists():
        try:
            saved = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                base.update({k: saved[k] for k in base if k in saved})
        except Exception:  # noqa: BLE001
            pass
    return base


def save_prefs(prefs: dict) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def friendly_feature(name: str) -> str:
    return FEATURE_LABELS.get(str(name), str(name).replace("_", " ").title())


@lru_cache(maxsize=1)
def load_company_meta() -> dict[str, dict[str, str]]:
    if not LOGOS_CSV.exists():
        return {}
    df = pd.read_csv(LOGOS_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        code = str(r.get("stock_code", "")).strip().upper()
        if not code:
            continue
        out[code] = {
            "company": str(r.get("company") or "").strip(),
            "domain": str(r.get("domain") or "").strip().lower(),
        }
    return out


def _local_logo_path(code: str) -> Path | None:
    code = code.upper()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"):
        path = LOGOS_DIR / f"{code}{ext}"
        if path.exists():
            return path
    return None


def _file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def logo_image_src(code: str) -> str | None:
    """Local cached file wins; else gstatic favicon URL for mapped domain."""
    code = str(code).upper()
    local = _local_logo_path(code)
    if local is not None:
        try:
            return _file_to_data_uri(local)
        except Exception:  # noqa: BLE001
            pass
    meta = load_company_meta().get(code) or {}
    domain = (meta.get("domain") or "").strip()
    if domain:
        domain = domain.removeprefix("https://").removeprefix("http://").strip("/")
        # More reliable than s2/favicons for many .co.ke sites
        return (
            "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON"
            f"&fallback_opts=TYPE,SIZE,URL&url=https://{domain}&size=128"
        )
    return None


def company_display_name(code: str) -> str:
    meta = load_company_meta().get(str(code).upper()) or {}
    return meta.get("company") or str(code).upper()


def logo_html(code: str, large: bool = False) -> str:
    """Always render a monogram; show image on top and fall back if it breaks."""
    src = logo_image_src(code)
    size_cls = " lg" if large else ""
    letters = str(code).upper()[:2]
    mono = f'<div class="logo-mono{size_cls}" title="{code}">{letters}</div>'
    if not src:
        return mono
    wrap_cls = "logo-wrap lg" if large else "logo-wrap"
    # Hide broken remote/local images and keep the monogram visible
    return (
        f'<span class="{wrap_cls}" title="{code}">'
        f'<img class="stock-logo{size_cls}" src="{src}" alt="" '
        f'loading="lazy" referrerpolicy="no-referrer" '
        f"onerror=\"this.style.display='none';"
        f"var m=this.nextElementSibling; if(m) m.style.display='inline-flex';\" />"
        f'<div class="logo-mono{size_cls}" style="display:none" title="{code}">{letters}</div>'
        f"</span>"
    )


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    path = settings.data_processed_dir / "latest_forecasts.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_prices() -> pd.DataFrame:
    path = settings.data_processed_dir / "prices.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_eval() -> dict:
    path = ROOT / "reports" / "evaluation_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_shap() -> pd.DataFrame:
    path = ROOT / "reports" / "shap_direction.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_fundamentals() -> pd.DataFrame:
    path = settings.data_external_dir / "fundamentals_sample.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_error_history() -> pd.DataFrame:
    path = settings.data_processed_dir / "error_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_collection_status() -> dict:
    path = settings.data_processed_dir / "collection_status.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_monitor_status() -> dict:
    from nse_predictor.monitoring.status import load_monitor_status as _load

    return _load(settings.data_processed_dir)


@st.cache_data
def load_monitor_history(tail: int = 14):
    from nse_predictor.monitoring.status import load_monitor_history as _load

    try:
        return _load(settings.data_processed_dir, tail=tail)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data
def load_news() -> pd.DataFrame:
    path = ROOT / "data" / "external" / "news.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "stock_code" not in df.columns:
        return pd.DataFrame()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)
    return df


def render_stock_news(news: pd.DataFrame, stock_code: str, limit: int = 8) -> None:
    st.markdown('<p class="section-label">Related news</p>', unsafe_allow_html=True)
    if news.empty:
        st.caption(
            "No news file yet. Drop CSVs into `data/external/news/incoming/` "
            "then run `python scripts/collect_news.py`."
        )
        return
    hit = news[news["stock_code"] == stock_code.upper()].head(limit)
    if hit.empty:
        st.caption(f"No curated news for {stock_code}. Add rows to data/external/news.csv.")
        return
    for _, r in hit.iterrows():
        when = pd.to_datetime(r.get("date"), errors="coerce")
        when_s = when.strftime("%Y-%m-%d") if pd.notna(when) else "—"
        source = str(r.get("source") or "").strip() or "Source n/a"
        headline = str(r.get("headline") or "").strip()
        url = str(r.get("url") or "").strip()
        sent = str(r.get("sentiment") or "").strip().lower()
        badge = ""
        if sent in {"positive", "negative", "neutral"}:
            badge = f" · {sent}"
        if url.startswith("http"):
            st.markdown(f"- **{when_s}** — [{headline}]({url}) _{source}{badge}_")
        else:
            st.markdown(f"- **{when_s}** — {headline} _{source}{badge}_")
    st.caption("Curated drop-folder news for context only — not a trading signal.")


def stance_from_prob(prob_up: float, pred_label: str | None = None) -> tuple[str, str, str]:
    if pred_label is not None and str(pred_label).lower() == "flat":
        return "Flat / abstain", "stance-neutral", "lean-flat"
    if prob_up >= 0.55:
        return "Lean up", "stance-watch", "lean-up"
    if prob_up <= 0.45:
        return "Lean down", "stance-avoid", "lean-down"
    return "Mixed", "stance-neutral", "lean-flat"


def hit_rate_map(error_hist: pd.DataFrame, stocks: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for code in stocks:
        acc = summarize_error_history(error_hist, code, lookback_n=60) if not error_hist.empty else {}
        out[code] = float(acc["hit_rate"]) if acc.get("n") else None
    return out


def select_stock(code: str, prefs: dict, go_detail: bool = True) -> None:
    code = str(code).upper()
    st.session_state["stock"] = code
    recent = [c for c in prefs.get("recent", []) if c != code]
    recent = [code] + recent
    prefs["recent"] = recent[:8]
    prefs["last_stock"] = code
    if go_detail:
        prefs["main_tab"] = "Stock detail"
        st.session_state["main_tab"] = "Stock detail"
    save_prefs(prefs)
    st.session_state["prefs"] = prefs


def render_ticker_bar(forecasts: pd.DataFrame) -> None:
    ordered = forecasts.copy()
    ordered["_s"] = (ordered["prob_up"] - 0.5).abs()
    ordered = ordered.sort_values("_s", ascending=False)
    items: list[str] = []
    for _, r in ordered.iterrows():
        code = r["stock_code"]
        px = f"{r['price']:.2f}" if pd.notna(r.get("price")) else "—"
        label = str(r["pred_label"]).lower()
        prob = float(r["prob_up"])
        if label == "up":
            cls, arrow, move = "ticker-up", "▲", f"+{prob:.0%}"
        elif label == "down":
            cls, arrow, move = "ticker-down", "▼", f"{prob:.0%}"
        else:
            cls, arrow, move = "ticker-flat", "•", f"{prob:.0%}"
        items.append(
            f'<div class="ticker-item">'
            f'<span class="ticker-code">{code}</span>'
            f'<span class="ticker-px">{px}</span>'
            f'<span class="{cls}">{arrow} {move}</span>'
            f"</div>"
        )
    track = "".join(items) + "".join(items)
    html = f'<div class="ticker-bar"><div class="ticker-track">{track}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def sparkline_change_pct(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 2:
        return None
    a = float(hist["price"].iloc[0])
    b = float(hist["price"].iloc[-1])
    if a == 0 or pd.isna(a) or pd.isna(b):
        return None
    return (b / a - 1.0) * 100.0


def build_sparkline(hist: pd.DataFrame, color: str = NSE_NAVY) -> go.Figure:
    x = hist["trade_date"]
    y = hist["price"].astype(float)
    y_min = float(y.min()) if len(y) else 0.0
    baseline = y_min * 0.995 if y_min else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[baseline] * len(y),
            mode="lines",
            line=dict(width=0, color=_rgba(color, 0)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tonexty",
            fillcolor=_rgba(color, 0.18),
            hovertemplate="%{x|%d %b %Y}<br>KES %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    if len(y):
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[-1]],
                y=[y.iloc[-1]],
                mode="markers",
                marker=dict(size=8, color=color, line=dict(width=1.5, color=WHITE)),
                hovertemplate="%{x|%d %b %Y}<br>KES %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=4, t=4, b=0),
        height=78,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
        hovermode="x",
    )
    return fig


def build_prob_bar(prob: float) -> go.Figure:
    color = UP if prob >= 0.55 else (DOWN if prob <= 0.45 else NSE_NAVY_MID)
    fig = go.Figure(
        go.Bar(
            x=[prob],
            y=[""],
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            width=0.55,
            hovertemplate="P(up) %{x:.0%}<extra></extra>",
            text=[f"{prob:.0%}"],
            textposition="inside",
            textfont=dict(color=WHITE, size=11, family="DM Sans"),
            constraintext="none",
        )
    )
    fig.add_vline(x=0.5, line=dict(color="#B0B7C3", width=1, dash="dot"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=2, b=2),
        height=34,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
        bargap=0.05,
    )
    return fig


def build_market_overview(
    prices: pd.DataFrame,
    window_key: str,
    focus_codes: list[str] | None = None,
) -> tuple[go.Figure | None, str]:
    """Prefer ^NASI; else equal-weight index of focus (or liquid) names."""
    n = CHART_WINDOWS.get(window_key)
    nasi = prices[prices["stock_code"].astype(str) == "^NASI"].sort_values("trade_date")
    if not nasi.empty:
        series = nasi.tail(n) if n else nasi
        title = f"NSE All Share (^NASI) · {window_key}"
        y = series["price"].astype(float)
        x = series["trade_date"]
        label = "Index"
    else:
        codes = [c for c in (focus_codes or []) if not str(c).startswith("^")]
        if not codes:
            return None, ""
        panel = prices[prices["stock_code"].isin(codes)].copy()
        if panel.empty:
            return None, ""
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        wide = panel.pivot_table(index="trade_date", columns="stock_code", values="price")
        rebased = wide.apply(lambda col: col / col.dropna().iloc[0] * 100 if col.dropna().shape[0] else col)
        eq = rebased.mean(axis=1).dropna()
        if n:
            eq = eq.tail(n)
        if eq.empty:
            return None, ""
        x = eq.index
        y = eq
        title = f"Equal-weight basket ({len(codes)} names) · {window_key}"
        label = "Basket (rebased 100)"

    color = NSE_NAVY
    fig = go.Figure()
    y0 = float(y.iloc[0]) if len(y) else 0.0
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[y0] * len(y),
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color, width=2.4),
            fill="tonexty",
            fillcolor=_rgba(NSE_GREEN_DARK, 0.16),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        )
    )
    if len(y):
        x_last = x.iloc[-1] if hasattr(x, "iloc") else list(x)[-1]
        fig.add_trace(
            go.Scatter(
                x=[x_last],
                y=[float(y.iloc[-1])],
                mode="markers",
                marker=dict(size=9, color=NSE_GREEN_DARK, line=dict(width=2, color=WHITE)),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    chg = None
    if len(y) >= 2 and float(y.iloc[0]) != 0:
        chg = (float(y.iloc[-1]) / float(y.iloc[0]) - 1.0) * 100.0
    chg_txt = f"{chg:+.1f}% over window" if chg is not None else ""
    fig.update_layout(
        title=dict(
            text=f"{title}" + (f"<br><sup style='color:{MUTED}'>{chg_txt}</sup>" if chg_txt else ""),
            font=dict(family="Montserrat, DM Sans, sans-serif", size=14, color=NSE_NAVY),
            x=0,
            xanchor="left",
        ),
        margin=dict(l=8, r=12, t=56, b=8),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(showgrid=False, zeroline=False, type="date"),
        yaxis=dict(showgrid=True, gridcolor="#E8EAED", zeroline=False, title=""),
    )
    return fig, title


def latest_equity_moves(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """One row per stock: latest session move for gainers / losers / movers boards."""
    if prices.empty:
        return pd.DataFrame()
    df = prices.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df = df[~df["stock_code"].str.contains(r"-R$|^\^", regex=True)]
    if universe:
        uni = {str(c).upper() for c in universe}
        df = df[df["stock_code"].isin(uni)]
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "price"])
    latest = df.sort_values("trade_date").groupby("stock_code", as_index=False).tail(1)
    if "ret_1d" not in latest.columns or latest["ret_1d"].isna().all():
        if "change_pct" in latest.columns:
            latest["ret_1d"] = pd.to_numeric(latest["change_pct"], errors="coerce") / 100.0
        else:
            latest["ret_1d"] = pd.NA
    latest["ret_1d"] = pd.to_numeric(latest["ret_1d"], errors="coerce")
    latest["volume"] = pd.to_numeric(latest.get("volume", 0), errors="coerce").fillna(0)
    latest["abs_ret"] = latest["ret_1d"].abs()
    return latest.dropna(subset=["ret_1d"]).reset_index(drop=True)


def render_movers_text_boxes(
    prices: pd.DataFrame,
    universe: list[str],
    prefs: dict,
    top_n: int = 5,
) -> None:
    moves = latest_equity_moves(prices, universe=universe)
    st.markdown('<p class="section-label">Session movers</p>', unsafe_allow_html=True)
    if moves.empty:
        st.caption("No 1-day moves available yet for this desk.")
        return
    as_of = pd.to_datetime(moves["trade_date"]).max()
    as_of_s = as_of.strftime("%d/%m/%Y") if pd.notna(as_of) else "—"
    st.markdown(
        f'<p class="mover-asof">Based on each name’s latest session (desk as of {as_of_s}). '
        "Price moves — not model signals.</p>",
        unsafe_allow_html=True,
    )
    gainers = moves.sort_values("ret_1d", ascending=False).head(top_n)
    losers = moves.sort_values("ret_1d", ascending=True).head(top_n)
    movers = moves.sort_values(["abs_ret", "volume"], ascending=[False, False]).head(top_n)

    boards = [
        ("Gainers", gainers, "gainer"),
        ("Losers", losers, "loser"),
        ("Movers", movers, "mover"),
    ]
    cols = st.columns(3)
    for col, (title, board, key_prefix) in zip(cols, boards):
        with col:
            rows_html = []
            for _, r in board.iterrows():
                code = str(r["stock_code"])
                ret = float(r["ret_1d"])
                px = float(r["price"]) if pd.notna(r.get("price")) else None
                px_s = f"{px:.2f}" if px is not None else "—"
                pct_cls = "chg-up" if ret > 0 else ("chg-down" if ret < 0 else "chg-flat")
                vol = float(r.get("volume") or 0)
                extra = ""
                if title == "Movers" and vol > 0:
                    if vol >= 1_000_000:
                        extra = f" · {vol/1_000_000:.1f}M"
                    elif vol >= 1_000:
                        extra = f" · {vol/1_000:.0f}K"
                    else:
                        extra = f" · {vol:.0f}"
                rows_html.append(
                    f'<div class="mover-row">{logo_html(code)}'
                    f'<span class="sym">{code}</span>'
                    f'<span class="px">{px_s}{extra}</span>'
                    f'<span class="pct {pct_cls}">{ret:+.1%}</span></div>'
                )
            body = "".join(rows_html) if rows_html else '<div class="mover-row">No rows</div>'
            st.markdown(
                f'<div class="mover-box"><p class="box-title">{title}</p>{body}</div>',
                unsafe_allow_html=True,
            )
            options = [str(c) for c in board["stock_code"].tolist()]
            if options:
                pick = st.selectbox(
                    f"Open from {title.lower()}",
                    options,
                    key=f"{key_prefix}_sel",
                    label_visibility="collapsed",
                )
                if st.button(
                    f"Open {pick}",
                    key=f"{key_prefix}_go",
                    use_container_width=True,
                ):
                    select_stock(pick, prefs, go_detail=True)
                    st.rerun()


def render_market_overview(
    prices: pd.DataFrame,
    prefs: dict,
    focus_codes: list[str],
) -> None:
    st.markdown('<p class="section-label">Market overview</p>', unsafe_allow_html=True)
    window = prefs.get("chart_window") or "1Y"
    fig, _ = build_market_overview(prices, window, focus_codes=focus_codes)
    if fig is None:
        st.caption("No index or basket series available for overview.")
        return
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Context only — uses ^NASI when available, otherwise an equal-weight basket of the "
        "names on this desk. Not a live exchange quote."
    )


def build_price_figure(
    hist: pd.DataFrame,
    ret5d_lo: float | None = None,
    ret5d_hi: float | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Scatter(
            x=hist["trade_date"],
            y=hist["price"],
            mode="lines",
            name="Price",
            line=dict(color=LINE, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>KES %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if (
        len(hist)
        and pd.notna(ret5d_lo)
        and pd.notna(ret5d_hi)
        and pd.notna(hist["price"].iloc[-1])
    ):
        last_dt = pd.to_datetime(hist["trade_date"].iloc[-1])
        last_px = float(hist["price"].iloc[-1])
        lo = last_px * (1 + float(ret5d_lo))
        hi = last_px * (1 + float(ret5d_hi))
        end_dt = last_dt + pd.Timedelta(days=7)
        fig.add_trace(
            go.Scatter(
                x=[last_dt, end_dt, end_dt, last_dt],
                y=[hi, hi, lo, lo],
                fill="toself",
                fillcolor="rgba(97, 162, 41, 0.18)",
                line=dict(width=0),
                name="5d range",
                hovertemplate="5d model range<br>%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=last_px * (1 + (float(ret5d_lo) + float(ret5d_hi)) / 2),
            line=dict(color=NSE_GREEN_DARK, width=1, dash="dot"),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=hist["trade_date"],
            y=hist["volume"],
            name="Volume",
            marker_color=VOLUME,
            opacity=0.9,
            hovertemplate="%{x|%Y-%m-%d}<br>Vol %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        showlegend=False,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="KES", row=1, col=1, gridcolor="#E8EAED", zeroline=False)
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="#E8EAED", zeroline=False)
    return fig


def build_shap_figure(shap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = shap_df.head(top_n).copy()
    total = float(data["mean_abs_shap"].sum()) or 1.0
    data["share"] = data["mean_abs_shap"] / total * 100
    data["label"] = data["feature"].map(friendly_feature)
    data = data.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["share"],
            y=data["label"],
            orientation="h",
            marker_color=NSE_GREEN_DARK,
            hovertemplate="%{y}<br>%{x:.0f}% of top drivers<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        xaxis=dict(title="Relative importance (%)", gridcolor="#E8EAED", zeroline=False),
        yaxis=dict(title="", automargin=True),
    )
    return fig


def build_baseline_figure(evaluation: dict) -> go.Figure:
    bas = evaluation.get("baselines", {})
    mod = evaluation.get("models", {})
    labels = ["Always up", "MA 5/20", "LightGBM"]
    values = [
        float(bas.get("always_up", {}).get("accuracy") or 0) * 100,
        float(bas.get("ma_5_20", {}).get("accuracy") or 0) * 100,
        float(mod.get("lightgbm_direction", {}).get("accuracy") or 0) * 100,
    ]
    colors = [MUTED, NSE_NAVY_MID, NSE_GREEN_DARK]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>Accuracy %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        yaxis=dict(
            title="Direction accuracy (%)",
            gridcolor="#E8EAED",
            range=[0, max(values) * 1.25 or 100],
        ),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def filter_forecasts(
    forecasts: pd.DataFrame,
    prefs: dict,
    hits: dict[str, float | None],
) -> pd.DataFrame:
    df = forecasts.copy()
    favs = set(prefs.get("favourites") or [])
    if prefs.get("favourites_only") and favs:
        df = df[df["stock_code"].isin(favs)]
    min_p = float(prefs.get("min_prob_up") or 0)
    if min_p > 0:
        df = df[df["prob_up"] >= min_p]
    min_h = float(prefs.get("min_hit_rate") or 0)
    if min_h > 0:
        keep = []
        for code in df["stock_code"]:
            hr = hits.get(code)
            keep.append(hr is not None and hr >= min_h)
        df = df.loc[keep]
    return df


def render_signal_cards(
    forecasts: pd.DataFrame,
    prices: pd.DataFrame,
    hits: dict[str, float | None],
    prefs: dict,
    active: str,
) -> None:
    st.markdown('<p class="section-label">Signal cards</p>', unsafe_allow_html=True)
    if forecasts.empty:
        st.info("No stocks match your filters. Loosen P(up) / hit-rate or turn off Favourites only.")
        return
    ordered = forecasts.copy()
    ordered["_s"] = (ordered["prob_up"] - 0.5).abs()
    ordered = ordered.sort_values("_s", ascending=False)
    window = prefs.get("chart_window") or "1Y"
    # Sparklines stay short for card density; map window lightly
    spark_n = {"1M": 22, "3M": 44, "1Y": 66, "All": 66}.get(window, 44)
    st.caption(
        f"Showing {len(ordered)} name(s) · sparkline ~{spark_n} sessions · sorted by signal strength."
    )
    cols = st.columns(3)
    for i, (_, r) in enumerate(ordered.iterrows()):
        code = str(r["stock_code"])
        prob = float(r["prob_up"])
        pred_lab = str(r["pred_label"]) if "pred_label" in r.index and pd.notna(r.get("pred_label")) else None
        stance, _pill, _lean = stance_from_prob(prob, pred_lab)
        hr = hits.get(code)
        hr_s = f"{hr:.0%} hit" if hr is not None else "no track yet"
        active_cls = " active" if code == active else ""
        hist = (
            prices[prices["stock_code"] == code]
            .sort_values("trade_date")
            .tail(spark_n)
        )
        chg = sparkline_change_pct(hist)
        if chg is None:
            chg_html = '<span class="chg-flat">—</span>'
        elif chg > 0.05:
            chg_html = f'<span class="chg-up">{chg:+.1f}%</span>'
        elif chg < -0.05:
            chg_html = f'<span class="chg-down">{chg:+.1f}%</span>'
        else:
            chg_html = f'<span class="chg-flat">{chg:+.1f}%</span>'
        with cols[i % 3]:
            st.markdown(
                f'<div class="signal-card{active_cls}">'
                f'<div class="card-head">{logo_html(code)}'
                f'<div><div class="code">{code} &nbsp; {chg_html}</div>'
                f'<div class="meta">{company_display_name(code)}</div>'
                f'<div class="meta">{stance} · {hr_s}</div></div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            color = UP if prob >= 0.55 else (DOWN if prob <= 0.45 else NSE_NAVY_MID)
            if not hist.empty:
                st.plotly_chart(
                    build_sparkline(hist, color=color),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"spark_{code}",
                )
            st.markdown('<p class="prob-label">P(excess up &gt; +1%)</p>', unsafe_allow_html=True)
            st.plotly_chart(
                build_prob_bar(prob),
                width="stretch",
                config={"displayModeBar": False},
                key=f"prob_{code}",
            )
            b1, b2 = st.columns(2)
            if b1.button("Open", key=f"open_{code}", use_container_width=True):
                select_stock(code, prefs, go_detail=True)
                st.rerun()
            is_fav = code in (prefs.get("favourites") or [])
            fav_label = "★ Fav" if is_fav else "☆ Fav"
            if b2.button(fav_label, key=f"fav_{code}", use_container_width=True):
                favs = list(prefs.get("favourites") or [])
                if is_fav:
                    favs = [c for c in favs if c != code]
                else:
                    favs.append(code)
                prefs["favourites"] = favs
                save_prefs(prefs)
                st.session_state["prefs"] = prefs
                st.rerun()


def render_jump_nav() -> str | None:
    st.markdown(
        '<div class="jump-nav">'
        '<span class="jump-chip">Jump</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    labels = ["Overview", "Chart", "Drivers", "News"]
    cols = st.columns(len(labels))
    chosen = None
    for i, lab in enumerate(labels):
        if cols[i].button(lab, key=f"jump_{lab}", use_container_width=True):
            chosen = lab
            st.session_state["detail_section"] = lab
    return st.session_state.get("detail_section") or chosen


def render_stock_detail(
    stock: str,
    forecasts: pd.DataFrame,
    prices: pd.DataFrame,
    shap_df: pd.DataFrame,
    fundamentals: pd.DataFrame,
    error_hist: pd.DataFrame,
    news: pd.DataFrame,
    prefs: dict,
) -> None:
    row = forecasts[forecasts["stock_code"] == stock]
    if row.empty:
        st.warning(f"{stock} is not in the current forecast set.")
        return
    row = row.iloc[0]
    prob = float(row["prob_up"])
    pred_lab = str(row["pred_label"]) if "pred_label" in row.index and pd.notna(row.get("pred_label")) else None
    stance, stance_cls, lean_cls = stance_from_prob(prob, pred_lab)
    acc = summarize_error_history(error_hist, stock, lookback_n=60) if not error_hist.empty else {}

    section = render_jump_nav()

    show_overview = section in (None, "Overview")
    show_chart = section in (None, "Chart")
    show_drivers = section in (None, "Drivers")
    show_news = section in (None, "News")
    # When a jump is chosen, focus that block; Overview shows hero+metrics
    if section == "Overview":
        show_chart = show_drivers = show_news = False
    elif section == "Chart":
        show_overview = show_drivers = show_news = False
        show_chart = True
    elif section == "Drivers":
        show_overview = show_chart = show_news = False
        show_drivers = True
    elif section == "News":
        show_overview = show_chart = show_drivers = False
        show_news = True
    else:
        show_overview = show_chart = show_drivers = show_news = True

    if show_overview:
        track_html = "No labelled track record yet"
        if acc.get("n"):
            track_html = (
                f"<strong>{acc['hit_rate']:.0%}</strong> correct "
                f"· last {acc['n']} · streak {acc['recent_streak']}"
            )
        st.markdown(
            f'<div class="stance-hero {lean_cls}">'
            f'<span class="stance-pill {stance_cls}">{stance}</span>'
            f'<div><div class="stance-prob">{prob:.0%}</div>'
            f'<div class="stance-sub">P(5d excess vs NASI &gt; +1%)</div></div>'
            f'<div class="track-badge">Track record<br/>{track_html}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0), text=f"Probability of rising: {prob:.0%}")
        st.markdown(
            '<p class="one-liner">Human makes the final call. Not investment advice. '
            "Not affiliated with the Nairobi Securities Exchange.</p>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("As of", str(pd.to_datetime(row["trade_date"]).date()))
        c2.metric("Last price (KES)", f"{row['price']:.2f}" if pd.notna(row["price"]) else "—")
        c3.metric("5d direction", str(row["pred_label"]).upper())
        if pd.notna(row.get("ret5d_lo")) and pd.notna(row.get("ret5d_hi")):
            band = f"{row['ret5d_lo']:.1%} to {row['ret5d_hi']:.1%}"
        else:
            band = "—"
        c4.metric("5d excess range", band)

        favs = list(prefs.get("favourites") or [])
        is_fav = stock in favs
        if st.button(
            "Remove from favourites" if is_fav else "Add to favourites",
            key="detail_fav_toggle",
        ):
            if is_fav:
                favs = [c for c in favs if c != stock]
            else:
                favs.append(stock)
            prefs["favourites"] = favs
            save_prefs(prefs)
            st.session_state["prefs"] = prefs
            st.rerun()

    if show_chart:
        st.markdown('<p class="section-label">Price & volume</p>', unsafe_allow_html=True)
        window = prefs.get("chart_window") or "1Y"
        n = CHART_WINDOWS.get(window)
        hist = prices[prices["stock_code"] == stock].sort_values("trade_date")
        if n:
            hist = hist.tail(n)
        if not hist.empty:
            lo = float(row["ret5d_lo"]) if pd.notna(row.get("ret5d_lo")) else None
            hi = float(row["ret5d_hi"]) if pd.notna(row.get("ret5d_hi")) else None
            st.plotly_chart(build_price_figure(hist, lo, hi), width="stretch")
            st.caption(
                f"Window: {window}. Green band = model 5-day return range from last close "
                "(research estimate, not a guarantee)."
            )
        else:
            st.caption("No price history for this ticker.")

    if show_drivers:
        left, right = st.columns(2)
        with left:
            st.markdown('<p class="section-label">Key drivers</p>', unsafe_allow_html=True)
            if shap_df.empty:
                st.caption("Drivers not available yet.")
            else:
                st.plotly_chart(build_shap_figure(shap_df), width="stretch")
                top = shap_df.head(5)
                bullets = []
                total = float(shap_df.head(10)["mean_abs_shap"].sum()) or 1.0
                for _, r in top.iterrows():
                    share = float(r["mean_abs_shap"]) / total * 100
                    bullets.append(f"- {friendly_feature(r['feature'])} (~{share:.0f}% of top 10)")
                st.markdown("\n".join(bullets))
                st.caption(
                    "Across the test period, these inputs most often influenced the up/down model — "
                    "not a stock-specific explanation for today's forecast."
                )
        with right:
            st.markdown('<p class="section-label">Fundamentals</p>', unsafe_allow_html=True)
            fund = (
                fundamentals[fundamentals["stock_code"] == stock]
                if not fundamentals.empty
                else pd.DataFrame()
            )
            if fund.empty:
                st.caption("No fundamentals for this ticker yet.")
            else:
                st.dataframe(fund, width="stretch", hide_index=True)

    if show_news:
        render_stock_news(news, stock)

    if section is not None:
        if st.button("Show full stock page", key="show_full_detail"):
            st.session_state["detail_section"] = None
            st.rerun()


def _fmt_num(x, fmt: str = ".3f", fallback: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    return fallback if v != v else format(v, fmt)


def render_monitor_block(monitor: dict, history: pd.DataFrame) -> None:
    st.markdown('<p class="section-label">Monitor</p>', unsafe_allow_html=True)
    if not monitor:
        st.caption("No monitor_status.json yet. Run `python scripts/run_monitor.py` or the daily job.")
        return
    score = monitor.get("score") or {}
    outcomes = monitor.get("outcomes") or {}
    gates_m = monitor.get("gates") or {}
    cols = st.columns(5)
    cols[0].metric("Status", str(monitor.get("status") or "—").upper())
    cols[1].metric("Forecast as_of", str(score.get("as_of") or "—"))
    cols[2].metric("Abstain %", _fmt_num(score.get("abstain_rate"), ".0%"))
    hit = outcomes.get("hit_rate_nonflat") if not outcomes.get("skipped") else None
    cols[3].metric("Lagged hit-rate", _fmt_num(hit, ".0%"))
    cols[4].metric("Val gates", str(gates_m.get("validation_status") or "—").upper())
    checked = monitor.get("checked_at")
    if checked:
        st.caption(f"Last check: {str(checked)[:19]}Z")
    with st.expander("Monitor detail", expanded=False):
        alerts = monitor.get("alerts") or []
        if alerts:
            st.dataframe(pd.DataFrame(alerts), width="stretch", hide_index=True)
        else:
            st.caption("No alerts on last run.")
        if history is not None and not history.empty:
            st.caption("Recent monitor history")
            st.dataframe(history, width="stretch", hide_index=True)


def render_model_health(evaluation: dict, monitor: dict | None = None, mon_hist: pd.DataFrame | None = None) -> None:
    st.markdown('<p class="section-label">Model honesty vs baselines</p>', unsafe_allow_html=True)
    render_monitor_block(monitor or {}, mon_hist if mon_hist is not None else pd.DataFrame())
    if evaluation:
        prob = evaluation.get("problem") or {}
        if prob:
            st.caption(
                "Direction = 5d **excess vs NASI** with ±1% dead zone "
                f"(dead_zone={float(prob.get('dead_zone_xs') or 0.01):.0%}). "
                "Return bands are 5d excess vs NASI (legacy ret5d* aliases match)."
            )

        gates = evaluation.get("validation_gates") or {}
        diag = evaluation.get("diagnostics") or {}
        ric = diag.get("rank_ic") or {}
        wf_sum = evaluation.get("walk_forward_summary") or {}
        cal = diag.get("calibration") or {}
        fragile = bool(evaluation.get("stress_fragile"))
        ar = diag.get("abstain_rate")
        if ar is None:
            ar = (evaluation.get("models") or {}).get("lightgbm_direction", {}).get("abstain_rate")

        st.markdown('<p class="section-label">Validation scorecard</p>', unsafe_allow_html=True)
        cols = st.columns(6)
        cols[0].metric("Status", str(gates.get("status") or "—").upper())
        cols[1].metric("Test Rank IC", _fmt_num(ric.get("mean_ic")))
        cols[2].metric("WF mean Rank IC", _fmt_num(wf_sum.get("wf_mean_rank_ic")))
        cols[3].metric("Abstain %", _fmt_num(ar, ".0%"))
        cols[4].metric("ECE", _fmt_num(cal.get("ece")))
        cols[5].metric("Stress fragile", "yes" if fragile else "no")
        if gates.get("reasons"):
            st.caption(" · ".join(str(r) for r in gates["reasons"]))

        proto = evaluation.get("protocol") or {}
        if proto.get("embargo"):
            st.caption(
                f"Protocol: embargo {proto.get('horizon_days')}d · "
                f"train_end {proto.get('train_end')} · valid_end {proto.get('valid_end')}"
            )

        if ric.get("n_days"):
            st.caption(
                f"Test Rank IC (prob vs excess return): mean **{ric.get('mean_ic', float('nan')):.3f}** "
                f"over {ric.get('n_days')} days."
            )
        sel = evaluation.get("selection") or {}
        if sel:
            st.markdown('<p class="section-label">Validation selection</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Direction threshold", f"{float(sel.get('threshold', 0.5)):.2f}")
            c2.metric("Calibration", str(sel.get("calibration") or "none"))
            rq = sel.get("return_quantiles") or {}
            if rq:
                c3.metric(
                    "Return band (residual q)",
                    f"{float(rq.get('q_low', 0)):.1%} … {float(rq.get('q_high', 0)):.1%}",
                )
            vm = sel.get("valid_direction") or {}
            if vm:
                st.caption(
                    f"Valid accuracy {vm.get('accuracy', 0):.1%} · "
                    f"lift vs always-up {vm.get('lift_vs_always_up', 0):+.1%} · "
                    f"scale_pos_weight {float(sel.get('scale_pos_weight') or 1):.2f}"
                )
        st.plotly_chart(build_baseline_figure(evaluation), width="stretch")
        wf = evaluation.get("walk_forward", [])
        if wf:
            beats = sum(1 for f in wf if f.get("beats_baseline"))
            st.caption(
                f"Walk-forward folds beating always-up baseline: **{beats}/{len(wf)}**. "
                "Prefer probabilities and ranges over point returns."
            )
        cov = evaluation.get("models", {}).get("return_interval_coverage", {})
        if cov:
            st.caption(
                f"Return interval coverage on test: {cov.get('coverage', 0):.0%} "
                f"(mean width {cov.get('mean_width', 0):.1%})."
            )
        cm = diag.get("confusion") or {}
        if cm:
            st.caption(
                f"Test confusion — TP {cm.get('tp')} · FP {cm.get('fp')} · "
                f"TN {cm.get('tn')} · FN {cm.get('fn')}"
            )
        per = diag.get("per_stock") or []
        if per:
            st.markdown('<p class="section-label">Per-stock test accuracy</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(per), width="stretch", hide_index=True, height=260)

        with st.expander("Validation detail", expanded=False):
            stress = evaluation.get("stress_rank_ic") or {}
            if stress:
                st.caption(f"stress_fragile={'true' if fragile else 'false'} (OOS retrain)")
                rows = []
                for period, v in stress.items():
                    if not isinstance(v, dict):
                        continue
                    rows.append(
                        {
                            "period": period,
                            "mean_ic": v.get("mean_ic"),
                            "n_days": v.get("n_days"),
                            "abstain_rate": v.get("abstain_rate"),
                            "skipped": v.get("skipped", False),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            bins = diag.get("calibration_bins") or []
            if bins:
                bdf = pd.DataFrame(bins)
                if {"mean_prob", "frac_up"}.issubset(bdf.columns):
                    import plotly.express as px

                    fig = px.line(
                        bdf.dropna(subset=["mean_prob"]),
                        x="mean_prob",
                        y="frac_up",
                        markers=True,
                        title="Calibration reliability (test, non-flat)",
                    )
                    fig.add_scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash")
                    )
                    st.plotly_chart(fig, width="stretch")
            if wf:
                wf_rows = []
                for f in wf:
                    ric_f = f.get("rank_ic") or {}
                    m = f.get("model") or {}
                    wf_rows.append(
                        {
                            "fold": f.get("fold"),
                            "accuracy": m.get("accuracy"),
                            "rank_ic": ric_f.get("mean_ic"),
                            "abstain_rate": f.get("abstain_rate"),
                            "beats_baseline": f.get("beats_baseline"),
                        }
                    )
                st.dataframe(pd.DataFrame(wf_rows), width="stretch", hide_index=True)
            bt = evaluation.get("research_backtest") or {}
            if bt.get("ok"):
                st.caption(
                    f"Research backtest (not for selection): total {bt.get('total_return', float('nan')):.1%} · "
                    f"max DD {bt.get('max_drawdown', float('nan')):.1%} · names {bt.get('n_names')}"
                )

        report_path = ROOT / "reports" / "train_val_report.md"
        if report_path.exists():
            with st.expander("Train / validation report", expanded=False):
                st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.caption("Run training to populate evaluation_results.json")
    with st.expander("Disclaimer", expanded=False):
        st.write(DISCLAIMER)
        st.caption(
            "Visual design inspired by nse.co.ke. This research tool is not issued by, "
            "endorsed by, or affiliated with the Nairobi Securities Exchange PLC."
        )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    if "prefs" not in st.session_state:
        st.session_state["prefs"] = load_prefs()
    prefs = st.session_state["prefs"]

    forecasts = load_forecasts()
    prices = load_prices()
    evaluation = load_eval()
    shap_df = load_shap()
    fundamentals = load_fundamentals()
    error_hist = load_error_history()
    news = load_news()

    if forecasts.empty:
        st.markdown(
            '<div class="topbar"><div class="brand-lockup">'
            '<p class="brand-wordmark">NSE <span>Predictor</span></p>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.error(
            "No forecasts found. Run ingest + train first:\n\n"
            "`python scripts/download_wanjawa.py`\n"
            "`python scripts/run_ingest.py`\n"
            "`python scripts/train_evaluate.py`"
        )
        return

    stocks = sorted(forecasts["stock_code"].unique().tolist())
    hits = hit_rate_map(error_hist, stocks)

    if "stock" not in st.session_state:
        last = prefs.get("last_stock")
        if last in stocks:
            st.session_state["stock"] = last
        elif "SCOM" in stocks:
            st.session_state["stock"] = "SCOM"
        else:
            st.session_state["stock"] = stocks[0]

    # --- Sidebar: personalization & nav ---
    st.sidebar.markdown("### Your desk")
    search = st.sidebar.text_input("Search ticker", placeholder="e.g. SCOM")
    if search:
        q = search.strip().upper()
        matches = [s for s in stocks if q in s]
        if matches:
            pick = st.sidebar.selectbox("Matches", matches, key="search_matches")
            if st.sidebar.button("Open match", use_container_width=True):
                select_stock(pick, prefs, go_detail=True)
                st.rerun()
        else:
            st.sidebar.caption("No match.")

    fav_default = [c for c in (prefs.get("favourites") or []) if c in stocks]
    favourites = st.sidebar.multiselect(
        "Favourites",
        options=stocks,
        default=fav_default,
        help="Pinned tickers for My market",
    )
    if favourites != fav_default:
        prefs["favourites"] = favourites
        save_prefs(prefs)
        st.session_state["prefs"] = prefs

    recent = [c for c in (prefs.get("recent") or []) if c in stocks]
    if recent:
        st.sidebar.caption("Recently viewed")
        rcols = st.sidebar.columns(min(4, len(recent)))
        for i, code in enumerate(recent[:4]):
            if rcols[i % len(rcols)].button(code, key=f"recent_{code}"):
                select_stock(code, prefs, go_detail=True)
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    fav_only = st.sidebar.toggle(
        "Favourites only",
        value=bool(prefs.get("favourites_only")),
    )
    min_prob = st.sidebar.slider(
        "Min P(up)",
        0.0,
        0.9,
        float(prefs.get("min_prob_up") or 0.0),
        0.05,
    )
    min_hit = st.sidebar.slider(
        "Min hit rate",
        0.0,
        0.9,
        float(prefs.get("min_hit_rate") or 0.0),
        0.05,
        help="Uses last ~60 labelled forecasts per stock when available",
    )
    chart_window = st.sidebar.selectbox(
        "Chart window",
        list(CHART_WINDOWS.keys()),
        index=list(CHART_WINDOWS.keys()).index(prefs.get("chart_window") or "1Y")
        if (prefs.get("chart_window") or "1Y") in CHART_WINDOWS
        else 2,
    )
    if (
        fav_only != bool(prefs.get("favourites_only"))
        or min_prob != float(prefs.get("min_prob_up") or 0)
        or min_hit != float(prefs.get("min_hit_rate") or 0)
        or chart_window != prefs.get("chart_window")
    ):
        prefs["favourites_only"] = fav_only
        prefs["min_prob_up"] = min_prob
        prefs["min_hit_rate"] = min_hit
        prefs["chart_window"] = chart_window
        save_prefs(prefs)
        st.session_state["prefs"] = prefs

    stock = st.sidebar.selectbox(
        "All stocks",
        stocks,
        index=stocks.index(st.session_state["stock"])
        if st.session_state["stock"] in stocks
        else 0,
        key="sidebar_stock_select",
    )
    if stock != st.session_state.get("stock"):
        select_stock(stock, prefs, go_detail=False)
        st.session_state["stock"] = stock

    with st.sidebar.expander("How to read this", expanded=False):
        st.write(
            "Short-term prices are noisy. Models will often be wrong. "
            "Use probabilities, ranges, and error history — not a single confident number. "
            "You make the final call."
        )
        st.caption("Visual style inspired by nse.co.ke — not an official NSE product.")

    as_of = pd.to_datetime(forecasts["trade_date"]).max()
    as_of_str = as_of.strftime("%d/%m/%Y") if pd.notna(as_of) else "—"
    coll = load_collection_status()
    if coll.get("max_date"):
        cov = coll.get("coverage_pct")
        n_tk = coll.get("n_tickers")
        cov_bit = f", {n_tk} tickers" if n_tk else ""
        if cov is not None:
            cov_bit += f", coverage {cov}%"
        collected_line = (
            f"Last collected: {coll['max_date']} (+{coll.get('rows_added', 0)} rows{cov_bit})"
        )
    elif coll.get("last_run"):
        collected_line = f"Collector ran: {str(coll['last_run'])[:19]}Z (no new rows)"
    else:
        collected_line = "Collector: drop CSV into data/raw/nse/incoming/"

    monitor = load_monitor_status()
    mon_hist = load_monitor_history(14)
    if monitor.get("status"):
        m_score = monitor.get("score") or {}
        m_pipe = monitor.get("pipeline") or {}
        cov_m = m_pipe.get("coverage_pct")
        cov_s = f" · coverage {cov_m}%" if cov_m is not None else ""
        monitor_line = (
            f"Monitor: {str(monitor.get('status')).upper()}"
            f" · as_of {m_score.get('as_of') or '—'}"
            f"{cov_s}"
        )
    else:
        monitor_line = "Monitor: not run — python scripts/run_monitor.py"

    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand-lockup">
            <p class="brand-wordmark">NSE <span>Predictor</span></p>
            <span class="brand-tag">Research</span>
          </div>
          <div class="market-status">
            Market Status: <strong class="open">Research mode</strong>
            <span class="date">Forecast as of {as_of_str}</span>
            <span class="date">{collected_line}</span>
            <span class="date">{monitor_line}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_labels = ["My market", "Stock detail", "Model health"]
    default_tab = prefs.get("main_tab") if prefs.get("main_tab") in tab_labels else "My market"
    if "main_tab" not in st.session_state:
        st.session_state["main_tab"] = default_tab

    # Streamlit tabs don't support controlled index easily; use radio as primary nav
    nav = st.radio(
        "Navigate",
        tab_labels,
        index=tab_labels.index(st.session_state["main_tab"]),
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav_radio",
    )
    if nav != st.session_state["main_tab"]:
        st.session_state["main_tab"] = nav
        prefs["main_tab"] = nav
        save_prefs(prefs)

    stock = st.session_state["stock"]
    filtered = filter_forecasts(forecasts, prefs, hits)

    if nav == "My market":
        desk = filtered if not filtered.empty else forecasts
        render_ticker_bar(desk)
        st.caption(
            "Quick open from ticker symbols (buttons). Cards sorted by signal strength. "
            "Signals = 5d excess vs NASI (±1% dead zone). Filters apply from the sidebar."
        )
        show = desk.copy()
        show["_s"] = (show["prob_up"] - 0.5).abs()
        show = show.sort_values("_s", ascending=False)
        codes = show["stock_code"].tolist()
        n = min(len(codes), 8)
        if n:
            tcols = st.columns(n)
            for i in range(n):
                code = codes[i]
                with tcols[i]:
                    st.markdown(
                        f'<div style="text-align:center;margin-bottom:0.25rem">'
                        f"{logo_html(code)}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(code, key=f"tickbtn_{code}", use_container_width=True):
                        select_stock(code, prefs, go_detail=True)
                        st.rerun()
        desk_codes = [c for c in stocks if not str(c).startswith("^")]
        render_movers_text_boxes(prices, universe=desk_codes, prefs=prefs, top_n=5)
        favs = [c for c in (prefs.get("favourites") or []) if c in stocks]
        overview_focus = favs or [c for c in codes if not str(c).startswith("^")]
        render_market_overview(prices, prefs, focus_codes=overview_focus)
        render_signal_cards(filtered, prices, hits, prefs, active=stock)

    elif nav == "Stock detail":
        st.markdown(
            f'<div class="detail-title">{logo_html(stock, large=True)}'
            f"<div><h3>{stock}</h3>"
            f'<span class="co-name">{company_display_name(stock)}</span></div></div>',
            unsafe_allow_html=True,
        )
        render_stock_detail(
            stock,
            forecasts,
            prices,
            shap_df,
            fundamentals,
            error_hist,
            news,
            prefs,
        )

    else:
        render_model_health(evaluation, monitor=monitor, mon_hist=mon_hist)


if __name__ == "__main__":
    main()
