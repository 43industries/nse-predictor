"""Streamlit dashboard for NSE decision-support forecasts.

Visual language inspired by https://www.nse.co.ke/ (lime green, navy ticker, clean white).
Not affiliated with the Nairobi Securities Exchange.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history

st.set_page_config(
    page_title="NSE Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = settings.disclaimer

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
  margin: 0 0 1rem 0;
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

.stance-pill {{
  display: inline-block;
  padding: 0.3rem 0.75rem;
  border-radius: 2px;
  font-weight: 700;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}}

.stance-watch {{ background: {NSE_GREEN}; color: {NSE_NAVY}; }}
.stance-avoid {{ background: #FDECEA; color: {DOWN}; }}
.stance-neutral {{ background: #ECEFF3; color: {MUTED}; }}

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

/* Progress bar → NSE green */
.stProgress > div > div > div > div {{
  background-color: {NSE_GREEN_DARK};
}}
</style>
"""


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


def stance_from_prob(prob_up: float) -> tuple[str, str]:
    if prob_up >= 0.55:
        return "Watch", "stance-watch"
    if prob_up <= 0.45:
        return "Avoid", "stance-avoid"
    return "Neutral", "stance-neutral"


def build_watchlist(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in forecasts.iterrows():
        prob = float(r["prob_up"])
        stance, _ = stance_from_prob(prob)
        if pd.notna(r.get("ret5d_lo")) and pd.notna(r.get("ret5d_hi")):
            band = f"{r['ret5d_lo']:.1%} to {r['ret5d_hi']:.1%}"
        else:
            band = "—"
        rows.append(
            {
                "stock": r["stock_code"],
                "stance": stance,
                "P(up)": prob,
                "direction": str(r["pred_label"]).upper(),
                "return range": band,
                "price": float(r["price"]) if pd.notna(r.get("price")) else None,
                "_signal": abs(prob - 0.5),
            }
        )
    out = pd.DataFrame(rows).sort_values("_signal", ascending=False).drop(columns=["_signal"])
    return out.reset_index(drop=True)


def render_ticker_bar(forecasts: pd.DataFrame) -> None:
    """NSE-style navy ticker strip that auto-scrolls (research signals)."""
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
    # Duplicate strip for a seamless -50% loop
    track = "".join(items) + "".join(items)
    html = f'<div class="ticker-bar"><div class="ticker-track">{track}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def build_price_figure(hist: pd.DataFrame) -> go.Figure:
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
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=12, label="1Y", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#ECEFF3",
                activecolor=NSE_GREEN,
                font=dict(color=INK),
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="KES", row=1, col=1, gridcolor="#E8EAED", zeroline=False)
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="#E8EAED", zeroline=False)
    return fig


def build_shap_figure(shap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = shap_df.head(top_n).iloc[::-1].copy()
    fig = go.Figure(
        go.Bar(
            x=data["mean_abs_shap"],
            y=data["feature"],
            orientation="h",
            marker_color=NSE_GREEN_DARK,
            hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        xaxis=dict(title="Mean |SHAP|", gridcolor="#E8EAED", zeroline=False),
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


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

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
    default_ix = stocks.index("SCOM") if "SCOM" in stocks else 0
    stock = st.sidebar.selectbox("Stock", stocks, index=default_ix)

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
        collected_line = f"Last collected: {coll['max_date']} (+{coll.get('rows_added', 0)} rows)"
    elif coll.get("last_run"):
        collected_line = f"Collector ran: {str(coll['last_run'])[:19]}Z (no new rows)"
    else:
        collected_line = "Collector: drop CSV into data/raw/nse/incoming/"

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
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_ticker_bar(forecasts)

    st.markdown('<p class="section-label">Market watchlist</p>', unsafe_allow_html=True)
    watch = build_watchlist(forecasts)
    display = watch.copy()
    display["P(up)"] = display["P(up)"].map(lambda x: f"{x:.0%}")
    display["price"] = display["price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Ticker shows 5-day direction signal and P(up) — not live exchange quotes. "
        "Sorted by signal strength. Use the sidebar to open a stock."
    )

    row = forecasts[forecasts["stock_code"] == stock].iloc[0]
    prob = float(row["prob_up"])
    stance, stance_cls = stance_from_prob(prob)

    st.markdown(f"### {stock}")
    st.markdown(
        f'<span class="stance-pill {stance_cls}">{stance}</span>'
        f'<span class="one-liner"> &nbsp; P(up over 5 trading days) = {prob:.0%}</span>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0), text=f"Probability of rising: {prob:.0%}")

    acc = summarize_error_history(error_hist, stock, lookback_n=60)
    if acc.get("n"):
        st.caption(
            f"Last {acc['n']} labelled forecasts: **{acc['hit_rate']:.0%}** correct "
            f"(streak {acc['recent_streak']}; {acc['as_of_min']} → {acc['as_of_max']})."
        )
    else:
        st.caption("No per-stock error history yet — run `python scripts/train_evaluate.py`.")

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
    c4.metric("5d return range", band)

    st.markdown('<p class="section-label">Price & volume</p>', unsafe_allow_html=True)
    hist = prices[prices["stock_code"] == stock].sort_values("trade_date").tail(252)
    if not hist.empty:
        st.plotly_chart(build_price_figure(hist), width="stretch")
    else:
        st.caption("No price history for this ticker.")

    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-label">Key drivers</p>', unsafe_allow_html=True)
        if shap_df.empty:
            st.caption("SHAP drivers not available yet.")
        else:
            st.plotly_chart(build_shap_figure(shap_df), width="stretch")
            st.caption("Global mean |SHAP| on the test sample.")
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

    render_stock_news(news, stock)

    st.markdown('<p class="section-label">Model honesty vs baselines</p>', unsafe_allow_html=True)
    if evaluation:
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
    else:
        st.caption("Run training to populate evaluation_results.json")

    with st.expander("Disclaimer", expanded=False):
        st.write(DISCLAIMER)
        st.caption(
            "Visual design inspired by nse.co.ke. This research tool is not issued by, "
            "endorsed by, or affiliated with the Nairobi Securities Exchange PLC."
        )


if __name__ == "__main__":
    main()
