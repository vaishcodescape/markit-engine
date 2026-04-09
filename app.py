"""
markit-engine — Streamlit Dashboard

Tabs: Overview · Analysis · Signals · Macro
Live data fetched from the same modules as analyzer.py (5-min cache).
Historical P&L read from logs/stock_analysis.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="markit-engine",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Root overrides */
  [data-testid="stAppViewContainer"] { background: #000; }
  [data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #1c1c1c; }
  [data-testid="stHeader"] { background: #000; }
  section.main > div { background: #000; }

  /* Hide streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Typography */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: #e8e8e8;
  }

  /* Sidebar */
  [data-testid="stSidebar"] * { color: #aaa !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] strong { color: #fff !important; }

  /* Metric cards */
  [data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    letter-spacing: -0.02em;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    color: #555 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
  [data-testid="metric-container"] {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-top: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 14px 18px 10px;
    transition: border-color 0.15s;
  }
  [data-testid="metric-container"]:hover {
    border-color: #333;
    border-top-color: #3a3a3a;
  }

  /* Tabs */
  [data-testid="stTabs"] [role="tab"] {
    color: #3a3a3a !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    padding: 10px 22px;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    transition: color 0.15s;
  }
  [data-testid="stTabs"] [role="tab"]:hover {
    color: #888 !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #fff !important;
  }
  [data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1a1a1a;
    gap: 0;
    background: transparent;
  }

  /* Buttons */
  [data-testid="baseButton-secondary"] {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    color: #888 !important;
    border-radius: 4px !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  [data-testid="baseButton-secondary"]:hover {
    border-color: #555 !important;
    color: #fff !important;
  }
  [data-testid="baseButton-primary"] {
    background: #fff !important;
    border: none !important;
    color: #000 !important;
    border-radius: 4px !important;
    font-weight: 800 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="baseButton-primary"]:hover { background: #ddd !important; }

  /* Dividers */
  hr { border-color: #111 !important; margin: 14px 0; }

  /* Alert banners */
  [data-testid="stAlert"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 4px;
    color: #666 !important;
    font-size: 0.78rem !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: #0a0a0a;
    border: 1px solid #1e1e1e !important;
    border-radius: 6px;
    transition: border-color 0.15s;
  }
  [data-testid="stExpander"]:hover { border-color: #2a2a2a !important; }
  [data-testid="stExpander"] summary { color: #555 !important; font-size: 0.75rem !important; }

  /* Progress bar */
  [data-testid="stProgress"] > div > div { background: #fff !important; }
  [data-testid="stProgress"] > div { background: #1a1a1a !important; }

  /* ── Custom components ── */
  .section-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a4a4a;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 16px;
  }

  .ticker-hero {
    font-size: 1.4rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.03em;
    line-height: 1;
  }
  .ticker-sub {
    font-size: 0.68rem;
    color: #4a4a4a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
  }

  .analysis-terminal {
    background: #030303;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    padding: 24px 28px;
    font-family: "SF Mono", "Fira Code", Menlo, "Cascadia Code", monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    line-height: 1.75;
    color: #999;
    max-height: 660px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #222 #030303;
  }
  .analysis-terminal::-webkit-scrollbar { width: 4px; }
  .analysis-terminal::-webkit-scrollbar-track { background: #030303; }
  .analysis-terminal::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
  .hl-intact { color: #3ddc84; font-weight: 700; }
  .hl-shaken { color: #f5c542; font-weight: 700; }
  .hl-broken { color: #ff6b6b; font-weight: 700; }

  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #111;
  }
  .stat-lbl { font-size: 0.68rem; color: #4a4a4a; letter-spacing: 0.04em; }
  .stat-val { font-size: 0.82rem; font-weight: 600; color: #ccc; }

  .pill {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
  }
  .pill-buy  { background: #0d2218; color: #3ddc84; border: 1px solid #1a4a32; }
  .pill-sell { background: #2a0000; color: #ff6b6b; border: 1px solid #4a0000; }
  .pill-hold { background: #111;    color: #555;    border: 1px solid #222; }

  .news-item {
    padding: 9px 0 9px 12px;
    border-bottom: 1px solid #0f0f0f;
    font-size: 0.75rem;
    line-height: 1.5;
    transition: background 0.1s;
  }
  .news-item:hover { background: rgba(255,255,255,0.015); }
  .news-pos { border-left: 2px solid #3ddc84; }
  .news-neg { border-left: 2px solid #ff6b6b; }
  .news-neu { border-left: 2px solid #252525; }
  .news-src { font-size: 0.62rem; color: #3a3a3a; letter-spacing: 0.06em; margin-bottom: 3px; }
  .news-ttl { color: #bbb; }

  .wl-card {
    background: #080808;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    padding: 14px;
    transition: border-color 0.15s;
  }
  .wl-card:hover { border-color: #2a2a2a; }
  .wl-ticker { font-size: 1.1rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
  .wl-reason { font-size: 0.7rem; color: #4a4a4a; margin-top: 5px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#000",
    plot_bgcolor="#050505",
    font=dict(color="#333", family="-apple-system, BlinkMacSystemFont, sans-serif"),
    margin=dict(l=0, r=0, t=28, b=0),
)
GREEN = "#3ddc84"
RED   = "#ff6b6b"
WHITE = "#ffffff"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists("stocks.yaml"):
        with open("stocks.yaml") as f:
            return yaml.safe_load(f)
    return {}


def load_logs() -> list[dict]:
    path = "logs/stock_analysis.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_weekend() -> bool:
    return datetime.now().weekday() >= 5


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_data(tickers_key: str, _portfolio: dict) -> dict:
    """Fetch all data modules in parallel (5-min TTL cache)."""
    tickers     = [s["ticker"] for s in _portfolio["portfolio"]]
    watchlist   = [s["ticker"] for s in _portfolio.get("watchlist", [])]
    all_tickers = tickers + watchlist
    weekend     = is_weekend()

    from modules.congress_trades import fetch_congress_trades
    from modules.google_trends import fetch_google_trends
    from modules.news_rss import fetch_news_rss

    tasks: dict[str, object] = {
        "news_rss":        lambda: fetch_news_rss(all_tickers),
        "google_trends":   lambda: fetch_google_trends(all_tickers),
        "congress_trades": lambda: fetch_congress_trades(tickers, _portfolio),
    }

    if not weekend:
        from modules.fundamentals import fetch_fundamentals
        from modules.insider_trades import fetch_insider_trades
        from modules.macro import fetch_macro
        from modules.prices import fetch_prices
        from modules.technicals import calculate_technicals

        tasks.update({
            "prices":         lambda: fetch_prices(tickers, _portfolio),
            "fundamentals":   lambda: fetch_fundamentals(tickers),
            "technicals":     lambda: calculate_technicals(tickers),
            "macro":          lambda: fetch_macro(),
            "insider_trades": lambda: fetch_insider_trades(tickers),
        })

    results: dict = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        try:
            for future in as_completed(futures, timeout=90):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = {"error": str(exc)}
        except TimeoutError:
            for future, name in futures.items():
                if name not in results:
                    future.cancel()
                    results[name] = {"error": "timed out"}

    return results


def run_analysis(portfolio: dict, data: dict) -> str:
    """Call Claude and return the response text."""
    import anthropic

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyzer import build_prompt, update_context_files
    from modules.sustainability import extract_sustainability_signals

    data["sustainability"] = extract_sustainability_signals(data, portfolio)

    rag_context = ""
    try:
        from modules.rag_agent import RAGAgent
        rag         = RAGAgent(persist_dir="vector_store/")
        rag_context = rag.enrich_prompt(portfolio, data)
    except Exception:
        pass

    prompt = build_prompt(portfolio, data, weekend=is_weekend(), rag_context=rag_context)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg    = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    response = msg.content[0].text
    update_context_files(response, portfolio)
    return response


# ── Charts ────────────────────────────────────────────────────────────────────

def pnl_history_chart(logs: list[dict]) -> go.Figure:
    rows = [
        {"t": pd.to_datetime(e["timestamp"]), "pnl": e.get("pnl_pct", 0)}
        for e in logs
    ]
    df = pd.DataFrame(rows).sort_values("t").dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["pnl"],
        mode="lines",
        line=dict(color=WHITE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,255,255,0.03)",
        hovertemplate="%{x|%b %d %H:%M}<br><b>%{y:+.2f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#1c1c1c", line_width=1)
    fig.update_layout(
        **PLOTLY_BASE,
        height=180,
        xaxis=dict(showgrid=False, color="#222", tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#0d0d0d", color="#222",
                   ticksuffix="%", tickfont=dict(size=9)),
        showlegend=False,
        title=dict(text="P&L HISTORY", font=dict(size=8, color="#222"), x=0),
    )
    return fig


def ticker_sparkline(logs: list[dict], ticker: str) -> go.Figure | None:
    rows = []
    for e in logs:
        price = e.get("prices", {}).get(ticker)
        if price is not None:
            rows.append({"t": pd.to_datetime(e["timestamp"]), "p": price})
    if len(rows) < 2:
        return None
    df    = pd.DataFrame(rows).sort_values("t")
    color = GREEN if df["p"].iloc[-1] >= df["p"].iloc[0] else RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["p"],
        mode="lines",
        line=dict(color=color, width=1.2),
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=55,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def analyst_donut(f: dict) -> go.Figure:
    buys  = f.get("analyst_buys", 0) or 0
    holds = f.get("analyst_holds", 0) or 0
    sells = f.get("analyst_sells", 0) or 0
    fig   = go.Figure(go.Pie(
        labels=["Buy", "Hold", "Sell"],
        values=[buys, holds, sells],
        hole=0.68,
        marker_colors=[GREEN, "#444", RED],
        textinfo="none",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **{**PLOTLY_BASE, "margin": dict(l=0, r=0, t=0, b=0)},
        height=120,
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(portfolio: dict, logs: list[dict]) -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown(
            "<div style='font-size:1rem;font-weight:800;color:#fff;"
            "letter-spacing:-0.01em;padding:2px 0 20px;'>▲ markit-engine</div>",
            unsafe_allow_html=True,
        )

        if portfolio:
            meta = portfolio.get("meta", {})
            st.markdown(
                f"<div style='font-size:0.62rem;color:#444;text-transform:uppercase;"
                f"letter-spacing:0.12em;margin-bottom:3px;'>Goal</div>"
                f"<div style='font-size:0.75rem;color:#777;margin-bottom:18px;line-height:1.4;'>"
                f"{meta.get('investor_goal','—')[:80]}</div>",
                unsafe_allow_html=True,
            )
            tickers   = [s["ticker"] for s in portfolio["portfolio"]]
            watchlist = [s["ticker"] for s in portfolio.get("watchlist", [])]

            st.markdown(
                "<div style='font-size:0.62rem;color:#444;text-transform:uppercase;"
                "letter-spacing:0.12em;margin-bottom:6px;'>Portfolio</div>",
                unsafe_allow_html=True,
            )
            for t in tickers:
                st.markdown(
                    f"<div style='font-size:0.82rem;font-weight:700;color:#bbb;"
                    f"padding:2px 0;'>{t}</div>",
                    unsafe_allow_html=True,
                )
            if watchlist:
                st.markdown(
                    "<div style='font-size:0.62rem;color:#444;text-transform:uppercase;"
                    "letter-spacing:0.12em;margin:14px 0 6px;'>Watchlist</div>",
                    unsafe_allow_html=True,
                )
                for t in watchlist:
                    st.markdown(
                        f"<div style='font-size:0.75rem;color:#444;padding:2px 0;'>{t}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr>", unsafe_allow_html=True)

        mode    = "WEEKEND" if is_weekend() else "WEEKDAY"
        now_str = datetime.now().strftime("%b %d  %H:%M")
        st.markdown(
            f"<div style='font-size:0.65rem;color:#444;line-height:2.2;'>"
            f"Mode &nbsp;&nbsp;<span style='color:#777;'>{mode}</span><br>"
            f"Now &nbsp;&nbsp;&nbsp;<span style='color:#777;'>{now_str}</span>",
            unsafe_allow_html=True,
        )
        if logs:
            try:
                last_dt = datetime.fromisoformat(logs[-1]["timestamp"])
                st.markdown(
                    f"<span style='font-size:0.65rem;color:#444;'>"
                    f"Last run &nbsp;<span style='color:#666;'>"
                    f"{last_dt.strftime('%b %d %H:%M')}</span></span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        st.markdown("<hr>", unsafe_allow_html=True)
        refresh = st.button("Refresh Data", use_container_width=True)
        run_btn = st.button("Run Analysis", use_container_width=True, type="primary")

        if refresh:
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            "<div style='font-size:0.6rem;color:#333;margin-top:14px;'>cache 5 min</div>",
            unsafe_allow_html=True,
        )

    return refresh, run_btn


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def tab_overview(portfolio: dict, data: dict, logs: list[dict]) -> None:
    prices = data.get("prices", {})
    funds  = data.get("fundamentals", {})

    total_pnl_pct  = prices.get("__portfolio_pnl_pct__", None)
    total_pnl      = prices.get("__total_pnl__", None)
    total_current  = prices.get("__total_current__", None)
    total_invested = prices.get("__total_invested__", None)

    if total_current is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${total_current:,.0f}")
        c2.metric("Total Invested",  f"${total_invested:,.0f}")
        c3.metric("Net P&L",         f"${total_pnl:+,.0f}", f"{total_pnl_pct:+.2f}%")
        c4.metric("Positions",       len(portfolio["portfolio"]))
    else:
        st.markdown(
            "<div style='font-size:0.75rem;color:#444;padding:10px 0;'>"
            "Price data unavailable — markets may be closed.</div>",
            unsafe_allow_html=True,
        )

    if logs:
        st.plotly_chart(pnl_history_chart(logs), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown(
        "<div class='section-label' style='margin-top:4px;'>Holdings</div>",
        unsafe_allow_html=True,
    )

    for stock in portfolio["portfolio"]:
        ticker = stock["ticker"]
        p = prices.get(ticker, {})
        f = funds.get(ticker, {}) if funds else {}

        col_info, col_metrics, col_spark = st.columns([2, 5, 2])

        with col_info:
            st.markdown(
                f"<div class='ticker-hero'>{ticker}</div>"
                f"<div class='ticker-sub'>{stock.get('name','')}</div>"
                f"<div style='margin-top:6px;font-size:0.6rem;color:#222;"
                f"letter-spacing:0.1em;'>{stock.get('role','').upper()}</div>",
                unsafe_allow_html=True,
            )

        with col_metrics:
            if p.get("error"):
                st.markdown(
                    "<span style='font-size:0.72rem;color:#222;'>no data</span>",
                    unsafe_allow_html=True,
                )
            else:
                price      = p.get("price", 0)
                change_pct = p.get("change_pct", 0)
                pnl        = p.get("pnl", 0)
                pnl_pct    = p.get("pnl_pct", 0)
                fwd_pe     = f.get("fwd_pe", "—") if f and not f.get("error") else "—"
                consensus  = f.get("analyst_consensus", "—") if f and not f.get("error") else "—"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price",   f"${price:,.2f}",                  f"{change_pct:+.2f}%")
                m2.metric("P&L",     f"${pnl:+,.0f}",                   f"{pnl_pct:+.1f}%")
                m3.metric("Cost",    f"${p.get('blended_cost',0):,.2f}")
                m4.metric("Fwd P/E", str(fwd_pe))
                m5.metric("Analyst", str(consensus))

        with col_spark:
            spark = ticker_sparkline(logs, ticker)
            if spark:
                st.plotly_chart(spark, use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown("<hr>", unsafe_allow_html=True)

    # Watchlist
    watchlist = portfolio.get("watchlist", [])
    if watchlist:
        st.markdown(
            "<div class='section-label' style='margin-top:2px;'>Watchlist</div>",
            unsafe_allow_html=True,
        )
        trends = data.get("google_trends", {})
        cols   = st.columns(min(len(watchlist), 5))
        for i, w in enumerate(watchlist):
            t  = w["ticker"]
            tr = trends.get(t, {})
            with cols[i]:
                spike_html = (
                    " <span style='font-size:0.6rem;color:#f5c542;'>SPIKE</span>"
                    if tr.get("spike") else ""
                )
                score_html = ""
                if tr and not tr.get("error"):
                    chg   = tr.get("change_pct", 0)
                    chg_c = GREEN if chg >= 0 else RED
                    score_html = (
                        f"<div style='font-size:0.68rem;color:#333;margin-top:8px;'>"
                        f"Trends <span style='color:#666;'>{tr.get('score','—')}/100</span> "
                        f"<span style='color:{chg_c};font-size:0.65rem;'>{chg:+.0f}%</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div class='wl-card'>"
                    f"<div class='wl-ticker'>{t}{spike_html}</div>"
                    f"<div class='wl-reason'>{w.get('reason','')[:90]}</div>"
                    f"{score_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ── Tab: Analysis ─────────────────────────────────────────────────────────────

def tab_analysis(portfolio: dict, data: dict, logs: list[dict], run_btn: bool) -> None:
    if run_btn:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not set.")
        else:
            with st.spinner("Calling Claude…"):
                try:
                    response = run_analysis(portfolio, data)
                    st.session_state["last_analysis"]    = response
                    st.session_state["last_analysis_ts"] = datetime.now().strftime("%b %d %H:%M")

                    prices = data.get("prices", {})
                    os.makedirs("logs", exist_ok=True)
                    entry = {
                        "timestamp":  datetime.now().isoformat(),
                        "alert_sent": False,
                        "weekend":    is_weekend(),
                        "prices": {
                            k: v.get("price")
                            for k, v in prices.items()
                            if not k.startswith("__") and isinstance(v, dict)
                            and not v.get("error")
                        },
                        "pnl_pct":  prices.get("__portfolio_pnl_pct__", 0),
                        "response": response[:600],
                    }
                    with open("logs/stock_analysis.jsonl", "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    response    = st.session_state.get("last_analysis")
    response_ts = st.session_state.get("last_analysis_ts")

    if not response and logs:
        last = logs[-1]
        response = last.get("response", "")
        try:
            response_ts = datetime.fromisoformat(last["timestamp"]).strftime("%b %d %H:%M")
        except Exception:
            response_ts = "unknown"
        if response:
            st.markdown(
                f"<div style='font-size:0.68rem;color:#444;margin-bottom:14px;'>"
                f"Showing cached response from {response_ts} — "
                f"click <strong style='color:#777;'>Run Analysis</strong> for a fresh one.</div>",
                unsafe_allow_html=True,
            )

    if response:
        ts_str = f"  —  {response_ts}" if response_ts else ""
        st.markdown(
            f"<div class='section-label'>Claude's Analysis{ts_str}</div>",
            unsafe_allow_html=True,
        )
        highlighted = (
            response
            .replace("INTACT", '<span class="hl-intact">INTACT</span>')
            .replace("SHAKEN", '<span class="hl-shaken">SHAKEN</span>')
            .replace("BROKEN", '<span class="hl-broken">BROKEN</span>')
        )
        st.markdown(
            f'<div class="analysis-terminal">{highlighted}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.82rem;color:#222;padding:48px 0;text-align:center;'>"
            "No analysis yet — click <strong style='color:#444;'>Run Analysis</strong>.</div>",
            unsafe_allow_html=True,
        )

    if logs:
        st.markdown(
            f"<div style='font-size:0.65rem;color:#3a3a3a;margin-top:18px;'>"
            f"{len(logs)} runs logged · "
            f"{sum(1 for e in logs if e.get('alert_sent'))} alerts fired</div>",
            unsafe_allow_html=True,
        )


# ── Tab: Signals ──────────────────────────────────────────────────────────────

def tab_signals(portfolio: dict, data: dict) -> None:
    news     = data.get("news_rss", {})
    trends   = data.get("google_trends", {})
    insider  = data.get("insider_trades", {})
    congress = data.get("congress_trades", {})
    techs    = data.get("technicals", {})
    funds    = data.get("fundamentals", {})

    SENT_CLASS = {"positive": "news-pos", "negative": "news-neg", "neutral": "news-neu"}

    for stock in portfolio["portfolio"]:
        ticker = stock["ticker"]

        st.markdown(
            f"<div style='font-size:0.95rem;font-weight:800;color:#fff;"
            f"letter-spacing:-0.02em;padding:4px 0 2px;'>{ticker}"
            f"<span style='font-size:0.68rem;font-weight:400;color:#4a4a4a;"
            f"margin-left:10px;'>{stock['name']}</span></div>",
            unsafe_allow_html=True,
        )

        col_news, col_sig, col_trade = st.columns(3)

        # News ─────────────────────────────────────────────────────────────────
        with col_news:
            st.markdown(
                "<div class='section-label' style='margin-top:8px;'>News</div>",
                unsafe_allow_html=True,
            )
            t_news = news.get(ticker, [])
            if not t_news:
                st.markdown(
                    "<div style='font-size:0.7rem;color:#3a3a3a;'>No recent news</div>",
                    unsafe_allow_html=True,
                )
            for item in t_news[:5]:
                cls = SENT_CLASS.get(item.get("sentiment", "neutral"), "news-neu")
                st.markdown(
                    f"<div class='news-item {cls}'>"
                    f"<div class='news-src'>{item.get('source','?')}  ·  {item.get('age','?')}</div>"
                    f"<div class='news-ttl'>{item.get('title','')[:100]}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Trends + Technicals ──────────────────────────────────────────────────
        with col_sig:
            st.markdown(
                "<div class='section-label' style='margin-top:8px;'>Signals</div>",
                unsafe_allow_html=True,
            )
            tr = trends.get(ticker, {})
            if tr and not tr.get("error"):
                spike_s = (" <span style='color:#f5c542;font-size:0.62rem;'>SPIKE</span>"
                           if tr.get("spike") else "")
                chg_c   = GREEN if tr.get("change_pct", 0) >= 0 else RED
                st.markdown(
                    f"<div class='stat-row'>"
                    f"<span class='stat-lbl'>Google Trends{spike_s}</span>"
                    f"<span class='stat-val'>{tr.get('score','—')}/100 "
                    f"<span style='color:{chg_c};font-size:0.72rem;'>"
                    f"{tr.get('change_pct',0):+.0f}%</span></span></div>",
                    unsafe_allow_html=True,
                )

            tc = techs.get(ticker, {}) if techs else {}
            if tc and not tc.get("error"):
                for label, val in [
                    ("RSI",    tc.get("rsi", "—")),
                    ("MACD",   tc.get("macd_signal", "—")),
                    ("BB",     tc.get("bb_position", "—")),
                    ("Volume", f"{tc.get('volume_vs_avg',1):.1f}× avg"),
                ]:
                    st.markdown(
                        f"<div class='stat-row'>"
                        f"<span class='stat-lbl'>{label}</span>"
                        f"<span class='stat-val'>{val}</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div style='font-size:0.7rem;color:#3a3a3a;'>Technicals N/A</div>",
                    unsafe_allow_html=True,
                )

        # Insider / Congress ───────────────────────────────────────────────────
        with col_trade:
            st.markdown(
                "<div class='section-label' style='margin-top:8px;'>Insider / Congress</div>",
                unsafe_allow_html=True,
            )
            it = insider.get(ticker, {}) if insider else {}
            if it and not it.get("error") and it.get("trades"):
                net   = it.get("net_sentiment", "neutral")
                net_c = GREEN if net == "bullish" else (RED if net == "bearish" else "#444")
                st.markdown(
                    f"<div style='font-size:0.72rem;font-weight:700;color:{net_c};"
                    f"margin-bottom:8px;letter-spacing:0.06em;'>{net.upper()}</div>"
                    f"<div class='stat-row'>"
                    f"<span class='stat-lbl'>Bought</span>"
                    f"<span class='stat-val' style='color:{GREEN};'>"
                    f"${it.get('total_bought',0):,.0f}</span></div>"
                    f"<div class='stat-row'>"
                    f"<span class='stat-lbl'>Sold</span>"
                    f"<span class='stat-val' style='color:{RED};'>"
                    f"${it.get('total_sold',0):,.0f}</span></div>",
                    unsafe_allow_html=True,
                )
                for trade in it.get("trades", [])[:3]:
                    is_buy = trade.get("is_buy")
                    pill_c = "pill-buy" if is_buy else "pill-sell"
                    action = "BUY" if is_buy else "SELL"
                    star   = "★ " if trade.get("is_csuite") else ""
                    st.markdown(
                        f"<div style='font-size:0.65rem;color:#333;padding:4px 0;"
                        f"border-bottom:1px solid #0d0d0d;'>{star}"
                        f"<span class='pill {pill_c}'>{action}</span> "
                        f"{trade.get('insider_name','?')} "
                        f"<span style='color:#555;'>${trade.get('value',0):,.0f}</span> "
                        f"<span style='color:#222;'>{trade.get('date','?')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div style='font-size:0.7rem;color:#3a3a3a;'>No insider activity</div>",
                    unsafe_allow_html=True,
                )

            ct = congress.get(ticker, []) if congress else []
            if isinstance(ct, list) and ct:
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                for trade in ct[:2]:
                    hi_c = WHITE if trade.get("relevant_committee") else "#444"
                    st.markdown(
                        f"<div style='font-size:0.65rem;color:{hi_c};padding:4px 0;"
                        f"border-bottom:1px solid #0d0d0d;'>"
                        f"{trade.get('name','?')} ({trade.get('chamber','?')}) "
                        f"{trade.get('transaction','?')} {trade.get('amount_range','?')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Analyst breakdown
        f = funds.get(ticker, {}) if funds else {}
        if f and not f.get("error") and f.get("analyst_count", 0) > 0:
            with st.expander(
                f"Analyst Breakdown  ·  {f.get('analyst_count',0)} analysts  ·  "
                f"{f.get('analyst_consensus','—')}",
            ):
                dc1, dc2 = st.columns([1, 2])
                dc1.plotly_chart(analyst_donut(f), use_container_width=True,
                                 config={"displayModeBar": False})
                with dc2:
                    pt = f.get("price_target")
                    if pt:
                        st.metric("Price Target", f"${pt:,.2f}")
                    st.metric("Short Interest", f"{f.get('short_interest') or '—'}%")

        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # Press releases
    prs    = data.get("press_releases", {})
    any_pr = any(prs.get(s["ticker"]) for s in portfolio["portfolio"])
    if any_pr:
        st.markdown(
            "<div class='section-label'>Press Releases & 8-K</div>",
            unsafe_allow_html=True,
        )
        for stock in portfolio["portfolio"]:
            ticker = stock["ticker"]
            for pr in prs.get(ticker, [])[:3]:
                pr_type = pr.get("type", "general")
                label   = {"dilution": "DILUTION", "contract": "CONTRACT",
                           "executive": "EXEC"}.get(pr_type, "INFO")
                lbl_c   = {"DILUTION": RED, "CONTRACT": GREEN}.get(label, "#444")
                summary = (
                    f"<div style='font-size:0.68rem;color:#444;margin-top:3px;'>"
                    f"{pr['summary'][:160]}</div>" if pr.get("summary") else ""
                )
                st.markdown(
                    f"<div style='padding:10px 0;border-bottom:1px solid #0d0d0d;'>"
                    f"<span style='color:{lbl_c};font-size:0.6rem;font-weight:800;"
                    f"letter-spacing:0.12em;'>[{label}]</span> "
                    f"<span style='color:#555;font-size:0.72rem;'>{ticker}</span> "
                    f"<span style='color:#bbb;font-size:0.78rem;'>{pr.get('title','')}</span>"
                    f"<div style='font-size:0.65rem;color:#3a3a3a;margin-top:3px;'>"
                    f"{pr.get('date','?')} · {pr.get('source','?')}</div>"
                    f"{summary}</div>",
                    unsafe_allow_html=True,
                )


# ── Tab: Macro ────────────────────────────────────────────────────────────────

def tab_macro(data: dict, logs: list[dict]) -> None:
    macro = data.get("macro", {})

    if not macro or macro.get("error"):
        st.markdown(
            "<div style='font-size:0.75rem;color:#444;padding:20px 0;'>"
            "Macro data unavailable — markets may be closed.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='section-label'>Rates & Inflation</div>",
            unsafe_allow_html=True,
        )
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Fed Rate",      f"{macro.get('fed_rate','—')}%")
        r2.metric("10yr Treasury", f"{macro.get('treasury_10y','—')}%")
        r3.metric("2yr Treasury",  f"{macro.get('treasury_2y','—')}%")
        r4.metric("CPI",           f"{macro.get('cpi','—')}%", macro.get("cpi_trend",""))
        r5.metric("PCE",           f"{macro.get('pce','—')}%")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            "<div class='section-label'>Market Conditions</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        vix       = macro.get("vix")
        vix_delta = ("Elevated" if vix and float(vix) > 20
                     else ("Fear" if vix and float(vix) > 30 else "Low"))
        m1.metric("VIX",          f"{vix or '—'}",                vix_delta)
        m2.metric("Oil WTI",      f"${macro.get('oil_wti','—')}")
        m3.metric("DXY (USD)",    f"{macro.get('dxy','—')}")
        m4.metric("Unemployment", f"{macro.get('unemployment','—')}%")

        yield_curve = macro.get("yield_curve", "N/A")
        curve_c     = RED if "inv" in str(yield_curve).lower() else GREEN
        st.markdown(
            f"<div style='font-size:0.75rem;color:#444;margin-top:10px;'>"
            f"Yield Curve  <span style='color:{curve_c};font-weight:700;'>{yield_curve}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    world = data.get("world_news", {})
    if world and not world.get("error"):
        st.markdown(
            "<div class='section-label'>Global Sentiment (GDELT)</div>",
            unsafe_allow_html=True,
        )
        tone   = world.get("tone_score", "—")
        tone_l = world.get("tone_label", "")
        tone_c = GREEN if str(tone_l).lower() in ("positive", "neutral") else RED
        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:800;color:{tone_c};"
            f"letter-spacing:-0.04em;margin-bottom:12px;'>{tone}"
            f"<span style='font-size:0.7rem;color:#333;font-weight:400;"
            f"margin-left:8px;'>/10  ·  {tone_l}</span></div>",
            unsafe_allow_html=True,
        )
        for evt in world.get("top_events", [])[:5]:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#444;padding:6px 0;"
                f"border-bottom:1px solid #0d0d0d;'>{evt}</div>",
                unsafe_allow_html=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    portfolio = load_portfolio()

    if not portfolio:
        st.markdown(
            "<div style='padding:60px 0;text-align:center;'>"
            "<div style='font-size:1rem;font-weight:800;color:#fff;'>stocks.yaml not found</div>"
            "<div style='font-size:0.78rem;color:#333;margin-top:12px;'>"
            "<code style='color:#555;'>cp stocks.yaml.example stocks.yaml</code></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    logs = load_logs()
    _, run_btn = render_sidebar(portfolio, logs)

    tickers_key = ",".join(s["ticker"] for s in portfolio["portfolio"])

    with st.spinner(""):
        try:
            data = fetch_live_data(tickers_key, portfolio)
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            data = {}

    tabs = st.tabs(["OVERVIEW", "ANALYSIS", "SIGNALS", "MACRO"])

    with tabs[0]:
        tab_overview(portfolio, data, logs)
    with tabs[1]:
        tab_analysis(portfolio, data, logs, run_btn)
    with tabs[2]:
        tab_signals(portfolio, data)
    with tabs[3]:
        tab_macro(data, logs)


if __name__ == "__main__":
    main()
