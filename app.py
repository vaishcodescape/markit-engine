"""
thesis-engine — Streamlit Dashboard

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
    page_icon="𓄀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .metric-card {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
  }
  .ticker-label { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
  .role-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .tag-intact  { background:#14532d; color:#86efac; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
  .tag-shaken  { background:#78350f; color:#fcd34d; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
  .tag-broken  { background:#450a0a; color:#fca5a5; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
  .analysis-box {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 8px;
    padding: 20px;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    line-height: 1.55;
    color: #cbd5e1;
    max-height: 600px;
    overflow-y: auto;
  }
  div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0b0e1a",
    plot_bgcolor="#131929",
    margin=dict(l=40, r=20, t=40, b=40),
    font=dict(color="#e2e8f0"),
)


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


def pct_color(val: float) -> str:
    return "#22c55e" if val >= 0 else "#ef4444"


def thesis_tag(status_text: str) -> str:
    upper = status_text.upper()
    if "INTACT" in upper:
        return '<span class="tag-intact">INTACT</span>'
    if "BROKEN" in upper:
        return '<span class="tag-broken">BROKEN</span>'
    if "SHAKEN" in upper:
        return '<span class="tag-shaken">SHAKEN</span>'
    return f"<span>{status_text}</span>"


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_data(tickers_key: str, _portfolio: dict) -> dict:
    """
    Fetch all data modules in parallel (5-min TTL cache).
    `tickers_key` is a cache-busting string derived from the portfolio tickers.
    """
    tickers = [s["ticker"] for s in _portfolio["portfolio"]]
    watchlist = [s["ticker"] for s in _portfolio.get("watchlist", [])]
    all_tickers = tickers + watchlist
    weekend = is_weekend()

    from modules.congress_trades import fetch_congress_trades
    from modules.google_trends import fetch_google_trends
    from modules.news_rss import fetch_news_rss
    from modules.reddit import fetch_reddit_sentiment

    tasks: dict[str, object] = {
        "news_rss": lambda: fetch_news_rss(all_tickers),
        "reddit": lambda: fetch_reddit_sentiment(all_tickers),
        "google_trends": lambda: fetch_google_trends(all_tickers),
        "congress_trades": lambda: fetch_congress_trades(tickers, _portfolio),
    }

    if not weekend:
        from modules.fundamentals import fetch_fundamentals
        from modules.insider_trades import fetch_insider_trades
        from modules.macro import fetch_macro
        from modules.prices import fetch_prices
        from modules.technicals import calculate_technicals

        tasks.update({
            "prices": lambda: fetch_prices(tickers, _portfolio),
            "fundamentals": lambda: fetch_fundamentals(tickers),
            "technicals": lambda: calculate_technicals(tickers),
            "macro": lambda: fetch_macro(),
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
    """Call Claude synchronously and return the response text."""
    import anthropic

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyzer import build_prompt, update_context_files

    from modules.sustainability import extract_sustainability_signals

    data["sustainability"] = extract_sustainability_signals(data, portfolio)

    rag_context = ""
    try:
        from modules.rag_agent import RAGAgent
        rag = RAGAgent(persist_dir="vector_store/")
        rag_context = rag.enrich_prompt(portfolio, data)
    except Exception:
        pass

    prompt = build_prompt(portfolio, data, weekend=is_weekend(), rag_context=rag_context)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    response = msg.content[0].text
    update_context_files(response, portfolio)
    return response


# ── Charts ────────────────────────────────────────────────────────────────────

def pnl_history_chart(logs: list[dict]) -> go.Figure:
    rows = []
    for entry in logs:
        ts = entry.get("timestamp", "")
        rows.append({
            "timestamp": pd.to_datetime(ts),
            "pnl_pct": entry.get("pnl_pct", 0),
        })
    df = pd.DataFrame(rows).sort_values("timestamp").dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["pnl_pct"],
        mode="lines+markers",
        name="Portfolio P&L %",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        hovertemplate="%{x|%b %d %H:%M}<br>P&L: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis_title=None,
        yaxis_title="P&L %",
        showlegend=False,
    )
    return fig


def ticker_price_chart(logs: list[dict], ticker: str) -> go.Figure | None:
    rows = []
    for entry in logs:
        ts = entry.get("timestamp", "")
        price = entry.get("prices", {}).get(ticker)
        if price is not None:
            rows.append({"timestamp": pd.to_datetime(ts), "price": price})
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        mode="lines",
        line=dict(color="#a78bfa", width=1.5),
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=120,
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def analyst_donut(f: dict) -> go.Figure:
    buys = f.get("analyst_buys", 0) or 0
    holds = f.get("analyst_holds", 0) or 0
    sells = f.get("analyst_sells", 0) or 0
    fig = go.Figure(go.Pie(
        labels=["Buy", "Hold", "Sell"],
        values=[buys, holds, sells],
        hole=0.6,
        marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
        textinfo="none",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=140,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(portfolio: dict, logs: list[dict]) -> tuple[bool, bool]:
    """Returns (refresh_clicked, run_analysis_clicked)."""
    with st.sidebar:
        st.markdown("## 📈 thesis-engine")
        st.divider()

        if portfolio:
            meta = portfolio.get("meta", {})
            st.caption(f"**Goal:** {meta.get('investor_goal', '—')[:80]}")
            st.caption(f"**Risk:** {meta.get('risk_profile', '—')[:60]}")
            st.divider()

            tickers = [s["ticker"] for s in portfolio["portfolio"]]
            watchlist = [s["ticker"] for s in portfolio.get("watchlist", [])]
            st.markdown(f"**Portfolio** `{'  ·  '.join(tickers)}`")
            if watchlist:
                st.markdown(f"**Watchlist** `{'  ·  '.join(watchlist)}`")
            st.divider()

        mode = "WEEKEND" if is_weekend() else "WEEKDAY"
        st.markdown(f"**Mode:** `{mode}`")
        st.markdown(f"**Now:** `{datetime.now().strftime('%b %d %H:%M')}`")

        if logs:
            last_ts = logs[-1].get("timestamp", "")
            try:
                last_dt = datetime.fromisoformat(last_ts)
                st.markdown(f"**Last run:** `{last_dt.strftime('%b %d %H:%M')}`")
            except Exception:
                pass

        st.divider()
        refresh = st.button("⟳  Refresh Data", use_container_width=True)
        run_btn = st.button("🤖  Run Analysis", use_container_width=True, type="primary")

        if refresh:
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("Data cache: 5 min · Logs: `logs/stock_analysis.jsonl`")

    return refresh, run_btn


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def tab_overview(portfolio: dict, data: dict, logs: list[dict]) -> None:
    prices = data.get("prices", {})
    funds = data.get("fundamentals", {})

    # ── Portfolio-level banner ──
    total_pnl_pct = prices.get("__portfolio_pnl_pct__", None)
    total_pnl = prices.get("__total_pnl__", None)
    total_current = prices.get("__total_current__", None)
    total_invested = prices.get("__total_invested__", None)

    if total_current is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${total_current:,.0f}")
        c2.metric("Total Invested", f"${total_invested:,.0f}")
        c3.metric("Net P&L", f"${total_pnl:+,.0f}", f"{total_pnl_pct:+.1f}%")
        c4.metric("Positions", len(portfolio["portfolio"]))
    else:
        st.info("Price data unavailable — markets may be closed or API key missing.")

    st.divider()

    # ── P&L history chart ──
    if logs:
        st.markdown("#### Portfolio P&L History")
        st.plotly_chart(pnl_history_chart(logs), use_container_width=True)
        st.divider()

    # ── Stock cards ──
    st.markdown("#### Holdings")
    for stock in portfolio["portfolio"]:
        ticker = stock["ticker"]
        p = prices.get(ticker, {})
        f = funds.get(ticker, {}) if funds else {}

        with st.container():
            cols = st.columns([1.8, 1.1, 1.1, 1.1, 1.1, 1.1, 1.5])

            # Name / role
            with cols[0]:
                st.markdown(f'<span class="ticker-label">{ticker}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="role-label">{stock.get("role","").upper()}</span>', unsafe_allow_html=True)

            if p.get("error"):
                cols[1].warning("No data")
            else:
                price = p.get("price", 0)
                change_pct = p.get("change_pct", 0)
                pnl = p.get("pnl", 0)
                pnl_pct = p.get("pnl_pct", 0)

                cols[1].metric("Price", f"${price:,.2f}", f"{change_pct:+.2f}%")
                cols[2].metric("P&L", f"${pnl:+,.0f}", f"{pnl_pct:+.1f}%")
                cols[3].metric("Cost", f"${p.get('blended_cost', 0):,.2f}")
                cols[4].metric("Shares", f"{p.get('shares', 0):.2f}")

                if f and not f.get("error"):
                    cols[5].metric("Fwd P/E", f.get("fwd_pe") or "—")
                    cols[6].metric("Analyst", f"{f.get('analyst_consensus','—')} ({f.get('analyst_count',0)})")
                else:
                    cols[5].metric("Fwd P/E", "—")
                    cols[6].metric("Analyst", "—")

            # Mini price chart from logs
            mini = ticker_price_chart(logs, ticker)
            if mini:
                st.plotly_chart(mini, use_container_width=True, config={"displayModeBar": False})

        st.divider()

    # ── Watchlist ──
    watchlist = portfolio.get("watchlist", [])
    if watchlist:
        st.markdown("#### Watchlist")
        wl_cols = st.columns(len(watchlist))
        trends = data.get("google_trends", {})
        reddit = data.get("reddit", {})
        for i, w in enumerate(watchlist):
            t = w["ticker"]
            tr = trends.get(t, {})
            r = reddit.get(t, {})
            with wl_cols[i]:
                spike = " 🔥" if tr.get("spike") else ""
                st.markdown(f"**{t}**{spike}")
                st.caption(w.get("reason", "")[:80])
                if tr and not tr.get("error"):
                    st.markdown(f"Trends: `{tr.get('score','?')}/100` ({tr.get('change_pct',0):+.0f}%)")
                if r and not r.get("error"):
                    st.markdown(f"Reddit: `{r.get('mentions_24h',0)} mentions/24h` · {r.get('bullish_pct',50):.0f}% bull")


# ── Tab: Analysis ─────────────────────────────────────────────────────────────

def tab_analysis(portfolio: dict, data: dict, logs: list[dict], run_btn: bool) -> None:
    # Run new analysis if requested
    if run_btn:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not set in environment.")
        else:
            with st.spinner("Running Claude analysis…"):
                try:
                    response = run_analysis(portfolio, data)
                    st.session_state["last_analysis"] = response
                    st.session_state["last_analysis_ts"] = datetime.now().strftime("%b %d %H:%M")

                    # Append to log
                    prices = data.get("prices", {})
                    os.makedirs("logs", exist_ok=True)
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "alert_sent": False,
                        "weekend": is_weekend(),
                        "prices": {
                            k: v.get("price")
                            for k, v in prices.items()
                            if not k.startswith("__") and isinstance(v, dict) and not v.get("error")
                        },
                        "pnl_pct": prices.get("__portfolio_pnl_pct__", 0),
                        "response": response[:600],
                    }
                    with open("logs/stock_analysis.jsonl", "a") as f:
                        f.write(json.dumps(entry) + "\n")

                    st.success("Analysis complete.")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    # Show last analysis — prefer session state (just ran), then logs
    response = st.session_state.get("last_analysis")
    response_ts = st.session_state.get("last_analysis_ts")

    if not response and logs:
        last = logs[-1]
        response = last.get("response", "")
        try:
            response_ts = datetime.fromisoformat(last["timestamp"]).strftime("%b %d %H:%M")
        except Exception:
            response_ts = "unknown"

        if response:
            st.info(
                f"Showing cached response from **{response_ts}** "
                "(truncated to 600 chars — click **Run Analysis** for a fresh full response)."
            )

    if response:
        header = f"Claude's Analysis — {response_ts}" if response_ts else "Claude's Analysis"
        st.markdown(f"#### {header}")

        # Parse and highlight THESIS STATUS lines
        lines = response.split("\n")
        highlighted = []
        for line in lines:
            if "THESIS STATUS:" in line.upper():
                for word in ("INTACT", "SHAKEN", "BROKEN"):
                    if word in line.upper():
                        line = line.replace(word, f"**{word}**")
                        break
            highlighted.append(line)

        st.markdown(
            f'<div class="analysis-box">{chr(10).join(highlighted)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No analysis yet. Click **Run Analysis** in the sidebar to generate one.")

    # Historical run count
    if logs:
        st.divider()
        st.markdown(f"**{len(logs)}** analysis runs logged · "
                    f"Alerts sent: **{sum(1 for e in logs if e.get('alert_sent'))}**")


# ── Tab: Signals ──────────────────────────────────────────────────────────────

def tab_signals(portfolio: dict, data: dict) -> None:
    tickers = [s["ticker"] for s in portfolio["portfolio"]]

    news = data.get("news_rss", {})
    reddit = data.get("reddit", {})
    trends = data.get("google_trends", {})
    insider = data.get("insider_trades", {})
    congress = data.get("congress_trades", {})
    techs = data.get("technicals", {})
    funds = data.get("fundamentals", {})

    SENT_ICON = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}

    for stock in portfolio["portfolio"]:
        ticker = stock["ticker"]
        st.markdown(f"### {ticker} — {stock['name']}")

        sig_cols = st.columns(4)

        # News
        with sig_cols[0]:
            st.markdown("**📰 News**")
            t_news = news.get(ticker, [])
            if not t_news:
                st.caption("No news")
            for item in t_news[:4]:
                icon = SENT_ICON.get(item.get("sentiment", "neutral"), "⚪")
                st.markdown(
                    f"{icon} {item.get('age','?')} · *{item.get('source','?')}*  \n"
                    f"{item.get('title','')[:90]}"
                )

        # Reddit
        with sig_cols[1]:
            st.markdown("**💬 Reddit**")
            r = reddit.get(ticker, {})
            if r and not r.get("error"):
                spike = " 🔥" if r.get("spike") else ""
                st.metric("Mentions/24h", r.get("mentions_24h", 0),
                          f"{r.get('velocity_pct', 0):+.0f}% vs avg{spike}")
                bull = r.get("bullish_pct", 50)
                bear = 100 - bull
                st.progress(int(bull) / 100, text=f"{bull:.0f}% Bull / {bear:.0f}% Bear")
                tp = r.get("top_post", {})
                if tp:
                    quality = " ✅ Quality DD" if tp.get("quality") else ""
                    st.caption(f"Top: \"{tp.get('title','')[:60]}\" ({tp.get('score',0)} pts){quality}")
            else:
                st.caption("No Reddit data")

        # Trends + Technicals
        with sig_cols[2]:
            st.markdown("**📈 Trends / Technicals**")
            tr = trends.get(ticker, {})
            if tr and not tr.get("error"):
                spike = " 🔥" if tr.get("spike") else ""
                st.metric("Google Trends", f"{tr.get('score', '?')}/100",
                          f"{tr.get('change_pct', 0):+.0f}% vs 30d{spike}")

            tc = techs.get(ticker, {}) if techs else {}
            if tc and not tc.get("error"):
                rsi = tc.get("rsi", "—")
                macd = tc.get("macd_signal", "—")
                bb = tc.get("bb_position", "—")
                vol = tc.get("volume_vs_avg", 1)
                st.markdown(
                    f"RSI `{rsi}` · MACD `{macd}` · BB `{bb}`  \n"
                    f"Volume `{vol:.1f}×` avg"
                )
            else:
                st.caption("Technicals N/A (weekend or API error)")

        # Insider / Congress trades
        with sig_cols[3]:
            st.markdown("**🏛 Insider / Congress**")
            it = insider.get(ticker, {}) if insider else {}
            if it and not it.get("error") and it.get("trades"):
                net = it.get("net_sentiment", "neutral")
                net_icon = "🟢" if net == "bullish" else ("🔴" if net == "bearish" else "⚪")
                st.markdown(f"{net_icon} **{net.upper()}** net")
                st.caption(
                    f"Bought ${it.get('total_bought',0):,.0f} | "
                    f"Sold ${it.get('total_sold',0):,.0f}"
                )
                for trade in it.get("trades", [])[:3]:
                    action = "BOUGHT" if trade.get("is_buy") else "SOLD"
                    csuite = "⭐ " if trade.get("is_csuite") else ""
                    st.caption(
                        f"{csuite}{trade.get('date','?')} {trade.get('insider_name','?')} "
                        f"({trade.get('insider_title','?')}) {action} "
                        f"${trade.get('value',0):,.0f}"
                    )

            ct = congress.get(ticker, []) if congress else []
            if isinstance(ct, list) and ct:
                for trade in ct[:2]:
                    hi = " ⭐ HIGH SIGNAL" if trade.get("relevant_committee") else ""
                    st.caption(
                        f"{trade.get('name','?')} ({trade.get('chamber','?')}) "
                        f"{trade.get('transaction','?')} {trade.get('amount_range','?')} "
                        f"on {trade.get('transaction_date','?')}{hi}"
                    )
            elif not (it and it.get("trades")):
                st.caption("No insider / congress activity")

        # Analyst donut (if available)
        f = funds.get(ticker, {}) if funds else {}
        if f and not f.get("error") and f.get("analyst_count", 0) > 0:
            with st.expander(f"Analyst Breakdown ({f.get('analyst_count',0)} analysts)"):
                dc1, dc2 = st.columns([1, 2])
                dc1.plotly_chart(analyst_donut(f), use_container_width=True,
                                 config={"displayModeBar": False})
                with dc2:
                    pt = f.get("price_target")
                    if pt:
                        st.metric("Price Target", f"${pt:,.2f}")
                    st.metric("Consensus", f.get("analyst_consensus", "—"))
                    st.metric("Short Interest", f"{f.get('short_interest') or '—'}%")

        st.divider()

    # Press releases
    prs = data.get("press_releases", {})
    any_pr = any(prs.get(s["ticker"]) for s in portfolio["portfolio"])
    if any_pr:
        st.markdown("### 📋 Press Releases & 8-K Filings")
        for stock in portfolio["portfolio"]:
            ticker = stock["ticker"]
            for pr in prs.get(ticker, [])[:3]:
                pr_type = pr.get("type", "general")
                label = {"dilution": "DILUTION", "contract": "CONTRACT", "executive": "EXEC"}.get(
                    pr_type, "INFO"
                )
                st.markdown(
                    f"**[{label}]** `{ticker}` {pr.get('date','?')} · *{pr.get('source','?')}*  \n"
                    f"{pr.get('title','')}"
                )
                if pr.get("summary"):
                    st.caption(pr["summary"][:160])


# ── Tab: Macro ────────────────────────────────────────────────────────────────

def tab_macro(data: dict, logs: list[dict]) -> None:
    macro = data.get("macro", {})

    if not macro or macro.get("error"):
        st.info(
            "Macro data unavailable. "
            + (f"Error: {macro.get('error')}" if macro else "Markets may be closed.")
        )
    else:
        st.markdown("#### Rates & Inflation")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Fed Rate", f"{macro.get('fed_rate', '—')}%")
        r2.metric("10yr Treasury", f"{macro.get('treasury_10y', '—')}%")
        r3.metric("2yr Treasury", f"{macro.get('treasury_2y', '—')}%")
        r4.metric("CPI", f"{macro.get('cpi', '—')}%", macro.get("cpi_trend", ""))
        r5.metric("PCE", f"{macro.get('pce', '—')}%")

        st.divider()

        st.markdown("#### Market Conditions")
        m1, m2, m3, m4 = st.columns(4)
        vix = macro.get("vix")
        vix_label = "⚠️ Elevated" if vix and float(vix) > 20 else ("🔴 Fear" if vix and float(vix) > 30 else "🟢 Low")
        m1.metric("VIX", f"{vix or '—'}", vix_label)
        m2.metric("Oil WTI", f"${macro.get('oil_wti', '—')}")
        m3.metric("DXY (USD)", f"{macro.get('dxy', '—')}")
        m4.metric("Unemployment", f"{macro.get('unemployment', '—')}%")

        yield_curve = macro.get("yield_curve", "N/A")
        curve_icon = "⚠️ Inverted" if "inv" in str(yield_curve).lower() else f"✅ {yield_curve}"
        st.markdown(f"**Yield Curve:** {curve_icon}")

    st.divider()

    # Macro history from logs (VIX proxy via pnl as we don't store macro in logs)
    world = data.get("world_news", {})
    if world and not world.get("error"):
        st.markdown("#### Global Sentiment (GDELT)")
        tone = world.get("tone_score", "N/A")
        tone_label = world.get("tone_label", "")
        st.metric("Global Tone", f"{tone}/10", tone_label)
        for evt in world.get("top_events", [])[:5]:
            st.markdown(f"- {evt}")
        for theme, td in world.get("theme_summaries", {}).items():
            st.caption(f"[{theme}] avg tone: {td.get('avg_tone', '—')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    portfolio = load_portfolio()

    if not portfolio:
        st.error(
            "**`stocks.yaml` not found.**  \n"
            "Copy the example file to get started:\n"
            "```\ncp stocks.yaml.example stocks.yaml\n```"
        )
        st.stop()

    logs = load_logs()

    refresh_clicked, run_btn = render_sidebar(portfolio, logs)

    tickers_key = ",".join(s["ticker"] for s in portfolio["portfolio"])

    # Fetch live data (cached 5 min)
    with st.spinner("Loading data…"):
        try:
            data = fetch_live_data(tickers_key, portfolio)
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            data = {}

    tabs = st.tabs(["📊 Overview", "🤖 Analysis", "📡 Signals", "🌐 Macro"])

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
