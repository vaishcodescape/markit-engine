"""
RAG utility helpers — document builders and formatters.

All functions are pure (no I/O, no external calls) so they can be
unit-tested without any API keys or vector store setup.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """
    Split long text into overlapping chunks for embedding.

    Splits on sentence boundaries (". ") where possible so chunks are
    semantically coherent. Falls back to hard char splits for very long
    sentences.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    sentences = text.replace("\n", " ").split(". ")
    current = ""

    for sent in sentences:
        candidate = (current + ". " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # sentence itself longer than max_chars → hard split with overlap
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars - overlap):
                    chunks.append(sent[i : i + max_chars].strip())
                current = sent[-(overlap):] if len(sent) > overlap else sent
            else:
                current = sent

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Retrieval result formatter
# ---------------------------------------------------------------------------

def format_historical_context(
    ticker_results: dict[str, list[dict[str, Any]]],
    macro_results: list[dict[str, Any]],
) -> str:
    """
    Format retrieval results into the Layer 15 prompt block.

    ticker_results: {ticker: [{"date": str, "summary": str, "distance": float}, ...]}
    macro_results:  [{"date": str, "summary": str, "distance": float}, ...]

    Returns an empty string if there are no meaningful results.
    """
    lines: list[str] = []

    for ticker, hits in ticker_results.items():
        if not hits:
            continue
        lines.append(f"\n{ticker} — {len(hits)} historical parallel(s):")
        for h in hits[:2]:  # cap at 2 per ticker
            date_str = h.get("date", "unknown")
            summary = h.get("summary", "")[:200]
            lines.append(f"  [{date_str}] {summary}")

    if macro_results:
        hit = macro_results[0]
        date_str = hit.get("date", "unknown")
        summary = hit.get("summary", "")[:200]
        lines.append(f"\nMACRO regime parallel [{date_str}]: {summary}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document builders — one per collection type
# ---------------------------------------------------------------------------

def build_analysis_doc(
    ticker: str,
    response_excerpt: str,
    price: float | None,
    pnl_pct: float | None,
    date: str,
) -> str:
    """Build text document for the 'analyses' collection."""
    price_str = f"${price:.2f}" if price is not None else "N/A"
    pnl_str = f"{pnl_pct:+.1f}%" if pnl_pct is not None else "N/A"
    excerpt = response_excerpt[:400].replace("\n", " ")
    return (
        f"TICKER: {ticker} | DATE: {date} | PRICE: {price_str} | P&L: {pnl_str}\n"
        f"{excerpt}"
    )


def build_news_doc(
    ticker: str,
    headline: str,
    sentiment: str,
    source: str,
    date: str,
) -> str:
    """Build text document for the 'news' collection."""
    return (
        f"TICKER: {ticker} | SOURCE: {source} | DATE: {date} | SENTIMENT: {sentiment}\n"
        f"HEADLINE: {headline[:200]}"
    )


def build_macro_doc(macro_data: dict[str, Any], date: str) -> str:
    """Build text document for the 'macro' collection."""
    vix = macro_data.get("vix", "N/A")
    t10 = macro_data.get("treasury_10y", "N/A")
    t2 = macro_data.get("treasury_2y", "N/A")
    curve = macro_data.get("yield_curve", "N/A")
    spread = macro_data.get("yield_spread", "N/A")
    cpi = macro_data.get("cpi", "N/A")
    fed = macro_data.get("fed_rate", "N/A")
    oil = macro_data.get("oil_wti", "N/A")
    dxy = macro_data.get("dxy", "N/A")

    spread_str = f"{spread:+.0f}bps" if isinstance(spread, (int, float)) else str(spread)
    return (
        f"DATE: {date} | VIX: {vix} | CURVE: {curve} ({spread_str})\n"
        f"Fed: {fed}% | 10yr: {t10}% | 2yr: {t2}% | CPI: {cpi}% | Oil: ${oil} | DXY: {dxy}"
    )


def build_trade_doc(
    ticker: str,
    trade: dict[str, Any],
    trade_type: str,  # "insider" or "congress"
) -> str:
    """Build text document for the 'trades' collection."""
    if trade_type == "insider":
        action = "BOUGHT" if trade.get("is_buy") else "SOLD"
        name = trade.get("insider_name", "Unknown")
        title = trade.get("insider_title", "")
        shares = trade.get("shares", 0)
        value = trade.get("value", 0)
        date = trade.get("date", "N/A")
        csuite = " [C-SUITE]" if trade.get("is_csuite") else ""
        return (
            f"TICKER: {ticker} | DATE: {date} | TYPE: insider_trade{csuite}\n"
            f"{name} ({title}) {action} {shares:,.0f} shares = ${value:,.0f}"
        )
    else:  # congress
        name = trade.get("name", "Unknown")
        chamber = trade.get("chamber", "")
        committee = trade.get("committee", "")
        txn = trade.get("transaction", "N/A")
        amount = trade.get("amount_range", "N/A")
        date = trade.get("transaction_date", "N/A")
        signal = trade.get("signal_quality", "MEDIUM")
        return (
            f"TICKER: {ticker} | DATE: {date} | TYPE: congress_trade | SIGNAL: {signal}\n"
            f"{name} ({chamber}, {committee}) {txn} {amount}"
        )


def build_context_doc(ticker: str, timestamp: str, update_text: str) -> str:
    """Build text document for the 'context' collection."""
    return f"TICKER: {ticker} | TIMESTAMP: {timestamp}\n{update_text[:400]}"


# ---------------------------------------------------------------------------
# Summary builders for retrieval queries
# ---------------------------------------------------------------------------

def build_ticker_query(
    ticker: str,
    price: float | None,
    change_pct: float | None,
    top_headlines: list[str],
) -> str:
    """Build a short query string for finding similar past analyses."""
    price_str = f"${price:.2f}" if price is not None else "N/A"
    chg_str = f"({change_pct:+.1f}%)" if change_pct is not None else ""
    headlines = " | ".join(top_headlines[:2])
    return f"{ticker} price {price_str} {chg_str} | {headlines}"


def build_macro_query(macro_data: dict[str, Any]) -> str:
    """Build a query string for finding similar macro regimes."""
    return build_macro_doc(macro_data, datetime.now().strftime("%Y-%m-%d"))
