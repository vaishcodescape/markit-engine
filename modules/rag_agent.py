"""
RAG Agent — Voyage AI embeddings + ChromaDB vector store.

Embeds and indexes each analysis run across 5 collections, then retrieves
historically similar patterns to inject as Layer 15 in the Claude prompt.

Graceful degradation: if VOYAGE_API_KEY is unset or ChromaDB fails to
initialize, all public methods become no-ops and the system runs normally
without RAG enrichment.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from modules.rag_utils import (
    build_analysis_doc,
    build_context_doc,
    build_macro_doc,
    build_macro_query,
    build_news_doc,
    build_ticker_query,
    build_trade_doc,
    chunk_text,
    format_historical_context,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOYAGE_MODEL = "voyage-finance-2"
EMBED_BATCH_SIZE = 128
MAX_STORE_AGE_DAYS = 180
COLD_START_MIN_DOCS = 5  # don't retrieve until we have this many docs total
COLLECTION_NAMES = ("analyses", "news", "trades", "macro", "context")


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:
    """
    Thin wrapper around ChromaDB + Voyage AI for financial RAG.

    Usage in analyzer.py:
        _rag = RAGAgent(persist_dir="vector_store/")
        rag_context = _rag.enrich_prompt(portfolio, data)   # before build_prompt
        _rag.index_run(data, response, portfolio, weekend)  # after update_context_files
    """

    def __init__(self, persist_dir: str = "vector_store/") -> None:
        self._enabled = False
        self._client = None
        self._voyage = None
        self._cols: dict[str, Any] = {}

        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            print("  [rag] VOYAGE_API_KEY not set — RAG disabled")
            return

        try:
            import chromadb
            import voyageai

            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._voyage = voyageai.Client(api_key=api_key)

            for name in COLLECTION_NAMES:
                self._cols[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )

            self._enabled = True
            total = sum(c.count() for c in self._cols.values())
            print(f"  [rag] ChromaDB ready at {persist_dir!r} ({total} docs across {len(COLLECTION_NAMES)} collections)")

        except Exception as exc:
            print(f"  [rag] Init failed ({exc}) — RAG disabled")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def index_run(
        self,
        data: dict[str, Any],
        response: str,
        portfolio: dict[str, Any],
        weekend: bool = False,
    ) -> None:
        """Embed and store the current analysis run in all relevant collections."""
        if not self._enabled:
            return
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            self._index_analysis(response, data, portfolio, date_str)
            self._index_news(data, portfolio, date_str)
            self._index_trades(data, portfolio, date_str)
            if not weekend:
                self._index_macro(data, date_str)
            self._index_context_files(portfolio, date_str)

            # Monthly prune (run on the 1st of each month)
            if datetime.now().day == 1:
                self._prune_old_entries()
        except Exception as exc:
            print(f"  [rag] index_run error: {exc}")

    def enrich_prompt(self, portfolio: dict[str, Any], data: dict[str, Any]) -> str:
        """
        Retrieve historically similar events and return a formatted Layer 15 block.

        Returns an empty string when the store is too sparse (cold start guard)
        or when retrieval fails — the caller should handle "" gracefully.
        """
        if not self._enabled:
            return ""
        try:
            total = sum(c.count() for c in self._cols.values())
            if total < COLD_START_MIN_DOCS:
                return ""

            tickers = [s["ticker"] for s in portfolio.get("portfolio", [])]
            prices = data.get("prices", {})
            news = data.get("news_rss", {})
            macro = data.get("macro", {})

            ticker_results: dict[str, list[dict[str, Any]]] = {}
            for ticker in tickers:
                p = prices.get(ticker, {})
                price = p.get("price")
                change_pct = p.get("change_pct")
                headlines = [
                    item["title"]
                    for item in news.get(ticker, [])[:3]
                    if isinstance(item, dict) and "title" in item
                ]

                hits: list[dict[str, Any]] = []
                query = build_ticker_query(ticker, price, change_pct, headlines)
                hits += self._query_similar_analyses(ticker, query)
                hits += self._query_similar_news(ticker, " | ".join(headlines))
                hits += self._query_trade_patterns(ticker, data)

                # Deduplicate by date+summary prefix, keep top 2
                seen: set[str] = set()
                unique: list[dict[str, Any]] = []
                for h in hits:
                    key = (h.get("date", ""), h.get("summary", "")[:40])
                    if key not in seen:
                        seen.add(key)
                        unique.append(h)
                ticker_results[ticker] = unique[:2]

            macro_results: list[dict[str, Any]] = []
            if macro and not macro.get("error"):
                macro_results = self._query_macro_regime(macro)

            block = format_historical_context(ticker_results, macro_results)
            return block

        except Exception as exc:
            print(f"  [rag] enrich_prompt error: {exc}")
            return ""

    # -----------------------------------------------------------------------
    # Embedding helpers
    # -----------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed up to EMBED_BATCH_SIZE texts at a time."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            result = self._voyage.embed(batch, model=VOYAGE_MODEL, input_type="document")
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    def _embed_query(self, text: str) -> list[float]:
        result = self._voyage.embed([text], model=VOYAGE_MODEL, input_type="query")
        return result.embeddings[0]

    # -----------------------------------------------------------------------
    # Indexing — one method per collection
    # -----------------------------------------------------------------------

    def _index_analysis(
        self,
        response: str,
        data: dict[str, Any],
        portfolio: dict[str, Any],
        date_str: str,
    ) -> None:
        prices = data.get("prices", {})
        docs, ids, metas = [], [], []

        for stock in portfolio.get("portfolio", []):
            ticker = stock["ticker"]
            p = prices.get(ticker, {})
            if p.get("error"):
                continue

            # Extract the ticker-specific excerpt from the response
            excerpt = self._extract_ticker_excerpt(response, ticker)
            if not excerpt:
                excerpt = response[:400]

            doc = build_analysis_doc(
                ticker=ticker,
                response_excerpt=excerpt,
                price=p.get("price"),
                pnl_pct=p.get("pnl_pct"),
                date=date_str,
            )
            doc_id = f"analysis_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
            docs.append(doc)
            ids.append(doc_id)
            metas.append({
                "ticker": ticker,
                "date": date_str,
                "alert_sent": False,
                "pnl_pct": float(p.get("pnl_pct") or 0),
                "collection_type": "analysis",
            })

        if docs:
            embeddings = self._embed(docs)
            self._cols["analyses"].add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    def _index_news(
        self,
        data: dict[str, Any],
        portfolio: dict[str, Any],
        date_str: str,
    ) -> None:
        tickers = [s["ticker"] for s in portfolio.get("portfolio", [])]
        news = data.get("news_rss", {})
        prs = data.get("press_releases", {})
        docs, ids, metas = [], [], []

        for ticker in tickers:
            for item in news.get(ticker, [])[:5]:
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                doc = build_news_doc(
                    ticker=ticker,
                    headline=item["title"],
                    sentiment=item.get("sentiment", "neutral"),
                    source=item.get("source", "unknown"),
                    date=item.get("age", date_str)[:10],
                )
                doc_id = f"news_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
                docs.append(doc)
                ids.append(doc_id)
                metas.append({
                    "ticker": ticker,
                    "date": date_str,
                    "source": item.get("source", "unknown"),
                    "sentiment": item.get("sentiment", "neutral"),
                    "collection_type": "news",
                })

            for item in prs.get(ticker, [])[:3]:
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                doc = build_news_doc(
                    ticker=ticker,
                    headline=item["title"],
                    sentiment="neutral",
                    source=item.get("source", "SEC"),
                    date=item.get("date", date_str),
                )
                doc_id = f"pr_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
                docs.append(doc)
                ids.append(doc_id)
                metas.append({
                    "ticker": ticker,
                    "date": date_str,
                    "source": item.get("source", "SEC"),
                    "sentiment": "neutral",
                    "pr_type": item.get("type", "general"),
                    "collection_type": "press_release",
                })

        if docs:
            embeddings = self._embed(docs)
            self._cols["news"].add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    def _index_macro(self, data: dict[str, Any], date_str: str) -> None:
        macro = data.get("macro", {})
        if not macro or macro.get("error"):
            return

        doc = build_macro_doc(macro, date_str)
        doc_id = f"macro_{date_str}_{uuid.uuid4().hex[:6]}"
        spread = macro.get("yield_spread", 0) or 0
        embedding = self._embed([doc])
        self._cols["macro"].add(
            documents=[doc],
            embeddings=embedding,
            ids=[doc_id],
            metadatas=[{
                "date": date_str,
                "vix": float(macro.get("vix") or 0),
                "yield_curve": str(macro.get("yield_curve", "normal")),
                "fed_rate": float(macro.get("fed_rate") or 0),
                "cpi_trend": str(macro.get("cpi_trend", "stable")),
                "spread_bps": int(spread * 100) if isinstance(spread, float) else 0,
                "collection_type": "macro",
            }],
        )

    def _index_trades(
        self,
        data: dict[str, Any],
        portfolio: dict[str, Any],
        date_str: str,
    ) -> None:
        tickers = [s["ticker"] for s in portfolio.get("portfolio", [])]
        insider = data.get("insider_trades", {})
        congress = data.get("congress_trades", {})
        docs, ids, metas = [], [], []

        for ticker in tickers:
            it = insider.get(ticker, {})
            for trade in it.get("trades", [])[:5]:
                doc = build_trade_doc(ticker, trade, "insider")
                doc_id = f"insider_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
                docs.append(doc)
                ids.append(doc_id)
                metas.append({
                    "ticker": ticker,
                    "date": trade.get("date", date_str),
                    "trade_type": "insider",
                    "is_csuite": bool(trade.get("is_csuite")),
                    "is_buy": bool(trade.get("is_buy")),
                    "collection_type": "insider_trade",
                })

            ct = congress.get(ticker, [])
            if isinstance(ct, list):
                for trade in ct[:3]:
                    doc = build_trade_doc(ticker, trade, "congress")
                    doc_id = f"congress_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
                    docs.append(doc)
                    ids.append(doc_id)
                    metas.append({
                        "ticker": ticker,
                        "date": trade.get("transaction_date", date_str),
                        "trade_type": "congress",
                        "committee_relevant": bool(trade.get("relevant_committee")),
                        "signal_quality": trade.get("signal_quality", "MEDIUM"),
                        "collection_type": "congress_trade",
                    })

        if docs:
            embeddings = self._embed(docs)
            self._cols["trades"].add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    def _index_context_files(
        self,
        portfolio: dict[str, Any],
        date_str: str,
    ) -> None:
        docs, ids, metas = [], [], []

        for stock in portfolio.get("portfolio", []):
            ticker = stock["ticker"]
            ctx_file = f"context/{ticker}.md"
            if not os.path.exists(ctx_file):
                continue

            with open(ctx_file, "r") as fh:
                content = fh.read()

            # Split on ### section boundaries — each block is one update
            sections = [s.strip() for s in content.split("###") if s.strip()]
            # Only index sections that look like timestamped updates (not headers)
            recent = [s for s in sections if s[:4].isdigit() or s[:2].isdigit()][-5:]

            for section in recent:
                lines = section.split("\n", 1)
                timestamp = lines[0].strip() if lines else date_str
                text = lines[1].strip() if len(lines) > 1 else section

                for chunk in chunk_text(text, max_chars=400):
                    doc = build_context_doc(ticker, timestamp, chunk)
                    doc_id = f"ctx_{ticker}_{date_str}_{uuid.uuid4().hex[:6]}"
                    docs.append(doc)
                    ids.append(doc_id)
                    metas.append({
                        "ticker": ticker,
                        "timestamp": timestamp,
                        "date": date_str,
                        "collection_type": "context",
                    })

        if docs:
            embeddings = self._embed(docs)
            self._cols["context"].add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    # -----------------------------------------------------------------------
    # Retrieval — one method per query type
    # -----------------------------------------------------------------------

    def _query_similar_analyses(
        self,
        ticker: str,
        query_text: str,
        n: int = 3,
    ) -> list[dict[str, Any]]:
        col = self._cols["analyses"]
        if col.count() == 0:
            return []
        embedding = self._embed_query(query_text)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
            where={"ticker": ticker},
        )
        return self._format_hits(results)

    def _query_similar_news(
        self,
        ticker: str,
        query_text: str,
        n: int = 3,
    ) -> list[dict[str, Any]]:
        col = self._cols["news"]
        if col.count() == 0:
            return []
        embedding = self._embed_query(query_text)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
            where={"ticker": ticker},
        )
        return self._format_hits(results)

    def _query_macro_regime(
        self,
        macro_data: dict[str, Any],
        n: int = 2,
    ) -> list[dict[str, Any]]:
        col = self._cols["macro"]
        if col.count() == 0:
            return []
        query_text = build_macro_query(macro_data)
        embedding = self._embed_query(query_text)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
        )
        return self._format_hits(results)

    def _query_trade_patterns(
        self,
        ticker: str,
        data: dict[str, Any],
        n: int = 2,
    ) -> list[dict[str, Any]]:
        col = self._cols["trades"]
        if col.count() == 0:
            return []

        it = data.get("insider_trades", {}).get(ticker, {})
        net = it.get("net_sentiment", "neutral")
        total_sold = it.get("total_sold", 0)
        total_bought = it.get("total_bought", 0)
        query_text = (
            f"{ticker} insider {net} net | "
            f"bought ${total_bought:,.0f} sold ${total_sold:,.0f}"
        )
        embedding = self._embed_query(query_text)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
            where={"ticker": ticker},
        )
        return self._format_hits(results)

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    def _prune_old_entries(self) -> None:
        """Remove entries older than MAX_STORE_AGE_DAYS from all collections."""
        cutoff = (datetime.now() - timedelta(days=MAX_STORE_AGE_DAYS)).strftime("%Y-%m-%d")
        pruned = 0
        for col in self._cols.values():
            try:
                results = col.get(where={"date": {"$lt": cutoff}})
                old_ids = results.get("ids", [])
                if old_ids:
                    col.delete(ids=old_ids)
                    pruned += len(old_ids)
            except Exception:
                pass
        if pruned:
            print(f"  [rag] pruned {pruned} entries older than {MAX_STORE_AGE_DAYS} days")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _extract_ticker_excerpt(self, response: str, ticker: str) -> str:
        """
        Pull the paragraph(s) mentioning a ticker from Claude's response.
        Returns up to 400 chars of the most relevant section.
        """
        lines = response.split("\n")
        collecting = False
        excerpt_lines: list[str] = []

        for line in lines:
            if ticker in line:
                collecting = True
            if collecting:
                excerpt_lines.append(line)
                if len("\n".join(excerpt_lines)) > 400:
                    break
            # Stop at next ticker-like token or section break
            if collecting and line.startswith("==="):
                break

        return "\n".join(excerpt_lines)[:400].strip()

    @staticmethod
    def _format_hits(chroma_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert ChromaDB query results into a flat list of hit dicts."""
        hits: list[dict[str, Any]] = []
        docs = chroma_results.get("documents", [[]])[0]
        metas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Skip very weak matches (distance > 1.2)
            if dist > 1.2:
                continue
            hits.append({
                "summary": doc,
                "date": meta.get("date", "unknown"),
                "ticker": meta.get("ticker", ""),
                "distance": round(dist, 3),
            })
        return hits
