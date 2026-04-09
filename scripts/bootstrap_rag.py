#!/usr/bin/env python3
"""
One-time script to backfill the RAG vector store from existing logs and context files.

Run this locally once before the first GitHub Actions run to seed the store:
    python scripts/bootstrap_rag.py

Requirements:
  - VOYAGE_API_KEY must be set in .env or environment
  - voyageai and chromadb must be installed (pip install voyageai chromadb)
"""

import json
import os
import sys

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from modules.rag_agent import RAGAgent
from modules.rag_utils import build_context_doc, chunk_text

LOGS_FILE = "logs/stock_analysis.jsonl"
CONTEXT_DIR = "context"


def backfill_analyses(rag: RAGAgent) -> int:
    """Index all past analysis runs from logs/stock_analysis.jsonl."""
    if not os.path.exists(LOGS_FILE):
        print(f"  [skip] {LOGS_FILE} not found")
        return 0

    count = 0
    with open(LOGS_FILE) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"  Found {len(entries)} log entries")

    # Build minimal portfolio + data stubs for each entry and index them
    for entry in entries:
        timestamp = entry.get("timestamp", "")
        date_str = timestamp[:10] if timestamp else "unknown"
        response = entry.get("response", "")
        prices_map = entry.get("prices", {})
        pnl_pct = entry.get("pnl_pct", 0)
        weekend = entry.get("weekend", False)

        # Build a minimal data dict
        data = {"prices": {}}
        for ticker, price in prices_map.items():
            data["prices"][ticker] = {"price": price, "pnl_pct": pnl_pct}

        # Build minimal portfolio (just tickers present in the log)
        portfolio = {
            "portfolio": [{"ticker": t} for t in prices_map.keys()],
            "meta": {},
        }

        try:
            rag.index_run(data, response, portfolio, weekend=weekend)
            count += 1
            if count % 10 == 0:
                print(f"    indexed {count}/{len(entries)} runs...")
        except Exception as exc:
            print(f"    [!!] failed entry {date_str}: {exc}")

    return count


def backfill_context_files(rag: RAGAgent) -> int:
    """Index all existing context/{TICKER}.md files."""
    if not os.path.exists(CONTEXT_DIR):
        print(f"  [skip] {CONTEXT_DIR}/ not found")
        return 0

    md_files = [f for f in os.listdir(CONTEXT_DIR) if f.endswith(".md")]
    if not md_files:
        print(f"  [skip] no .md files in {CONTEXT_DIR}/")
        return 0

    print(f"  Found {len(md_files)} context files")
    count = 0
    import uuid

    for filename in md_files:
        ticker = filename.replace(".md", "").upper()
        filepath = os.path.join(CONTEXT_DIR, filename)

        with open(filepath) as f:
            content = f.read()

        sections = [s.strip() for s in content.split("###") if s.strip()]
        updates = [s for s in sections if s[:4].isdigit() or s[:2].isdigit()]

        docs, ids, metas = [], [], []
        for section in updates:
            lines = section.split("\n", 1)
            timestamp = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else section

            for chunk in chunk_text(text, max_chars=400):
                doc = build_context_doc(ticker, timestamp, chunk)
                doc_id = f"ctx_bootstrap_{ticker}_{uuid.uuid4().hex[:8]}"
                docs.append(doc)
                ids.append(doc_id)
                metas.append({
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "date": timestamp[:10] if len(timestamp) >= 10 else "unknown",
                    "collection_type": "context",
                })

        if docs:
            try:
                embeddings = rag._embed(docs)
                rag._cols["context"].add(
                    documents=docs,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metas,
                )
                count += len(docs)
                print(f"    {ticker}: indexed {len(docs)} context chunks")
            except Exception as exc:
                print(f"    [!!] {ticker}: {exc}")

    return count


def main():
    print("=" * 60)
    print("  RAG Bootstrap — seeding vector store from existing data")
    print("=" * 60)

    if not os.environ.get("VOYAGE_API_KEY"):
        print("\n[ERROR] VOYAGE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    rag = RAGAgent(persist_dir="vector_store/")
    if not rag._enabled:
        print("\n[ERROR] RAGAgent failed to initialize. Check your VOYAGE_API_KEY.")
        sys.exit(1)

    print("\n--- Backfilling analysis runs ---")
    n_analyses = backfill_analyses(rag)
    print(f"  Done: {n_analyses} analysis runs indexed")

    print("\n--- Backfilling context files ---")
    n_ctx = backfill_context_files(rag)
    print(f"  Done: {n_ctx} context chunks indexed")

    # Summary
    print("\n--- Vector store summary ---")
    for name, col in rag._cols.items():
        print(f"  {name}: {col.count()} docs")

    print("\n[ok] Bootstrap complete. Commit vector_store/ to git:")
    print("     git add vector_store/ -f")
    print("     git commit -m 'Add bootstrapped RAG vector store'")


if __name__ == "__main__":
    main()
