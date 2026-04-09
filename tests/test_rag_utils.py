"""Tests for rag_utils — document builders and formatters (no API calls needed)."""


from modules.rag_utils import (
    build_analysis_doc,
    build_context_doc,
    build_macro_doc,
    build_news_doc,
    build_ticker_query,
    build_trade_doc,
    chunk_text,
    format_historical_context,
)


class TestChunkText:
    def test_short_text_not_split(self):
        text = "Short sentence."
        chunks = chunk_text(text, max_chars=500)
        assert chunks == ["Short sentence."]

    def test_long_text_split(self):
        sentence = "This is a sentence. "
        text = sentence * 40  # ~800 chars
        chunks = chunk_text(text, max_chars=200, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 200 + 20  # small tolerance for sentence boundaries

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   ") == []

    def test_exact_boundary(self):
        text = "A" * 500
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1


class TestBuildAnalysisDoc:
    def test_basic(self):
        doc = build_analysis_doc("AAPL", "Thesis intact. Price rising.", 185.50, 12.3, "2026-04-09")
        assert "TICKER: AAPL" in doc
        assert "DATE: 2026-04-09" in doc
        assert "$185.50" in doc
        assert "+12.3%" in doc
        assert "Thesis intact" in doc

    def test_none_price(self):
        doc = build_analysis_doc("NVDA", "Some analysis.", None, None, "2026-04-09")
        assert "N/A" in doc

    def test_long_excerpt_truncated(self):
        long_excerpt = "X" * 600
        doc = build_analysis_doc("TSLA", long_excerpt, 200.0, 5.0, "2026-04-09")
        # excerpt gets truncated to 400 chars in build_analysis_doc
        assert len(doc) < 600


class TestBuildNewsDoc:
    def test_basic(self):
        doc = build_news_doc("AAPL", "Apple reports record revenue", "positive", "Yahoo Finance", "2026-04-09")
        assert "TICKER: AAPL" in doc
        assert "Yahoo Finance" in doc
        assert "positive" in doc
        assert "Apple reports record revenue" in doc

    def test_long_headline_truncated(self):
        long_headline = "H" * 300
        doc = build_news_doc("AAPL", long_headline, "neutral", "Reuters", "2026-04-09")
        assert len(doc.split("HEADLINE: ")[1]) <= 200 + 5  # small buffer


class TestBuildMacroDoc:
    def test_basic(self):
        macro = {
            "vix": 22.5,
            "treasury_10y": 4.5,
            "treasury_2y": 4.9,
            "yield_curve": "inverted",
            "yield_spread": -0.004,
            "cpi": 3.1,
            "fed_rate": 5.25,
            "oil_wti": 82.0,
            "dxy": 104.5,
        }
        doc = build_macro_doc(macro, "2026-04-09")
        assert "VIX: 22.5" in doc
        assert "inverted" in doc
        assert "2026-04-09" in doc
        assert "5.25%" in doc

    def test_missing_fields(self):
        doc = build_macro_doc({}, "2026-01-01")
        assert "N/A" in doc


class TestBuildTradeDoc:
    def test_insider_buy(self):
        trade = {
            "insider_name": "Tim Cook",
            "insider_title": "CEO",
            "is_buy": True,
            "shares": 10000,
            "value": 1850000,
            "date": "2026-04-01",
            "is_csuite": True,
        }
        doc = build_trade_doc("AAPL", trade, "insider")
        assert "Tim Cook" in doc
        assert "BOUGHT" in doc
        assert "[C-SUITE]" in doc
        assert "1,850,000" in doc

    def test_insider_sell(self):
        trade = {
            "insider_name": "Satya Nadella",
            "insider_title": "CEO",
            "is_buy": False,
            "shares": 50000,
            "value": 20750000,
            "date": "2026-03-15",
            "is_csuite": True,
        }
        doc = build_trade_doc("MSFT", trade, "insider")
        assert "SOLD" in doc
        assert "Satya Nadella" in doc

    def test_congress_trade(self):
        trade = {
            "name": "Jane Smith",
            "chamber": "House",
            "committee": "Science Committee",
            "transaction": "Purchase",
            "amount_range": "$50,001-$100,000",
            "transaction_date": "2026-02-10",
            "relevant_committee": True,
            "signal_quality": "HIGH",
        }
        doc = build_trade_doc("NVDA", trade, "congress")
        assert "Jane Smith" in doc
        assert "HIGH" in doc
        assert "Science Committee" in doc


class TestBuildContextDoc:
    def test_basic(self):
        doc = build_context_doc("AAPL", "2026-04-01 10:00", "Thesis update: growth intact.")
        assert "TICKER: AAPL" in doc
        assert "2026-04-01 10:00" in doc
        assert "Thesis update" in doc


class TestBuildTickerQuery:
    def test_with_all_fields(self):
        q = build_ticker_query("AAPL", 185.0, -2.5, ["Apple reports miss", "China concerns"])
        assert "AAPL" in q
        assert "$185.00" in q
        assert "-2.5%" in q
        assert "Apple reports miss" in q

    def test_no_price(self):
        q = build_ticker_query("NVDA", None, None, [])
        assert "NVDA" in q
        assert "N/A" in q


class TestFormatHistoricalContext:
    def test_basic_output(self):
        ticker_results = {
            "AAPL": [
                {"date": "2025-11-14", "summary": "AAPL price dropped after China warning → THESIS SHAKEN", "distance": 0.2},
                {"date": "2025-08-02", "summary": "AAPL C-suite sold $18M → No thesis break", "distance": 0.35},
            ]
        }
        macro_results = [
            {"date": "2025-08-05", "summary": "VIX 28 inverted curve Fed held rates market recovered", "distance": 0.3}
        ]
        block = format_historical_context(ticker_results, macro_results)
        assert "AAPL" in block
        assert "2025-11-14" in block
        assert "2025-08-05" in block
        assert "MACRO" in block

    def test_empty_inputs(self):
        block = format_historical_context({}, [])
        assert block == ""

    def test_empty_ticker_hits(self):
        block = format_historical_context({"AAPL": []}, [])
        assert block == ""

    def test_caps_at_two_per_ticker(self):
        hits = [
            {"date": f"2025-0{i}-01", "summary": f"Hit {i}", "distance": 0.1 * i}
            for i in range(1, 6)
        ]
        block = format_historical_context({"NVDA": hits}, [])
        # Should only show 2 parallels
        assert block.count("[2025-") == 2
