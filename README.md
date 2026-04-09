# markit-engine

An AI-powered portfolio analysis engine that monitors 14 data layers in parallel and uses Claude to evaluate your investment theses in real time.

```
python analyzer.py --once --test
```

![UI](https://img.shields.io/badge/UI-Streamlit-black?style=flat-square)
![Anthropic SDK](https://img.shields.io/badge/AI-Claude%20Sonnet-black?style=flat-square)
 
---

## What it does

You define your portfolio and investment thesis in `stocks.yaml`. The engine runs every hour during market hours, pulls data from 14 sources, and asks Claude a single question: *is your thesis still intact?*

For each stock it produces:
- **THESIS STATUS**: INTACT / SHAKEN / BROKEN — with the specific signal that drove the call
- **ACTION**: Keep / Buy More / Sell — with reasoning
- **Alerts**: flags urgent developments and logs them to `logs/notifications.md`
- **Discoveries**: suggests new stocks fitting your investment goal
- **Context log**: appends per-ticker narrative evolution over time

---

## Data layers

| # | Layer | Source | Weekday | Weekend |
|---|-------|--------|---------|---------|
| 1 | Price + P&L | Finnhub | ✓ | — |
| 2 | Fundamentals | Finnhub | ✓ | — |
| 3 | Technicals (RSI, MACD, BB) | Calculated | ✓ | — |
| 4 | Macro (Fed, CPI, VIX, DXY) | FRED + Yahoo | ✓ | — |
| 5 | News RSS | Google News | ✓ | ✓ |
| 6 | Press releases & 8-K | SEC EDGAR | ✓ | ✓ |
| 7 | World news & geopolitics | GDELT | ✓ | ✓ |
| 8 | Google Trends | pytrends | ✓ | ✓ |
| 9 | Wikipedia page views | Wikimedia API | ✓ | ✓ |
| 10 | Hedge fund 13F filings | Finnhub | ✓ | — |
| 11 | Insider trades (Form 4) | Finnhub | ✓ | — |
| 12 | Congress trades | quiverquant | ✓ | ✓ |
| 13 | Sustainability / ESG | Derived | ✓ | ✓ |
| 14 | Historical patterns (RAG) | Vector store | ✓ | ✓ |

All layers run in parallel via `ThreadPoolExecutor`. A layer that errors or times out is skipped — the rest still run.

---

## Dashboard

```
streamlit run app.py
```

A minimal black-and-white Streamlit dashboard with four tabs:

- **Overview** — portfolio banner, P&L history chart, per-stock metrics + sparklines, watchlist
- **Analysis** — Claude's full response with INTACT/SHAKEN/BROKEN syntax highlighting
- **Signals** — per-ticker news, technicals, insider/congress activity, press releases
- **Macro** — Fed rates, inflation, VIX, GDELT global sentiment

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/avaishcodescape/markit-engine
cd markit-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your portfolio

Edit `stocks.yaml` — this is the only file you need to change:

```yaml
meta:
  investor_goal: "Long-term growth focused on AI infrastructure"
  risk_profile: "Moderate — hold through volatility, not through thesis breaks"

portfolio:
  - ticker: NVDA
    name: NVIDIA Corporation
    role: growth-conviction
    target_multiple: 40
    thesis: >
      Dominant AI accelerator platform with 80%+ data center GPU share.
      CUDA moat makes switching costly.
    thesis_risks:
      - Customer concentration in hyperscalers
      - Export controls on China
    purchases:
      - date: "2025-01-15"
        dollars: 5000
        price_per_share: 800.00

watchlist:
  - ticker: AMZN
    reason: "AWS reacceleration + advertising margin expansion"
```

### 3. Set environment variables

Create a `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=...
VOYAGE_API_KEY=...          # optional — needed for RAG layer
```

**Required**: `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`  
**Optional**: `VOYAGE_API_KEY` (RAG historical pattern matching)

Free tiers: [Finnhub](https://finnhub.io) — 60 req/min free. [Voyage AI](https://www.voyageai.com) — free tier available.

### 4. Run

```bash
# Run once and print output (no side effects)
python analyzer.py --once --test

# Run once for real (logs to logs/)
python analyzer.py --once

# Run on a loop every 60 minutes
python analyzer.py --loop

# Launch the dashboard
streamlit run app.py
```

---

## GitHub Actions (automated scheduling)

The workflow in `.github/workflows/analyze.yml` runs the analyzer hourly on weekdays and once Sunday evening. It commits updated logs and the vector store back to the repo.

Add these secrets to your repo (`Settings > Secrets > Actions`):

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `FINNHUB_API_KEY` | Market data |
| `VOYAGE_API_KEY` | RAG embeddings (optional) |

---

## Project structure

```
markit-engine/
├── analyzer.py           # Main orchestrator — runs layers, calls Claude, logs
├── app.py                # Streamlit dashboard
├── stocks.yaml           # Your portfolio config (the only file to edit)
├── modules/
│   ├── prices.py         # Layer 1 — price + P&L calculation
│   ├── fundamentals.py   # Layer 2 — PE, margins, analyst consensus
│   ├── technicals.py     # Layer 3 — RSI, MACD, Bollinger Bands
│   ├── macro.py          # Layer 4 — Fed rate, CPI, VIX
│   ├── news_rss.py       # Layer 5 — Google News RSS
│   ├── press_releases.py # Layer 6 — SEC EDGAR 8-K
│   ├── world_news.py     # Layer 7 — GDELT geopolitics
│   ├── google_trends.py  # Layer 8 — search interest
│   ├── wikipedia.py      # Layer 9 — page view spikes
│   ├── hedge_funds.py    # Layer 10 — 13F filings
│   ├── insider_trades.py # Layer 11 — Form 4
│   ├── congress_trades.py # Layer 12 — quiverquant
│   ├── sustainability.py # Layer 13 — ESG signal extraction
│   ├── alerts.py         # Notification logging
│   ├── rag_agent.py      # Layer 14 — historical pattern retrieval
│   └── rag_utils.py      # Vector store helpers
├── tests/                # pytest suite
├── logs/                 # Run history
├── context/              # Per-ticker living context logs
└── vector_store/         # ChromaDB embeddings
```

---

## How the prompt works

All 14 data layers are assembled into a single structured prompt sent to Claude Sonnet. Claude is asked to:

1. For each stock: THESIS STATUS + ACTION + the specific layer that drove the call
2. Flag an **URGENT ALERT** if needed (logged to `logs/notifications.md`)
3. Identify the biggest risk and opportunity
4. Evaluate each watchlist stock for entry
5. Suggest new discoveries fitting your investment goal
6. Flag material ESG / governance red flags

Claude also writes `CONTEXT_UPDATE: TICKER: [text]` lines which get appended to per-ticker markdown files in `context/` — building a living narrative of thesis evolution over time.

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Run tests: `pytest tests/ -v`
4. Run lint: `ruff check .`
5. Open a PR

New data layers are the most welcome contribution — see any existing module in `modules/` for the pattern. Each layer takes a list of tickers and returns a dict keyed by ticker.

---

## License

MIT
