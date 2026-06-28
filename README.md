# 🌤️ PolyWeather — Autonomous Weather Prediction Market Trading Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Storage-07405e?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Paper%20Trading%20Demo-orange)

> An autonomous agent that monitors weather prediction markets, generates probabilistic forecasts, and paper-trades positions based on calculated statistical edge — built with multi-layer API failover so the system never goes down even when external services do.

---

## 🎥 Demo

A short walkthrough of the system in action — API failover, autonomous trade execution, the live dashboard, and the risk circuit breaker tripping.

📹 [**Watch the demo**](./assets/demo.webm)

> GitHub doesn't autoplay video files inline in the README — clicking the link above will open/download `demo.webm` for playback. If you'd like it to play directly on the repo page, drag-and-drop the file into a GitHub issue or PR comment box first; GitHub will host it and give you a `https://github.com/user-attachments/...` URL that *does* embed and autoplay in Markdown — swap that URL in above once you have it.

---

## 🧠 What This Project Does

PolyWeather is a self-running trading agent for **weather-outcome prediction markets** (e.g. Polymarket-style "will it rain in Tokyo tomorrow?" markets). On a fixed schedule, it:

1. Pulls live weather data for 5 global cities (New York, London, Tokyo, Mumbai, Sydney)
2. Generates probability forecasts for weather events (temperature thresholds, precipitation, range buckets) using historical climate normals + live conditions
3. Compares those probabilities against market-implied prices to calculate statistical edge
4. Sizes and places paper trades using a fractional Kelly Criterion, respecting hard risk limits
5. Tracks every trade, position, and resulting P&L in a local SQLite database
6. Surfaces everything in a live Streamlit dashboard and optional Telegram alerts

The system is intentionally built to **degrade gracefully** rather than crash — every external dependency (weather scraper, market API, LLM) has a fallback path.

> ⚠️ **This is a paper-trading / educational simulation.** No real funds are deployed by this code. It is not financial advice, and is not intended for use with real capital without significant further review.

---

## 🔁 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    main.py (CLI entry point)                 │
│              `run` (single cycle) | `daemon` (loop)          │
└───────────────────────────┬────────────────────────────────-─┘
                             ▼
                  hermes_orchestrator.py
              (coordinates the full trading cycle)
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  data_engine.py      market_data.py       prediction_model.py
  Weather fetch        Market fetch         Probability model
  (Apify → Open-Meteo  (Polymarket →        (historical avgs +
   fallback)            simulated fallback)   live data + LLM)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                     risk_manager.py
            (Kelly sizing, exposure limits, hedging)
                             ▼
                   execution_engine.py
                (PaperTrader — SQLite-backed)
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
        trades.db                   telegram_bot.py
     (positions, PnL log)          (trade alerts, summaries)
              │
              ▼
            app.py
   (Streamlit dashboard — 4 tabs)
```

---

## ✨ Key Features

### 🛡️ Multi-Layer API Resilience
- **Weather data:** Apify scraper → automatic fallback to free Open-Meteo API on failure
- **Market data:** Polymarket Gamma/CLOB API → falls back to simulated markets if blocked, keeping the pipeline alive for demo/testing
- **LLM reasoning:** Cycles through multiple free OpenRouter models; if all fail, falls back to pure quantitative scoring (Z-scores + historical climate normals)

### 📊 Quantitative + LLM Hybrid Forecasting
- Probability estimates blend live weather data, NOAA/WMO 1991–2020 climate normals, and optional LLM-assisted reasoning
- Designed so the system keeps producing valid trade signals even with zero working AI calls

### ⚖️ Algorithmic Risk Management
- Half-Kelly position sizing (`KELLY_FRACTION_CAP = 0.5`) for conservative bet sizing
- Hard caps: max 10% of bankroll per position, max 50% total exposure
- Built-in "circuit breaker" — the agent autonomously blocks new trades once exposure limits are hit
- Hedging suggestions generated for at-risk open positions

### 📈 Live Dashboard (Streamlit)
Four tabs covering:
- **Active Positions** — current open trades and exposure
- **Trade History & PnL** — cumulative equity curve + full trade log
- **Weather Predictions** — live weather vs. model probability matrix, per city
- **Risk Profiles** — exposure distribution, safety parameters, hedge recommendations

### 🔔 Telegram Notifications (optional)
Trade execution alerts, daily portfolio summaries, and risk warnings — silently no-ops if not configured, so the rest of the system never needs to check whether notifications are enabled.

---

## 📁 Project Structure

```
polyweather/
│
├── main.py                    # CLI entry point (run / daemon commands)
├── hermes_orchestrator.py     # Orchestrates the full trading cycle
├── data_engine.py             # Weather data fetching + failover
├── market_data.py             # Polymarket data fetching + simulated fallback
├── prediction_model.py        # Probability forecasting model
├── risk_manager.py            # Kelly sizing, exposure limits, hedging logic
├── execution_engine.py        # PaperTrader — simulated order execution
├── telegram_bot.py            # Telegram alert integration
├── config.py                  # Central configuration (cities, params, API keys)
├── app.py                     # Streamlit dashboard
│
├── data/
│   └── historical_averages.json   # NOAA/WMO climate normals (5 cities)
│
├── assets/
│   └── demo.webm               # Demo walkthrough video
│
├── run.sh                     # Convenience launcher (venv + dashboard + daemon)
├── requirements.txt
├── .env.example                # Template — copy to .env and fill in your keys
├── .gitignore
├── LICENSE
└── README.md
```

> 📌 **Note:** `config.py` expects `historical_averages.json` at `data/historical_averages.json`. Make sure the file lives in a `data/` subfolder before running.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sunilraj180805/polyweather.git
cd polyweather
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```
Then open `.env` and fill in any keys you have. All keys are optional — the system runs in fully-degraded (free-tier / simulated) mode with no keys configured at all:

| Variable | Required? | Fallback if missing |
|---|---|---|
| `APIFY_API_TOKEN` | No | Open-Meteo (free) weather data |
| `OPENROUTER_API_KEY` | No | Pure quantitative scoring (no LLM) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | No | Notifications silently disabled |
| `INITIAL_BALANCE` | No | Defaults to `10000` |
| `AGENT_CYCLE_MINUTES` | No | Defaults to `30` |

### 5. Run It

**Single trading cycle (good for testing):**
```bash
python main.py run
```

**Continuous autonomous daemon:**
```bash
python main.py daemon
```

**Streamlit dashboard:**
```bash
streamlit run app.py
```
Then open `http://localhost:8501`.

**Or launch dashboard + daemon together:**
```bash
chmod +x run.sh
./run.sh
```
> ⚠️ `run.sh` currently activates a venv from a hardcoded absolute path. Before sharing or running on another machine, update the `VENV_PATH` variable near the top of the script to point at your local `venv/` directory (or simplify it to `source venv/bin/activate` for a relative path).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Live dashboard UI |
| Plotly | Interactive equity & exposure charts |
| SQLite | Trade & position storage |
| Pandas / NumPy | Data manipulation, Z-score & statistical calculations |
| Apify / Open-Meteo | Weather data sourcing |
| OpenRouter (LLM) | Optional AI-assisted reasoning layer |
| python-telegram-bot | Trade alerts & summaries |
| python-dotenv | Environment variable management |
| schedule | Periodic daemon scheduling |

---

## ⚠️ Disclaimer

This project is a **paper-trading simulation built for educational and demonstration purposes**. It does not place real trades or move real money. Trading involves substantial risk of loss, and nothing in this repository constitutes financial advice. Use of any concepts here in a real trading context is done entirely at your own risk.

---

## 🔮 Future Scope

- Persist trade history to a proper database (PostgreSQL) for multi-instance deployments
- Backtesting harness against historical market resolution data
- Additional weather event types (wind speed, humidity-based markets)
- Web-based config UI instead of `.env` editing

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Sunilraj D**
[GitHub](https://github.com/sunilraj180805) · [LinkedIn](https://www.linkedin.com/in/sunilraj18)
