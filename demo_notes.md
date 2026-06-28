# 🌤 PolyWeather: Autonomous Trading Agent
**A Resilient, Agentic Weather Prediction Market Trader**

## 1. Multi-Layer API Resilience
Watch the terminal for simulated API failures. The system is built with automatic failovers:
* **Weather Data:** When Apify API fails → Fails over to free Open-Meteo data.
* **Market Data:** When Polymarket API is blocked → Auto-generates simulated markets to keep the pipeline alive.

## 2. LLM Outage & Math Engine
* OpenRouter free-tier LLMs frequently return 404s.
* **Graceful Degradation:** When the AI fails, the system doesn't crash. It falls back to pure quantitative math (calculating Z-scores and fractional Kelly Criterion edges).

## 3. Autonomous Execution
* The system evaluates all 5 cities.
* It places 6 calculated paper trades based strictly on positive expected value (EV).

## 4. The UI Dashboard
* Switching to the Streamlit UI (`localhost:8501`).
* **Live Matrices:** Displaying current weather vs. model prediction probabilities side-by-side.
* **Real-time Tracking:** 6 open positions logged securely in SQLite.

## 5. Algorithmic Risk Management (The Circuit Breaker)
* The agent has a hard-coded maximum total exposure limit of 50% ($5,000).
* **Watch the terminal:** The system realizes it hit the $5k limit and autonomously blocks itself from placing further trades.