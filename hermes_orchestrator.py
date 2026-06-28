"""
PolyWeather — Hermes-Style Autonomous Trading Orchestrator

Lightweight autonomous agent loop that calls OpenRouter via the
OpenAI-compatible SDK.  Each cycle:

  1. Fetches live weather data for all 5 cities
  2. Generates predictions (probabilities) for relevant events
  3. Retrieves current Polymarket odds (live or simulated)
  4. Identifies edge: our_probability − market_price
  5. Sends the divergence metrics to the LLM for qualitative analysis
  6. Runs Kelly Criterion to size the position
  7. Executes paper trades where the edge exceeds the threshold
  8. Fires Telegram alerts

The LLM acts as a meta-analyst that reviews the data pipeline's output
and provides a reasoned GO / NO-GO recommendation.  It does NOT make
the probability calculation — that is purely statistical.

Usage:
    from hermes_orchestrator import HermesOrchestrator
    agent = HermesOrchestrator()
    report = agent.run_cycle()         # single cycle
    agent.run_daemon()                 # periodic loop
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import config
from data_engine import WeatherDataEngine
from prediction_model import WeatherPredictor
from market_data import PolymarketFetcher
from risk_manager import RiskManager
from execution_engine import PaperTrader
from telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# System prompt for the LLM analyst
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are PolyWeather's AI Trading Analyst. You review weather prediction data
and market pricing to make trading recommendations on Polymarket weather markets.

Your role is ADVISORY. The statistical model has already computed probabilities.
You provide a qualitative "second opinion" and flag anything the model might miss.

When given a trade opportunity, respond with a JSON object:
{
  "recommendation": "GO" or "NO_GO",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation (2-3 sentences max)",
  "risk_notes": "Any concerns about the trade"
}

Guidelines:
- If the statistical edge is > 10% and weather data supports it, recommend GO.
- If the edge is 5-10%, recommend GO only if weather patterns strongly support it.
- Flag seasonal anomalies, extreme weather events, or data quality concerns.
- Be decisive. Avoid hedge language. State your recommendation clearly.
- Keep responses under 100 words total.
"""


# ---------------------------------------------------------------------------
# HermesOrchestrator
# ---------------------------------------------------------------------------

class HermesOrchestrator:
    """Autonomous trading agent that ties all modules together."""

    def __init__(self) -> None:
        # Core modules
        self._engine = WeatherDataEngine()
        self._predictor = WeatherPredictor(self._engine)
        self._market = PolymarketFetcher()
        self._risk = RiskManager()
        self._trader = PaperTrader()
        self._notifier = TelegramNotifier()

        # LLM client (OpenRouter via OpenAI SDK)
        self._llm_client = None
        self._llm_available = False
        self._init_llm()

        # Cycle tracking
        self._cycle_count = 0
        self._total_trades = 0

        logger.info("HermesOrchestrator initialised")

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        """Set up the OpenAI client pointed at OpenRouter."""
        if not config.OPENROUTER_API_KEY:
            logger.warning(
                "OPENROUTER_API_KEY not set — LLM analysis disabled. "
                "Trades will proceed on statistical signals only."
            )
            return

        try:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=config.OPENROUTER_API_KEY,
            )
            self._llm_available = True
            logger.info("LLM client connected to OpenRouter")
        except ImportError:
            logger.warning("openai package not installed — LLM analysis disabled")
        except Exception as exc:
            logger.warning("LLM client init failed: %s", exc)

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _llm_analyze_trade(
        self,
        city_name: str,
        event_type: str,
        our_prob: float,
        market_price: float,
        edge: float,
        weather_summary: str,
        forecast_summary: str,
    ) -> dict:
        """Ask the LLM to review a potential trade opportunity.

        Returns a parsed dict with recommendation, confidence, reasoning.
        Falls back to auto-GO if LLM is unavailable.
        """
        if not self._llm_available or self._llm_client is None:
            return {
                "recommendation": "GO",
                "confidence": 0.7,
                "reasoning": "LLM unavailable — proceeding on statistical signal",
                "risk_notes": "No qualitative validation",
            }

        user_prompt = (
            f"Trade Opportunity Analysis:\n"
            f"- City: {city_name}\n"
            f"- Event: {event_type}\n"
            f"- Our probability: {our_prob:.1%}\n"
            f"- Market price: {market_price:.1%}\n"
            f"- Edge: {edge:+.1%}\n"
            f"- Current weather: {weather_summary}\n"
            f"- Forecast: {forecast_summary}\n\n"
            f"Should we take this trade? Respond with JSON only."
        )

        # Try each model in priority order
        for model in config.LLM_MODELS:
            try:
                response = self._llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                    timeout=config.LLM_REQUEST_TIMEOUT,
                )

                content = response.choices[0].message.content.strip()
                logger.debug("LLM response (%s): %s", model, content[:200])

                # Parse JSON from response (handle markdown code blocks)
                json_str = content
                if "```" in json_str:
                    # Extract JSON from code block
                    parts = json_str.split("```")
                    for part in parts:
                        cleaned = part.strip()
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()
                        if cleaned.startswith("{"):
                            json_str = cleaned
                            break

                # Try to find JSON object in the text
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

                result = json.loads(json_str)

                # Validate required fields
                if "recommendation" not in result:
                    result["recommendation"] = "GO" if edge > 0.05 else "NO_GO"
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "reasoning" not in result:
                    result["reasoning"] = content[:200]
                if "risk_notes" not in result:
                    result["risk_notes"] = ""

                result["model_used"] = model
                return result

            except json.JSONDecodeError:
                logger.debug("LLM response not valid JSON from %s, trying next", model)
                return {
                    "recommendation": "GO" if edge > 0.05 else "NO_GO",
                    "confidence": 0.5,
                    "reasoning": content[:200] if 'content' in dir() else "Parse error",
                    "risk_notes": "LLM response was not parseable JSON",
                    "model_used": model,
                }
            except Exception as exc:
                logger.warning("LLM call failed with %s: %s", model, exc)
                continue

        # All models failed
        return {
            "recommendation": "GO",
            "confidence": 0.5,
            "reasoning": "All LLM models failed — proceeding on statistical signal",
            "risk_notes": "No LLM validation available",
        }

    # ------------------------------------------------------------------
    # Core trading cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Execute one complete trading cycle across all cities.

        Returns a report dict summarising the cycle's activity.
        """
        self._cycle_count += 1
        cycle_start = time.time()
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(
            "═══ CYCLE %d START ═══ Target date: %s",
            self._cycle_count, tomorrow,
        )

        report = {
            "cycle": self._cycle_count,
            "timestamp": now.isoformat(),
            "target_date": tomorrow,
            "cities_processed": 0,
            "opportunities_found": 0,
            "trades_placed": 0,
            "trades_skipped": 0,
            "errors": [],
            "trade_details": [],
        }

        # --- Step 1: Fetch weather data ---
        logger.info("Step 1/5: Fetching weather data for all cities...")
        try:
            weather_data = self._engine.fetch_all_cities()
        except Exception as exc:
            logger.error("Weather data fetch failed: %s", exc)
            report["errors"].append(f"Weather fetch: {exc}")
            return report

        # --- Step 2: Fetch market odds ---
        logger.info("Step 2/5: Fetching market odds...")
        try:
            all_markets = self._market.fetch_weather_markets()
        except Exception as exc:
            logger.error("Market data fetch failed: %s", exc)
            report["errors"].append(f"Market fetch: {exc}")
            return report

        # Get current portfolio state for risk checks
        portfolio = self._trader.get_portfolio()
        current_balance = portfolio["balance"]
        open_positions = self._trader.get_open_positions()
        total_exposure = sum(p.get("amount", 0) for p in open_positions)

        self._risk.update_bankroll(current_balance)

        # --- Step 3-5: Process each city ---
        for city_key, city_cfg in config.CITIES.items():
            city_name = city_cfg["display_name"]
            city_weather = weather_data.get(city_key, {})

            if city_weather.get("error"):
                logger.warning("Skipping %s — data error: %s",
                             city_name, city_weather["error"])
                report["errors"].append(f"{city_name}: {city_weather['error']}")
                continue

            report["cities_processed"] += 1
            current = city_weather.get("current", {})
            forecast = city_weather.get("forecast", [])

            # Build weather summary for LLM
            weather_summary = (
                f"{current.get('temp_f', '?')}°F, "
                f"{current.get('condition', '?')}, "
                f"humidity {current.get('humidity_pct', '?')}%, "
                f"wind {current.get('wind_kph', '?')} kph"
            )

            # Get tomorrow's forecast
            forecast_tomorrow = None
            for f in forecast:
                if f.get("date") == tomorrow:
                    forecast_tomorrow = f
                    break

            forecast_summary = "No forecast available"
            if forecast_tomorrow:
                forecast_summary = (
                    f"{forecast_tomorrow.get('temp_min_f', '?')}-"
                    f"{forecast_tomorrow.get('temp_max_f', '?')}°F, "
                    f"{forecast_tomorrow.get('condition', '?')}, "
                    f"precip {forecast_tomorrow.get('precip_mm', '?')}mm "
                    f"({forecast_tomorrow.get('precip_prob_pct', '?')}%)"
                )

            logger.info("  Processing %s: %s", city_name, weather_summary)

            # Get markets for this city
            city_markets = [m for m in all_markets if m.get("city") == city_key]

            # Also generate simulated markets for events without live ones
            for event_key in config.WEATHER_EVENTS:
                if event_key == "temp_range_bucket":
                    continue

                # Check if we already have a market for this event/date
                has_market = any(
                    m["event_type"] == event_key
                    and m.get("end_date", "").startswith(tomorrow)
                    for m in city_markets
                )
                if not has_market:
                    sim = self._market.simulate_market_odds(
                        city_key, event_key, tomorrow
                    )
                    city_markets.append(sim)

            # --- Evaluate each market ---
            for market in city_markets:
                if not market.get("end_date", "").startswith(tomorrow):
                    continue

                event_type = market.get("event_type", "")
                market_price = market.get("yes_price", 0.5)
                market_id = market.get("market_id", "")

                if not event_type or not market_id:
                    continue

                # Step 3: Generate prediction
                try:
                    prediction = self._get_prediction(
                        city_key, event_type, tomorrow
                    )
                    if prediction is None:
                        continue
                    our_prob = prediction.probability
                except Exception as exc:
                    logger.debug("Prediction failed for %s/%s: %s",
                               city_key, event_type, exc)
                    continue

                # Step 4: Evaluate edge
                edge_info = self._risk.evaluate_edge(our_prob, market_price)

                if not edge_info["tradeable"]:
                    continue

                report["opportunities_found"] += 1

                # Step 5: LLM analysis (optional)
                llm_result = self._llm_analyze_trade(
                    city_name=city_name,
                    event_type=event_type,
                    our_prob=our_prob,
                    market_price=market_price,
                    edge=edge_info["edge"],
                    weather_summary=weather_summary,
                    forecast_summary=forecast_summary,
                )

                # Decision: only skip if LLM says NO_GO with high confidence
                llm_vetoed = (
                    llm_result.get("recommendation") == "NO_GO"
                    and llm_result.get("confidence", 0) > 0.7
                )

                if llm_vetoed:
                    logger.info(
                        "  ⛔ %s/%s: LLM vetoed (confidence=%.2f: %s)",
                        city_name, event_type,
                        llm_result.get("confidence", 0),
                        llm_result.get("reasoning", "")[:80],
                    )
                    report["trades_skipped"] += 1
                    continue

                # Step 6: Calculate position size
                position = self._risk.calculate_position(
                    win_prob=our_prob,
                    market_price=market_price,
                    bankroll=current_balance,
                    existing_exposure=total_exposure,
                )

                if not position.tradeable:
                    logger.debug(
                        "  ⏭ %s/%s: Risk manager rejected — %s",
                        city_name, event_type, position.reject_reason,
                    )
                    report["trades_skipped"] += 1
                    continue

                # Step 7: Execute trade
                try:
                    trade = self._trader.place_order(
                        market_id=market_id,
                        city=city_key,
                        event_type=event_type,
                        side=position.side,
                        amount=position.position_size,
                        order_type="MARKET",
                        price=market_price if position.side == "YES" else (1 - market_price),
                        question=market.get("question", ""),
                        notes=json.dumps({
                            "our_prob": our_prob,
                            "market_price": market_price,
                            "edge": edge_info["edge"],
                            "kelly": position.kelly_fraction,
                            "llm": llm_result.get("recommendation", "N/A"),
                        }),
                    )

                    report["trades_placed"] += 1
                    self._total_trades += 1
                    total_exposure += position.position_size

                    # Record market snapshot
                    self._trader.record_market_snapshot(
                        market_id=market_id,
                        yes_price=market.get("yes_price", 0.5),
                        no_price=market.get("no_price", 0.5),
                        source=market.get("source", ""),
                    )

                    trade_detail = {
                        "city": city_name,
                        "event": event_type,
                        "side": position.side,
                        "amount": position.position_size,
                        "price": trade["price"],
                        "our_prob": our_prob,
                        "market_price": market_price,
                        "edge": round(edge_info["edge"], 4),
                        "kelly": position.kelly_fraction,
                        "llm_recommendation": llm_result.get("recommendation"),
                        "llm_reasoning": llm_result.get("reasoning", "")[:100],
                    }
                    report["trade_details"].append(trade_detail)

                    logger.info(
                        "  ✅ %s %s $%.2f on %s/%s "
                        "(edge=%.1f%%, kelly=%.4f, prob=%.1f%% vs mkt=%.1f%%)",
                        position.side, "BOUGHT", position.position_size,
                        city_name, event_type,
                        edge_info["edge_pct"] * 100,
                        position.kelly_fraction,
                        our_prob * 100, market_price * 100,
                    )

                    # Step 8: Telegram alert
                    self._notifier.send_trade_alert(trade)

                except Exception as exc:
                    logger.error("Trade execution failed: %s", exc)
                    report["errors"].append(f"Trade exec {city_name}/{event_type}: {exc}")

        # --- Risk check after all trades ---
        updated_positions = self._trader.get_open_positions()
        risk_report = self._risk.check_portfolio_risk(updated_positions)

        if not risk_report.within_limits:
            for warning in risk_report.warnings:
                logger.warning("⚠ Risk: %s", warning)
                self._notifier.send_risk_alert(warning)

        # --- Cycle summary ---
        elapsed = time.time() - cycle_start
        report["elapsed_seconds"] = round(elapsed, 2)
        report["final_balance"] = self._trader.get_balance()

        logger.info(
            "═══ CYCLE %d COMPLETE ═══ "
            "Cities=%d | Opportunities=%d | Trades=%d | Skipped=%d | "
            "Balance=$%.2f | Time=%.1fs",
            self._cycle_count,
            report["cities_processed"],
            report["opportunities_found"],
            report["trades_placed"],
            report["trades_skipped"],
            report["final_balance"],
            elapsed,
        )

        return report

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    def _get_prediction(
        self,
        city_key: str,
        event_type: str,
        target_date: str,
    ):
        """Route to the appropriate prediction method."""
        event_cfg = config.WEATHER_EVENTS.get(event_type, {})
        direction = event_cfg.get("direction", "")
        threshold_f = event_cfg.get("threshold_f")

        if direction == "above" and threshold_f is not None:
            return self._predictor.predict_temperature_above(
                city_key, threshold_f, target_date
            )
        elif direction == "below" and threshold_f is not None:
            return self._predictor.predict_temperature_below(
                city_key, threshold_f, target_date
            )
        elif event_type == "precipitation":
            return self._predictor.predict_precipitation(
                city_key, target_date
            )

        return None

    # ------------------------------------------------------------------
    # Daemon mode
    # ------------------------------------------------------------------

    def run_daemon(self, interval_minutes: int | None = None) -> None:
        """Run the agent in a persistent loop.

        Executes a cycle every *interval_minutes* (default from config).
        Sends a daily summary at the end of each cycle.
        """
        interval = interval_minutes or config.AGENT_CYCLE_MINUTES

        logger.info(
            "🚀 PolyWeather daemon starting (interval=%d min)", interval
        )

        try:
            import schedule

            def _cycle_job():
                try:
                    report = self.run_cycle()

                    # Send daily summary every cycle
                    portfolio = self._trader.get_portfolio()
                    pnl = self._trader.get_pnl_summary()
                    self._notifier.send_daily_summary(portfolio, pnl)

                except Exception as exc:
                    logger.error("Cycle failed: %s\n%s", exc, traceback.format_exc())

            # Run immediately, then schedule
            _cycle_job()

            schedule.every(interval).minutes.do(_cycle_job)

            while True:
                schedule.run_pending()
                time.sleep(10)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
        finally:
            self._trader.close()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current agent status."""
        return {
            "cycles_completed": self._cycle_count,
            "total_trades": self._total_trades,
            "llm_available": self._llm_available,
            "telegram_enabled": self._notifier.is_enabled,
            "balance": self._trader.get_balance(),
            "data_source": self._engine.get_data_summary()["primary_source"],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("PolyWeather Hermes Orchestrator — Single Cycle Test")
    print("=" * 65)

    agent = HermesOrchestrator()

    status = agent.get_status()
    print(f"\n  LLM available: {status['llm_available']}")
    print(f"  Telegram:      {status['telegram_enabled']}")
    print(f"  Data source:   {status['data_source']}")
    print(f"  Balance:       ${status['balance']:.2f}")

    print("\n  Running single cycle...\n")
    report = agent.run_cycle()

    print(f"\n{'─' * 50}")
    print(f"  Cycle Report")
    print(f"{'─' * 50}")
    print(f"  Cities processed:    {report['cities_processed']}")
    print(f"  Opportunities found: {report['opportunities_found']}")
    print(f"  Trades placed:       {report['trades_placed']}")
    print(f"  Trades skipped:      {report['trades_skipped']}")
    print(f"  Final balance:       ${report.get('final_balance', 0):.2f}")
    print(f"  Elapsed:             {report.get('elapsed_seconds', 0):.1f}s")

    if report["trade_details"]:
        print(f"\n  Trade Details:")
        for td in report["trade_details"]:
            print(f"    {td['side']:3s} ${td['amount']:>7.2f} on "
                  f"{td['city']}/{td['event']} "
                  f"(edge={td['edge']:+.1%}, prob={td['our_prob']:.1%} "
                  f"vs mkt={td['market_price']:.1%})")

    if report["errors"]:
        print(f"\n  Errors:")
        for err in report["errors"]:
            print(f"    ⚠ {err}")
