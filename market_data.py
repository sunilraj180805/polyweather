"""
PolyWeather — Market Data Module

Interfaces with Polymarket's Gamma API to discover and price active
weather prediction markets.  When live markets are unavailable (e.g. off-
season or API outage), a deterministic simulation generates realistic
market odds from historical climate data plus controlled noise.

Usage:
    from market_data import PolymarketFetcher
    fetcher = PolymarketFetcher()

    # Try live markets
    markets = fetcher.fetch_weather_markets()

    # Simulated fallback (always works)
    odds = fetcher.simulate_market_odds("new_york", "temp_above_90f")
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _MarketCache:
    """Simple TTL cache for market data."""

    def __init__(self, ttl: int = config.MARKET_CACHE_TTL_SECONDS):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Normalised market data structure
# ---------------------------------------------------------------------------

def _empty_market() -> dict:
    return {
        "market_id": "",
        "slug": "",
        "question": "",
        "city": "",
        "event_type": "",
        "end_date": "",
        "yes_price": 0.5,
        "no_price": 0.5,
        "volume": 0.0,
        "liquidity": 0.0,
        "source": "",         # "gamma_api" or "simulated"
        "active": True,
    }


# ---------------------------------------------------------------------------
# City / event keyword mapping for market discovery
# ---------------------------------------------------------------------------

_CITY_KEYWORDS: dict[str, list[str]] = {
    "new_york": ["new york", "nyc", "jfk", "kjfk", "manhattan"],
    "london": ["london", "heathrow", "egll", "lhr"],
    "tokyo": ["tokyo", "haneda", "narita", "rjtt"],
    "mumbai": ["mumbai", "bombay", "vabb"],
    "sydney": ["sydney", "yssy", "kingsford"],
}

_EVENT_KEYWORDS: dict[str, list[str]] = {
    "temp_above_90f": ["above 90", "over 90", "exceed 90", "90°", "90 degrees",
                       "above 32c", "over 32c"],
    "temp_above_80f": ["above 80", "over 80", "exceed 80", "80°", "80 degrees",
                       "above 27c", "over 27c"],
    "temp_below_32f": ["below 32", "under 32", "freeze", "frost", "below 0c",
                       "under 0c"],
    "precipitation": ["rain", "precipitation", "shower", "storm", "snow",
                      "drizzle", "wet"],
}


def _match_city(text: str) -> str | None:
    """Identify city key from market text."""
    text_lower = text.lower()
    for city_key, keywords in _CITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return city_key
    return None


def _match_event(text: str) -> str | None:
    """Identify event type from market text."""
    text_lower = text.lower()
    for event_key, keywords in _EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return event_key
    return None


# ---------------------------------------------------------------------------
# PolymarketFetcher
# ---------------------------------------------------------------------------

class PolymarketFetcher:
    """Fetch and normalise Polymarket weather market data.

    Priority:
      1. Gamma API (live markets)
      2. Simulated markets (deterministic fallback)
    """

    def __init__(self) -> None:
        self._cache = _MarketCache()
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolyWeather/1.0",
        })

    # ------------------------------------------------------------------
    # Live Gamma API
    # ------------------------------------------------------------------

    def fetch_weather_markets(self) -> list[dict]:
        """Query Polymarket Gamma API for active weather markets.

        Returns a list of normalised market dicts for our target cities.
        Falls back to simulated markets if the API returns nothing.
        """
        cached = self._cache.get("weather_markets")
        if cached is not None:
            return cached

        live = self._fetch_from_gamma()
        if live:
            self._cache.set("weather_markets", live)
            return live

        logger.info("No live weather markets found — generating simulated markets")
        simulated = self._generate_all_simulated_markets()
        self._cache.set("weather_markets", simulated)
        return simulated

    def _fetch_from_gamma(self) -> list[dict]:
        """Hit the Gamma API and filter for weather markets in target cities."""
        markets: list[dict] = []

        try:
            # Search for weather-related markets
            for search_term in ["weather", "temperature", "rain"]:
                url = f"{config.POLYMARKET_GAMMA_API}/markets"
                params = {
                    "limit": 50,
                    "active": True,
                    "closed": False,
                    "tag": "weather",
                }
                resp = self._session.get(url, params=params, timeout=10)

                if resp.status_code == 200:
                    raw_markets = resp.json()
                    if isinstance(raw_markets, list):
                        for raw in raw_markets:
                            parsed = self._parse_gamma_market(raw)
                            if parsed and parsed["city"]:
                                markets.append(parsed)

                # Also try the events endpoint
                events_url = f"{config.POLYMARKET_GAMMA_API}/events"
                params_ev = {
                    "limit": 30,
                    "active": True,
                    "tag": "weather",
                }
                resp_ev = self._session.get(events_url, params=params_ev, timeout=10)

                if resp_ev.status_code == 200:
                    raw_events = resp_ev.json()
                    if isinstance(raw_events, list):
                        for event in raw_events:
                            event_markets = event.get("markets", [])
                            for raw in event_markets:
                                parsed = self._parse_gamma_market(raw)
                                if parsed and parsed["city"]:
                                    markets.append(parsed)

        except requests.RequestException as exc:
            logger.warning("Gamma API request failed: %s", exc)
        except (ValueError, KeyError) as exc:
            logger.warning("Gamma API parse error: %s", exc)

        # Deduplicate by market_id
        seen: set[str] = set()
        unique: list[dict] = []
        for m in markets:
            mid = m["market_id"]
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)

        logger.info("Gamma API returned %d weather markets", len(unique))
        return unique

    def _parse_gamma_market(self, raw: dict) -> dict | None:
        """Convert a raw Gamma API market object into our normalised schema."""
        question = raw.get("question", "") or raw.get("title", "")
        if not question:
            return None

        city = _match_city(question)
        event = _match_event(question)

        # Extract prices from outcomes/tokens
        yes_price = 0.5
        no_price = 0.5

        outcomes = raw.get("outcomePrices", raw.get("outcomes"))
        if isinstance(outcomes, str):
            # Some Gamma responses encode prices as JSON string
            try:
                import json
                prices = json.loads(outcomes)
                if isinstance(prices, list) and len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
            except (json.JSONDecodeError, ValueError, IndexError):
                pass
        elif isinstance(outcomes, list) and len(outcomes) >= 2:
            try:
                yes_price = float(outcomes[0])
                no_price = float(outcomes[1])
            except (ValueError, TypeError):
                pass

        # Try bestAsk/bestBid
        if yes_price == 0.5:
            best_ask = raw.get("bestAsk")
            best_bid = raw.get("bestBid")
            if best_ask is not None and best_bid is not None:
                try:
                    yes_price = (float(best_ask) + float(best_bid)) / 2
                    no_price = 1.0 - yes_price
                except (ValueError, TypeError):
                    pass

        market = _empty_market()
        market["market_id"] = str(raw.get("id", raw.get("conditionId", "")))
        market["slug"] = raw.get("slug", raw.get("questionSlug", ""))
        market["question"] = question
        market["city"] = city or ""
        market["event_type"] = event or ""
        market["end_date"] = raw.get("endDate", raw.get("endDateIso", ""))
        market["yes_price"] = round(max(0.01, min(0.99, yes_price)), 4)
        market["no_price"] = round(max(0.01, min(0.99, no_price)), 4)
        market["volume"] = float(raw.get("volume", 0) or 0)
        market["liquidity"] = float(raw.get("liquidity", 0) or 0)
        market["source"] = "gamma_api"
        market["active"] = raw.get("active", True)

        return market

    # ------------------------------------------------------------------
    # Market price lookup
    # ------------------------------------------------------------------

    def get_market_odds(self, market_id: str) -> dict | None:
        """Return current YES/NO prices for a specific market."""
        markets = self.fetch_weather_markets()
        for m in markets:
            if m["market_id"] == market_id:
                return {
                    "market_id": market_id,
                    "yes_price": m["yes_price"],
                    "no_price": m["no_price"],
                    "source": m["source"],
                }
        return None

    # ------------------------------------------------------------------
    # Simulated markets (deterministic fallback)
    # ------------------------------------------------------------------

    def simulate_market_odds(
        self,
        city_key: str,
        event_type: str,
        target_date: str | None = None,
    ) -> dict:
        """Generate plausible simulated market odds.

        Uses a deterministic seed derived from (city, event, date) so the
        same inputs always produce the same "market price", which makes
        backtesting reproducible.

        The simulated price is anchored to the historical base rate for
        the city/month, then perturbed with controlled noise to mimic
        market inefficiency.
        """
        if target_date is None:
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            target_date = tomorrow.strftime("%Y-%m-%d")

        # Deterministic seed
        seed_str = f"{city_key}:{event_type}:{target_date}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Get historical base rate
        base_rate = self._historical_base_rate(city_key, event_type, target_date)

        # Add noise: uniform ±10% around the base rate
        noise = rng.uniform(-0.10, 0.10)
        simulated_price = base_rate + noise

        # Occasionally add larger "mispricing" (±5-15% extra) to create
        # trading opportunities for the agent
        if rng.random() < 0.30:  # 30% of markets have bigger misprice
            extra_noise = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
            simulated_price += extra_noise

        # Clamp to valid price range
        simulated_price = max(0.05, min(0.95, simulated_price))

        # Build market ID
        market_id = f"sim_{city_key}_{event_type}_{target_date}"

        city_name = config.CITIES.get(city_key, {}).get("display_name", city_key)
        event_desc = config.WEATHER_EVENTS.get(event_type, {}).get(
            "description", event_type
        )

        return {
            "market_id": market_id,
            "slug": market_id,
            "question": f"Will {event_desc} occur in {city_name} on {target_date}?",
            "city": city_key,
            "event_type": event_type,
            "end_date": target_date,
            "yes_price": round(simulated_price, 4),
            "no_price": round(1.0 - simulated_price, 4),
            "volume": round(rng.uniform(5000, 50000), 2),
            "liquidity": round(rng.uniform(2000, 20000), 2),
            "source": "simulated",
            "active": True,
        }

    def _historical_base_rate(
        self,
        city_key: str,
        event_type: str,
        target_date: str,
    ) -> float:
        """Derive a base probability from historical climate data.

        This anchors simulated market prices near realistic values.
        """
        import json

        try:
            month = int(target_date.split("-")[1])
        except (ValueError, IndexError):
            month = 6  # default to June

        # Load historical data
        try:
            with open(config.HISTORICAL_AVERAGES_PATH, "r") as fh:
                hist_data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.50  # no data → 50/50

        city_hist = hist_data.get(city_key, {})
        month_data = city_hist.get(str(month), {})

        if not month_data:
            return 0.50

        high_f = month_data.get("temp_high_f", 75)
        low_f = month_data.get("temp_low_f", 55)
        stddev = month_data.get("temp_stddev_f", 6.0)
        precip_days = month_data.get("precip_days", 10)

        event_cfg = config.WEATHER_EVENTS.get(event_type, {})

        if event_type == "precipitation":
            # Base rate = precip_days / 30
            return min(0.90, max(0.10, precip_days / 30.0))

        threshold_f = event_cfg.get("threshold_f")
        if threshold_f is None:
            return 0.50

        direction = event_cfg.get("direction", "above")

        if direction == "above":
            # P(max temp > threshold) using historical mean
            z = (threshold_f - high_f) / max(stddev, 1.0)
            return max(0.05, min(0.95, 1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
        elif direction == "below":
            z = (threshold_f - low_f) / max(stddev, 1.0)
            return max(0.05, min(0.95, 0.5 * (1 + math.erf(z / math.sqrt(2)))))

        return 0.50

    # ------------------------------------------------------------------
    # Generate simulated markets for all cities × event types
    # ------------------------------------------------------------------

    def _generate_all_simulated_markets(self) -> list[dict]:
        """Create a full set of simulated weather markets."""
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        day_after = datetime.now(timezone.utc) + timedelta(days=2)
        dates = [
            tomorrow.strftime("%Y-%m-%d"),
            day_after.strftime("%Y-%m-%d"),
        ]

        markets: list[dict] = []
        for city_key in config.CITIES:
            for event_type in config.WEATHER_EVENTS:
                if event_type == "temp_range_bucket":
                    continue  # Skip — range markets need special handling
                for date_str in dates:
                    m = self.simulate_market_odds(city_key, event_type, date_str)
                    markets.append(m)

        logger.info("Generated %d simulated weather markets", len(markets))
        return markets

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_markets_for_city(self, city_key: str) -> list[dict]:
        """Return all markets (live + simulated) for a specific city."""
        all_markets = self.fetch_weather_markets()
        return [m for m in all_markets if m["city"] == city_key]

    def clear_cache(self) -> None:
        self._cache = _MarketCache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetcher = PolymarketFetcher()

    print("=" * 65)
    print("PolyWeather Market Data — Test Run")
    print("=" * 65)

    # 1. Attempt live Gamma API
    print("\n─── Live Gamma API Fetch ───")
    markets = fetcher.fetch_weather_markets()
    live_count = sum(1 for m in markets if m["source"] == "gamma_api")
    sim_count = sum(1 for m in markets if m["source"] == "simulated")
    print(f"  Total markets: {len(markets)} (live: {live_count}, simulated: {sim_count})")

    # 2. Show markets by city
    print("\n─── Markets by City ───")
    for city_key in config.CITIES:
        city_markets = fetcher.get_markets_for_city(city_key)
        city_name = config.CITIES[city_key]["display_name"]
        print(f"\n  {city_name} ({len(city_markets)} markets):")
        for m in city_markets[:4]:
            print(f"    {m['event_type']:20s} | YES: {m['yes_price']:.2f}  "
                  f"NO: {m['no_price']:.2f}  | {m['source']}")

    # 3. Specific simulation
    print("\n─── Specific Simulated Odds ───")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    for city_key in ["new_york", "london", "tokyo"]:
        for event in ["temp_above_90f", "precipitation"]:
            odds = fetcher.simulate_market_odds(city_key, event, tomorrow)
            print(f"  {city_key}/{event}: YES={odds['yes_price']:.3f} "
                  f"NO={odds['no_price']:.3f}")
