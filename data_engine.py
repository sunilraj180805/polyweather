"""
PolyWeather — Weather Data Engine

Fetches current conditions, forecasts, and historical context for 5 target
cities.  Primary data source is Apify's ``apify/weather-api`` actor;
when no Apify token is configured (or the actor fails), the engine falls
back transparently to the free Open-Meteo API.

All public methods return normalised dicts so consumers never need to care
about which upstream supplied the data.

Usage:
    from data_engine import WeatherDataEngine
    engine = WeatherDataEngine()
    data   = engine.fetch_all_cities()          # dict[city_key, CityWeather]
    nyc    = engine.fetch_current_weather("new_york")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
class _Cache:
    """Simple TTL cache keyed by arbitrary string keys."""

    def __init__(self, ttl_seconds: int = config.WEATHER_CACHE_TTL_SECONDS):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Normalised data structures (plain dicts for JSON-friendliness)
# ---------------------------------------------------------------------------

def _empty_current() -> dict:
    return {
        "city_key": "",
        "city_name": "",
        "timestamp": "",
        "source": "",
        "temp_c": None,
        "temp_f": None,
        "feels_like_c": None,
        "feels_like_f": None,
        "humidity_pct": None,
        "precip_mm": None,
        "wind_kph": None,
        "wind_dir": "",
        "condition": "",
        "cloud_cover_pct": None,
        "uv_index": None,
    }


def _empty_forecast_day() -> dict:
    return {
        "date": "",
        "temp_max_c": None,
        "temp_max_f": None,
        "temp_min_c": None,
        "temp_min_f": None,
        "precip_mm": None,
        "precip_prob_pct": None,
        "condition": "",
        "wind_max_kph": None,
        "humidity_pct": None,
        "uv_index": None,
    }


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _c_to_f(c: float | None) -> float | None:
    if c is None:
        return None
    return round(c * 9 / 5 + 32, 1)


def _f_to_c(f: float | None) -> float | None:
    if f is None:
        return None
    return round((f - 32) * 5 / 9, 1)


# ---------------------------------------------------------------------------
# Open-Meteo Fetcher (Free, no API key)
# ---------------------------------------------------------------------------

class _OpenMeteoFetcher:
    """Fetch weather from the Open-Meteo API (completely free, no key)."""

    BASE = config.OPEN_METEO_BASE_URL

    # Open-Meteo WMO weather-code → human description
    _WMO_CODES: dict[int, str] = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
        3: "Overcast", 45: "Foggy", 48: "Rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Freezing rain (light)", 67: "Freezing rain (heavy)",
        71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers",
        86: "Heavy snow showers", 95: "Thunderstorm",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }

    @classmethod
    def _wmo_desc(cls, code: int | None) -> str:
        if code is None:
            return "Unknown"
        return cls._WMO_CODES.get(code, f"WMO {code}")

    @classmethod
    def fetch_current(cls, city_key: str) -> dict:
        """Return normalised current-weather dict for *city_key*."""
        city = config.CITIES[city_key]
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,cloud_cover,"
                "wind_speed_10m,wind_direction_10m"
            ),
            "timezone": city["timezone"],
        }
        resp = requests.get(f"{cls.BASE}/forecast", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current", {})

        out = _empty_current()
        out["city_key"] = city_key
        out["city_name"] = city["display_name"]
        out["timestamp"] = cur.get("time", datetime.now(timezone.utc).isoformat())
        out["source"] = "open-meteo"
        out["temp_c"] = cur.get("temperature_2m")
        out["temp_f"] = _c_to_f(cur.get("temperature_2m"))
        out["feels_like_c"] = cur.get("apparent_temperature")
        out["feels_like_f"] = _c_to_f(cur.get("apparent_temperature"))
        out["humidity_pct"] = cur.get("relative_humidity_2m")
        out["precip_mm"] = cur.get("precipitation", 0.0)
        out["wind_kph"] = cur.get("wind_speed_10m")
        out["wind_dir"] = cls._wind_deg_to_dir(cur.get("wind_direction_10m"))
        out["condition"] = cls._wmo_desc(cur.get("weather_code"))
        out["cloud_cover_pct"] = cur.get("cloud_cover")
        return out

    @classmethod
    def fetch_forecast(cls, city_key: str, days: int = 7) -> list[dict]:
        """Return list of normalised daily-forecast dicts."""
        city = config.CITIES[city_key]
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max,"
                "weather_code,wind_speed_10m_max,"
                "relative_humidity_2m_mean,uv_index_max"
            ),
            "timezone": city["timezone"],
            "forecast_days": min(days, 16),
        }
        resp = requests.get(f"{cls.BASE}/forecast", params=params, timeout=15)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        dates = daily.get("time", [])
        results: list[dict] = []
        for i, date_str in enumerate(dates):
            day = _empty_forecast_day()
            day["date"] = date_str
            day["temp_max_c"] = _safe_idx(daily.get("temperature_2m_max"), i)
            day["temp_max_f"] = _c_to_f(day["temp_max_c"])
            day["temp_min_c"] = _safe_idx(daily.get("temperature_2m_min"), i)
            day["temp_min_f"] = _c_to_f(day["temp_min_c"])
            day["precip_mm"] = _safe_idx(daily.get("precipitation_sum"), i)
            day["precip_prob_pct"] = _safe_idx(
                daily.get("precipitation_probability_max"), i
            )
            day["condition"] = cls._wmo_desc(
                _safe_idx(daily.get("weather_code"), i)
            )
            day["wind_max_kph"] = _safe_idx(
                daily.get("wind_speed_10m_max"), i
            )
            day["humidity_pct"] = _safe_idx(
                daily.get("relative_humidity_2m_mean"), i
            )
            day["uv_index"] = _safe_idx(daily.get("uv_index_max"), i)
            results.append(day)
        return results

    @staticmethod
    def _wind_deg_to_dir(deg: float | None) -> str:
        if deg is None:
            return ""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(deg / 22.5) % 16
        return dirs[idx]


# ---------------------------------------------------------------------------
# Apify Fetcher
# ---------------------------------------------------------------------------

class _ApifyFetcher:
    """Fetch weather data via Apify's weather-api actor."""

    @staticmethod
    def is_available() -> bool:
        return bool(config.APIFY_API_TOKEN)

    @classmethod
    def fetch_current(cls, city_key: str) -> dict:
        """Run the Apify weather actor and return normalised current dict."""
        from apify_client import ApifyClient

        city = config.CITIES[city_key]
        client = ApifyClient(config.APIFY_API_TOKEN)

        run_input = {
            "locations": [city["display_name"]],
            "units": "metric",
        }

        logger.info("Apify: starting actor for %s", city["display_name"])
        run = client.actor(config.APIFY_WEATHER_ACTOR).call(
            run_input=run_input,
        )

        items = list(
            client.dataset(run["defaultDatasetId"]).iterate_items()
        )
        if not items:
            raise RuntimeError(f"Apify returned no data for {city_key}")

        raw = items[0]
        out = _empty_current()
        out["city_key"] = city_key
        out["city_name"] = city["display_name"]
        out["timestamp"] = raw.get("timestamp", datetime.now(timezone.utc).isoformat())
        out["source"] = "apify"

        # Apify weather-api returns metric by default
        temp_c = raw.get("temperature") or raw.get("tempC")
        if temp_c is not None:
            out["temp_c"] = float(temp_c)
            out["temp_f"] = _c_to_f(float(temp_c))

        feels_c = raw.get("feelsLike") or raw.get("feelsLikeC")
        if feels_c is not None:
            out["feels_like_c"] = float(feels_c)
            out["feels_like_f"] = _c_to_f(float(feels_c))

        out["humidity_pct"] = _safe_float(raw.get("humidity"))
        out["precip_mm"] = _safe_float(raw.get("precipMM", 0))
        out["wind_kph"] = _safe_float(raw.get("windspeedKmph") or raw.get("windKph"))
        out["wind_dir"] = raw.get("windDir", "")
        out["condition"] = raw.get("weatherDesc", [{}])[0].get("value", "") if isinstance(
            raw.get("weatherDesc"), list
        ) else str(raw.get("condition", raw.get("weatherDesc", "")))
        out["cloud_cover_pct"] = _safe_float(raw.get("cloudcover"))

        return out

    @classmethod
    def fetch_forecast(cls, city_key: str, days: int = 7) -> list[dict]:
        """Apify doesn't natively supply multi-day forecast in the free actor,
        so we fall back to Open-Meteo for forecasts even when Apify is the
        primary current-weather source."""
        return _OpenMeteoFetcher.fetch_forecast(city_key, days)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_idx(lst: list | None, idx: int) -> Any:
    if lst is None or idx >= len(lst):
        return None
    return lst[idx]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Historical Averages Loader
# ---------------------------------------------------------------------------

_historical_data: dict | None = None


def _load_historical() -> dict:
    global _historical_data
    if _historical_data is not None:
        return _historical_data
    path = config.HISTORICAL_AVERAGES_PATH
    if not path.exists():
        logger.warning("Historical averages file not found at %s", path)
        _historical_data = {}
        return _historical_data
    with open(path, "r", encoding="utf-8") as fh:
        _historical_data = json.load(fh)
    return _historical_data


# ---------------------------------------------------------------------------
# Public API — WeatherDataEngine
# ---------------------------------------------------------------------------

class WeatherDataEngine:
    """Unified interface for fetching weather data across all target cities.

    Automatically selects the best available data source:
      1. Apify (if APIFY_API_TOKEN is set)
      2. Open-Meteo (always available, free)

    All returned data is normalised to the same schema regardless of source.
    """

    def __init__(self) -> None:
        self._cache = _Cache()
        self._use_apify = _ApifyFetcher.is_available()
        if self._use_apify:
            logger.info("Data engine: Apify token detected — using Apify as primary source")
        else:
            logger.info("Data engine: No Apify token — using Open-Meteo (free)")

    # ------------------------------------------------------------------
    # Current weather
    # ------------------------------------------------------------------

    def fetch_current_weather(self, city_key: str) -> dict:
        """Fetch current conditions for a single city.

        Returns a normalised dict (see ``_empty_current``).
        Uses cache if fresh data is available.
        """
        if city_key not in config.CITIES:
            raise ValueError(
                f"Unknown city key '{city_key}'. "
                f"Valid keys: {list(config.CITIES.keys())}"
            )

        cache_key = f"current:{city_key}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s current weather", city_key)
            return cached

        data = self._fetch_current_with_fallback(city_key)
        self._cache.set(cache_key, data)
        return data

    def _fetch_current_with_fallback(self, city_key: str) -> dict:
        """Try Apify first (if available), fall back to Open-Meteo."""
        if self._use_apify:
            try:
                return _ApifyFetcher.fetch_current(city_key)
            except Exception as exc:
                logger.warning(
                    "Apify fetch failed for %s, falling back to Open-Meteo: %s",
                    city_key,
                    exc,
                )
        return _OpenMeteoFetcher.fetch_current(city_key)

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def fetch_forecast(self, city_key: str, days: int = 7) -> list[dict]:
        """Fetch multi-day forecast for a single city.

        Returns a list of normalised daily-forecast dicts.
        """
        if city_key not in config.CITIES:
            raise ValueError(f"Unknown city key '{city_key}'")

        cache_key = f"forecast:{city_key}:{days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._use_apify:
            try:
                data = _ApifyFetcher.fetch_forecast(city_key, days)
            except Exception as exc:
                logger.warning("Apify forecast failed for %s: %s", city_key, exc)
                data = _OpenMeteoFetcher.fetch_forecast(city_key, days)
        else:
            data = _OpenMeteoFetcher.fetch_forecast(city_key, days)

        self._cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    # Historical averages
    # ------------------------------------------------------------------

    def fetch_historical_averages(
        self,
        city_key: str,
        month: int | None = None,
    ) -> dict:
        """Return historical climate normals for *city_key*.

        If *month* is given (1-12), returns that month's averages.
        Otherwise returns the full 12-month dict.
        """
        if city_key not in config.CITIES:
            raise ValueError(f"Unknown city key '{city_key}'")

        data = _load_historical()
        city_data = data.get(city_key, {})

        if month is not None:
            return city_data.get(str(month), {})
        return city_data

    # ------------------------------------------------------------------
    # Bulk fetch
    # ------------------------------------------------------------------

    def fetch_all_cities(self) -> dict[str, dict]:
        """Fetch current weather + 7-day forecast for all configured cities.

        Returns::

            {
                "new_york": {
                    "current": { ... },
                    "forecast": [ ... ],
                    "historical": { ... },   # current month's averages
                    "fetched_at": "2026-06-28T12:00:00Z"
                },
                ...
            }
        """
        now = datetime.now(timezone.utc)
        current_month = now.month
        results: dict[str, dict] = {}

        for city_key in config.CITIES:
            try:
                current = self.fetch_current_weather(city_key)
                forecast = self.fetch_forecast(city_key, days=7)
                historical = self.fetch_historical_averages(city_key, current_month)
                results[city_key] = {
                    "current": current,
                    "forecast": forecast,
                    "historical": historical,
                    "fetched_at": now.isoformat(),
                }
                logger.info(
                    "✓ %s: %.1f°F (%s) | Source: %s",
                    config.CITIES[city_key]["display_name"],
                    current.get("temp_f") or 0,
                    current.get("condition", "?"),
                    current.get("source", "?"),
                )
            except Exception as exc:
                logger.error("Failed to fetch data for %s: %s", city_key, exc)
                results[city_key] = {
                    "current": _empty_current(),
                    "forecast": [],
                    "historical": {},
                    "fetched_at": now.isoformat(),
                    "error": str(exc),
                }

        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_forecast_for_date(
        self, city_key: str, target_date: str
    ) -> dict | None:
        """Return the forecast dict for a specific date (YYYY-MM-DD).

        Returns ``None`` if the date is not in the forecast window.
        """
        forecast = self.fetch_forecast(city_key)
        for day in forecast:
            if day["date"] == target_date:
                return day
        return None

    def clear_cache(self) -> None:
        """Flush all cached weather data."""
        self._cache.clear()
        logger.info("Weather data cache cleared")

    def get_data_summary(self) -> dict:
        """Return a compact summary of data availability and source status."""
        return {
            "primary_source": "apify" if self._use_apify else "open-meteo",
            "apify_configured": self._use_apify,
            "cities": list(config.CITIES.keys()),
            "cache_ttl_seconds": self._cache._ttl,
            "historical_data_loaded": _historical_data is not None,
        }


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    engine = WeatherDataEngine()

    print("=" * 70)
    print("PolyWeather Data Engine — Test Run")
    print("=" * 70)
    print(f"\nData source: {engine.get_data_summary()['primary_source']}")
    print()

    data = engine.fetch_all_cities()
    for city_key, city_data in data.items():
        cur = city_data["current"]
        fc = city_data["forecast"]
        hist = city_data["historical"]
        err = city_data.get("error")

        city_name = config.CITIES[city_key]["display_name"]
        print(f"{'─' * 50}")
        print(f"  {city_name}")
        print(f"{'─' * 50}")

        if err:
            print(f"  ⚠ Error: {err}")
            continue

        print(f"  Current:  {cur['temp_f']}°F / {cur['temp_c']}°C  |  {cur['condition']}")
        print(f"  Humidity: {cur['humidity_pct']}%  |  Wind: {cur['wind_kph']} kph {cur['wind_dir']}")
        print(f"  Precip:   {cur['precip_mm']} mm  |  Source: {cur['source']}")

        if hist:
            print(f"  Historic: avg high {hist.get('temp_high_f')}°F  |  "
                  f"avg low {hist.get('temp_low_f')}°F  |  "
                  f"σ = {hist.get('temp_stddev_f')}°F")

        if fc:
            print(f"  Forecast ({len(fc)} days):")
            for day in fc[:3]:
                print(f"    {day['date']}: {day['temp_min_f']}–{day['temp_max_f']}°F  "
                      f"| {day['condition']}  |  Precip: {day['precip_mm']} mm")

        print()
