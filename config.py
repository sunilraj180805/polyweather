"""
PolyWeather — Central Configuration Module

All system-wide constants, city definitions, model parameters, and trading
configuration live here.  Every other module imports from this file so
there is a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "trades.db"
HISTORICAL_AVERAGES_PATH = DATA_DIR / "historical_averages.json"

# ---------------------------------------------------------------------------
# API Keys (loaded from .env)
# ---------------------------------------------------------------------------
APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# City Definitions
# ---------------------------------------------------------------------------
# Each city has:
#   - display_name: Human-readable name
#   - lat / lon: Coordinates for weather API queries
#   - airport_code: ICAO code used for Polymarket settlement
#   - timezone: IANA timezone string
#   - elevation_m: Elevation in metres (for Open-Meteo accuracy)

CITIES: dict[str, dict] = {
    "new_york": {
        "display_name": "New York",
        "lat": 40.7128,
        "lon": -74.0060,
        "airport_code": "KJFK",
        "timezone": "America/New_York",
        "elevation_m": 13,
    },
    "london": {
        "display_name": "London",
        "lat": 51.5074,
        "lon": -0.1278,
        "airport_code": "EGLL",
        "timezone": "Europe/London",
        "elevation_m": 25,
    },
    "tokyo": {
        "display_name": "Tokyo",
        "lat": 35.6762,
        "lon": 139.6503,
        "airport_code": "RJTT",
        "timezone": "Asia/Tokyo",
        "elevation_m": 40,
    },
    "mumbai": {
        "display_name": "Mumbai",
        "lat": 19.0760,
        "lon": 72.8777,
        "airport_code": "VABB",
        "timezone": "Asia/Kolkata",
        "elevation_m": 14,
    },
    "sydney": {
        "display_name": "Sydney",
        "lat": -33.8688,
        "lon": 151.2093,
        "airport_code": "YSSY",
        "timezone": "Australia/Sydney",
        "elevation_m": 58,
    },
}

# ---------------------------------------------------------------------------
# Weather Event Types
# ---------------------------------------------------------------------------
# These define the kinds of prediction markets we look for or simulate.

WEATHER_EVENTS = {
    "temp_above_90f": {
        "description": "Max temperature above 90°F (32.2°C)",
        "threshold_f": 90,
        "threshold_c": 32.2,
        "metric": "temperature_max",
        "direction": "above",
    },
    "temp_above_80f": {
        "description": "Max temperature above 80°F (26.7°C)",
        "threshold_f": 80,
        "threshold_c": 26.7,
        "metric": "temperature_max",
        "direction": "above",
    },
    "temp_below_32f": {
        "description": "Min temperature below 32°F (0°C)",
        "threshold_f": 32,
        "threshold_c": 0.0,
        "metric": "temperature_min",
        "direction": "below",
    },
    "precipitation": {
        "description": "Measurable precipitation (≥ 0.01 in / 0.25 mm)",
        "threshold_mm": 0.25,
        "metric": "precipitation",
        "direction": "above",
    },
    "temp_range_bucket": {
        "description": "Temperature falls within a specific 5°F bucket",
        "metric": "temperature_max",
        "direction": "range",
    },
}

# ---------------------------------------------------------------------------
# Temperature Buckets (for range-style Polymarket markets)
# ---------------------------------------------------------------------------
# Polymarket weather markets often resolve on 5°F buckets.

TEMP_BUCKETS_F: list[tuple[float, float]] = [
    (50, 55), (55, 60), (60, 65), (65, 70), (70, 75),
    (75, 80), (80, 85), (85, 90), (90, 95), (95, 100),
    (100, 105), (105, 110),
]

# ---------------------------------------------------------------------------
# OpenRouter / LLM Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Free-tier models on OpenRouter (as of 2026-06)
# The orchestrator cycles through these if one fails.
LLM_MODELS: list[str] = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

LLM_PRIMARY_MODEL: str = LLM_MODELS[0]

LLM_TEMPERATURE: float = 0.3       # Low temp for analytical consistency
LLM_MAX_TOKENS: int = 2048         # Enough for analysis + JSON response
LLM_REQUEST_TIMEOUT: int = 60      # Seconds

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------
INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000"))

# Kelly Criterion guardrails
KELLY_FRACTION_CAP: float = 0.5     # Use half-Kelly for safety
MAX_SINGLE_POSITION_PCT: float = 0.10   # Max 10% of bankroll per trade
MAX_TOTAL_EXPOSURE_PCT: float = 0.50    # Max 50% of bankroll deployed
MIN_EDGE_THRESHOLD: float = 0.05        # Min 5% edge to trigger a trade
MIN_PROBABILITY: float = 0.05           # Ignore events below 5% probability
MAX_PROBABILITY: float = 0.95           # Ignore events above 95% probability

# ---------------------------------------------------------------------------
# Polymarket API
# ---------------------------------------------------------------------------
POLYMARKET_GAMMA_API: str = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API: str = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
AGENT_CYCLE_MINUTES: int = int(os.getenv("AGENT_CYCLE_MINUTES", "30"))

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
WEATHER_CACHE_TTL_SECONDS: int = 1800   # 30 minutes
MARKET_CACHE_TTL_SECONDS: int = 300     # 5 minutes

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"
APIFY_WEATHER_ACTOR: str = "apify/weather-api"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
