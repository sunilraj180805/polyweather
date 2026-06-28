"""
PolyWeather — Prediction Model

Statistical / heuristic probability calculator that compares current and
forecasted weather data against historical climate normals to produce
a 0.0–1.0 confidence score for specific weather events.

Core methodology:
  1. Retrieve the forecast value and the historical normal for the same city
     and calendar month.
  2. Compute a Z-score: ``z = (forecast - mean) / stddev``.
  3. Apply a CDF (cumulative distribution function) to convert the Z-score
     into a probability.
  4. Adjust the raw probability for:
     - Forecast horizon (confidence decays with distance)
     - Recent accuracy (if recent actuals are provided)
     - Minimum / maximum clipping (avoid degenerate 0 / 1 probabilities)

Usage:
    from prediction_model import WeatherPredictor
    from data_engine import WeatherDataEngine

    engine    = WeatherDataEngine()
    predictor = WeatherPredictor(engine)

    prob = predictor.predict_temperature_above("new_york", 90, "2026-06-30")
    # => 0.42  (42 % chance max temp > 90 °F)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import numpy as np

import config
from data_engine import WeatherDataEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via the complementary error function.

    Returns P(Z ≤ z) for a standard normal distribution.
    Equivalent to ``scipy.stats.norm.cdf(z)`` but avoids the scipy dep.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Probability density of N(mu, sigma) at point x."""
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    return (
        math.exp(-0.5 * ((x - mu) / sigma) ** 2)
        / (sigma * math.sqrt(2.0 * math.pi))
    )


def _horizon_decay(days_ahead: int) -> float:
    """Return a confidence multiplier that decays with forecast horizon.

    Day 0 (today)   → 1.00
    Day 1 (tomorrow) → 0.95
    Day 3            → 0.85
    Day 7            → 0.65
    Day 14           → 0.40

    Uses an exponential decay: ``exp(-0.05 * days_ahead)``.
    """
    return max(0.30, math.exp(-0.05 * days_ahead))


def _clip_probability(p: float) -> float:
    """Clip probability into [MIN_PROBABILITY, MAX_PROBABILITY]."""
    return max(config.MIN_PROBABILITY, min(config.MAX_PROBABILITY, p))


# ---------------------------------------------------------------------------
# Prediction Result Container
# ---------------------------------------------------------------------------

class PredictionResult:
    """Immutable container for a single prediction output."""

    __slots__ = (
        "city_key", "city_name", "event_type", "target_date",
        "probability", "confidence", "forecast_value", "historical_mean",
        "historical_stddev", "z_score", "horizon_days", "source_notes",
    )

    def __init__(
        self,
        city_key: str,
        city_name: str,
        event_type: str,
        target_date: str,
        probability: float,
        confidence: float,
        forecast_value: float | None,
        historical_mean: float | None,
        historical_stddev: float | None,
        z_score: float | None,
        horizon_days: int,
        source_notes: str = "",
    ) -> None:
        self.city_key = city_key
        self.city_name = city_name
        self.event_type = event_type
        self.target_date = target_date
        self.probability = probability
        self.confidence = confidence
        self.forecast_value = forecast_value
        self.historical_mean = historical_mean
        self.historical_stddev = historical_stddev
        self.z_score = z_score
        self.horizon_days = horizon_days
        self.source_notes = source_notes

    def to_dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"Prediction({self.city_name} | {self.event_type} on {self.target_date}: "
            f"P={self.probability:.3f}, confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# WeatherPredictor
# ---------------------------------------------------------------------------

class WeatherPredictor:
    """Generates probability estimates for weather-based prediction markets.

    All ``predict_*`` methods return a ``PredictionResult`` (or list thereof).
    """

    def __init__(self, engine: WeatherDataEngine | None = None) -> None:
        self._engine = engine or WeatherDataEngine()

    # ------------------------------------------------------------------
    # Temperature threshold predictions
    # ------------------------------------------------------------------

    def predict_temperature_above(
        self,
        city_key: str,
        threshold_f: float,
        target_date: str,
    ) -> PredictionResult:
        """Probability that the daily max temperature exceeds *threshold_f* °F.

        Algorithm:
          1. Get forecast max temp for *target_date*.
          2. Get historical monthly average high + stddev for the city.
          3. Model the max-temp distribution as N(forecast, σ_historical).
             The forecast itself is the best point estimate; historical σ
             captures day-to-day variability around the mean.
          4. P(T_max > threshold) = 1 - Φ((threshold - forecast) / σ).
          5. Multiply by horizon-decay confidence.
        """
        city = config.CITIES[city_key]
        horizon = self._days_until(target_date)
        forecast_day = self._engine.get_forecast_for_date(city_key, target_date)

        # Parse target month from the date for historical lookup
        target_month = int(target_date.split("-")[1])
        hist = self._engine.fetch_historical_averages(city_key, target_month)

        forecast_max_f = (
            forecast_day["temp_max_f"]
            if forecast_day and forecast_day.get("temp_max_f") is not None
            else hist.get("temp_high_f")
        )

        mean_high_f = hist.get("temp_high_f", forecast_max_f or 70)
        stddev_f = hist.get("temp_stddev_f", 6.0)

        # Use the forecast as the centre of our temperature distribution.
        # If forecast is unavailable, fall back to historical mean.
        mu = forecast_max_f if forecast_max_f is not None else mean_high_f

        if stddev_f <= 0:
            stddev_f = 5.0  # safety floor

        # Z-score: how many stddevs is the threshold above the expected max?
        z = (threshold_f - mu) / stddev_f

        # P(T > threshold) = 1 - CDF(z)
        raw_prob = 1.0 - _normal_cdf(z)

        # Apply horizon decay
        decay = _horizon_decay(horizon)
        # Blend towards base-rate (historical) with distance
        hist_base = 1.0 - _normal_cdf((threshold_f - mean_high_f) / stddev_f)
        adjusted_prob = raw_prob * decay + hist_base * (1 - decay)

        final_prob = _clip_probability(adjusted_prob)

        return PredictionResult(
            city_key=city_key,
            city_name=city["display_name"],
            event_type=f"temp_above_{int(threshold_f)}f",
            target_date=target_date,
            probability=round(final_prob, 4),
            confidence=round(decay, 4),
            forecast_value=forecast_max_f,
            historical_mean=mean_high_f,
            historical_stddev=stddev_f,
            z_score=round(z, 3),
            horizon_days=horizon,
            source_notes=(
                f"Forecast max={forecast_max_f}°F, hist avg high={mean_high_f}°F, "
                f"σ={stddev_f}°F, horizon={horizon}d"
            ),
        )

    def predict_temperature_below(
        self,
        city_key: str,
        threshold_f: float,
        target_date: str,
    ) -> PredictionResult:
        """Probability that the daily min temperature is below *threshold_f* °F.

        Same methodology as ``predict_temperature_above`` but uses min temps
        and the lower tail of the distribution.
        """
        city = config.CITIES[city_key]
        horizon = self._days_until(target_date)
        forecast_day = self._engine.get_forecast_for_date(city_key, target_date)

        target_month = int(target_date.split("-")[1])
        hist = self._engine.fetch_historical_averages(city_key, target_month)

        forecast_min_f = (
            forecast_day["temp_min_f"]
            if forecast_day and forecast_day.get("temp_min_f") is not None
            else hist.get("temp_low_f")
        )

        mean_low_f = hist.get("temp_low_f", forecast_min_f or 50)
        # Use slightly higher stddev for lows (overnight temps are more variable)
        stddev_f = hist.get("temp_stddev_f", 6.0) * 1.1

        mu = forecast_min_f if forecast_min_f is not None else mean_low_f

        if stddev_f <= 0:
            stddev_f = 5.0

        z = (threshold_f - mu) / stddev_f
        raw_prob = _normal_cdf(z)  # P(T < threshold)

        decay = _horizon_decay(horizon)
        hist_base = _normal_cdf((threshold_f - mean_low_f) / stddev_f)
        adjusted_prob = raw_prob * decay + hist_base * (1 - decay)

        final_prob = _clip_probability(adjusted_prob)

        return PredictionResult(
            city_key=city_key,
            city_name=city["display_name"],
            event_type=f"temp_below_{int(threshold_f)}f",
            target_date=target_date,
            probability=round(final_prob, 4),
            confidence=round(decay, 4),
            forecast_value=forecast_min_f,
            historical_mean=mean_low_f,
            historical_stddev=round(stddev_f, 2),
            z_score=round(z, 3),
            horizon_days=horizon,
            source_notes=(
                f"Forecast min={forecast_min_f}°F, hist avg low={mean_low_f}°F, "
                f"σ={stddev_f:.1f}°F, horizon={horizon}d"
            ),
        )

    # ------------------------------------------------------------------
    # Precipitation prediction
    # ------------------------------------------------------------------

    def predict_precipitation(
        self,
        city_key: str,
        target_date: str,
    ) -> PredictionResult:
        """Probability of measurable precipitation (≥ 0.25 mm) on *target_date*.

        Uses a hybrid approach:
          - If the forecast includes a precipitation probability, use it
            directly (weather models are well-calibrated for this).
          - Otherwise, derive from historical precip-days ratio and forecast
            conditions.
        """
        city = config.CITIES[city_key]
        horizon = self._days_until(target_date)
        forecast_day = self._engine.get_forecast_for_date(city_key, target_date)

        target_month = int(target_date.split("-")[1])
        hist = self._engine.fetch_historical_averages(city_key, target_month)

        # Historical base rate: precip_days / days_in_month
        days_in_month = 30  # approximation
        hist_precip_rate = hist.get("precip_days", 10) / days_in_month

        forecast_precip_prob = None
        forecast_precip_mm = None

        if forecast_day:
            forecast_precip_prob = forecast_day.get("precip_prob_pct")
            forecast_precip_mm = forecast_day.get("precip_mm")

        if forecast_precip_prob is not None:
            # The weather model gave us a direct probability — use it
            raw_prob = forecast_precip_prob / 100.0
            source = f"Direct from forecast: {forecast_precip_prob}%"
        elif forecast_precip_mm is not None and forecast_precip_mm > 0:
            # Forecast shows rain — high probability
            raw_prob = min(0.90, 0.50 + forecast_precip_mm / 20.0)
            source = f"Derived from forecast precip={forecast_precip_mm}mm"
        else:
            # No forecast data — fall back to historical rate
            raw_prob = hist_precip_rate
            source = f"Historical base rate: {hist_precip_rate:.2f}"

        # Blend with historical base rate by horizon decay
        decay = _horizon_decay(horizon)
        adjusted_prob = raw_prob * decay + hist_precip_rate * (1 - decay)
        final_prob = _clip_probability(adjusted_prob)

        return PredictionResult(
            city_key=city_key,
            city_name=city["display_name"],
            event_type="precipitation",
            target_date=target_date,
            probability=round(final_prob, 4),
            confidence=round(decay, 4),
            forecast_value=forecast_precip_mm,
            historical_mean=round(hist_precip_rate, 4),
            historical_stddev=None,
            z_score=None,
            horizon_days=horizon,
            source_notes=source,
        )

    # ------------------------------------------------------------------
    # Temperature range / bucket prediction
    # ------------------------------------------------------------------

    def predict_temperature_range(
        self,
        city_key: str,
        target_date: str,
    ) -> list[dict]:
        """Probability distribution across standard 5°F temperature buckets.

        Returns a sorted list of dicts::

            [
                {"bucket": "80-85", "low_f": 80, "high_f": 85, "probability": 0.32},
                {"bucket": "85-90", "low_f": 85, "high_f": 90, "probability": 0.28},
                ...
            ]

        Uses the Gaussian model centred on the forecast max temp with
        historical σ.  Each bucket's probability is the integral of the
        normal PDF over [low, high].
        """
        city = config.CITIES[city_key]
        horizon = self._days_until(target_date)
        forecast_day = self._engine.get_forecast_for_date(city_key, target_date)

        target_month = int(target_date.split("-")[1])
        hist = self._engine.fetch_historical_averages(city_key, target_month)

        forecast_max_f = (
            forecast_day["temp_max_f"]
            if forecast_day and forecast_day.get("temp_max_f") is not None
            else hist.get("temp_high_f", 75)
        )

        mean_high_f = hist.get("temp_high_f", 75)
        stddev_f = hist.get("temp_stddev_f", 6.0)

        # Centre distribution on forecast (decayed towards historical mean)
        decay = _horizon_decay(horizon)
        mu = forecast_max_f * decay + mean_high_f * (1 - decay)

        # Widen stddev slightly for longer horizons
        sigma = stddev_f * (1.0 + 0.05 * horizon)

        if sigma <= 0:
            sigma = 5.0

        results: list[dict] = []
        total_prob = 0.0

        for low_f, high_f in config.TEMP_BUCKETS_F:
            # P(low ≤ T < high) = Φ((high - mu)/σ) - Φ((low - mu)/σ)
            p = _normal_cdf((high_f - mu) / sigma) - _normal_cdf((low_f - mu) / sigma)
            results.append({
                "bucket": f"{int(low_f)}-{int(high_f)}",
                "low_f": low_f,
                "high_f": high_f,
                "probability": round(p, 4),
            })
            total_prob += p

        # Add tail buckets
        lowest = config.TEMP_BUCKETS_F[0][0]
        highest = config.TEMP_BUCKETS_F[-1][1]
        p_below = _normal_cdf((lowest - mu) / sigma)
        p_above = 1.0 - _normal_cdf((highest - mu) / sigma)

        results.insert(0, {
            "bucket": f"below_{int(lowest)}",
            "low_f": float("-inf"),
            "high_f": lowest,
            "probability": round(p_below, 4),
        })
        results.append({
            "bucket": f"above_{int(highest)}",
            "low_f": highest,
            "high_f": float("inf"),
            "probability": round(p_above, 4),
        })

        # Normalise so probabilities sum to 1.0 (they should already be close)
        total = sum(r["probability"] for r in results)
        if total > 0:
            for r in results:
                r["probability"] = round(r["probability"] / total, 4)

        # Sort by descending probability
        results.sort(key=lambda r: r["probability"], reverse=True)

        return results

    # ------------------------------------------------------------------
    # Batch: all predictions for a city on a date
    # ------------------------------------------------------------------

    def predict_all_events(
        self,
        city_key: str,
        target_date: str,
    ) -> dict[str, Any]:
        """Generate all prediction types for a city on a target date.

        Returns a dict with keys for each event type.
        """
        results: dict[str, Any] = {
            "city_key": city_key,
            "city_name": config.CITIES[city_key]["display_name"],
            "target_date": target_date,
            "predictions": {},
        }

        # Temperature thresholds
        for event_key, event_cfg in config.WEATHER_EVENTS.items():
            if event_cfg.get("direction") == "above" and "threshold_f" in event_cfg:
                pred = self.predict_temperature_above(
                    city_key, event_cfg["threshold_f"], target_date
                )
                results["predictions"][event_key] = pred.to_dict()

            elif event_cfg.get("direction") == "below" and "threshold_f" in event_cfg:
                pred = self.predict_temperature_below(
                    city_key, event_cfg["threshold_f"], target_date
                )
                results["predictions"][event_key] = pred.to_dict()

        # Precipitation
        precip = self.predict_precipitation(city_key, target_date)
        results["predictions"]["precipitation"] = precip.to_dict()

        # Temperature range buckets
        buckets = self.predict_temperature_range(city_key, target_date)
        results["predictions"]["temp_range_buckets"] = buckets

        return results

    # ------------------------------------------------------------------
    # Batch: all cities, all events
    # ------------------------------------------------------------------

    def predict_all_cities(
        self,
        target_date: str | None = None,
    ) -> dict[str, dict]:
        """Run all predictions for all configured cities.

        If *target_date* is None, defaults to tomorrow.
        """
        if target_date is None:
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            target_date = tomorrow.strftime("%Y-%m-%d")

        results: dict[str, dict] = {}
        for city_key in config.CITIES:
            try:
                results[city_key] = self.predict_all_events(city_key, target_date)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", city_key, exc)
                results[city_key] = {
                    "city_key": city_key,
                    "error": str(exc),
                }
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_until(target_date: str) -> int:
        """Number of days from today to *target_date* (YYYY-MM-DD)."""
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            now = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            delta = (target - now).days
            return max(0, delta)
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    engine = WeatherDataEngine()
    predictor = WeatherPredictor(engine)

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("PolyWeather Prediction Model — Test Run")
    print(f"Target dates: {tomorrow}, {day_after}")
    print("=" * 70)

    for city_key in config.CITIES:
        city_name = config.CITIES[city_key]["display_name"]
        print(f"\n{'━' * 60}")
        print(f"  {city_name}")
        print(f"{'━' * 60}")

        # Temperature above 90°F
        p90 = predictor.predict_temperature_above(city_key, 90, tomorrow)
        print(f"  P(T_max > 90°F on {tomorrow}):  {p90.probability:.1%}  "
              f"(z={p90.z_score}, conf={p90.confidence:.2f})")

        # Temperature above 80°F
        p80 = predictor.predict_temperature_above(city_key, 80, tomorrow)
        print(f"  P(T_max > 80°F on {tomorrow}):  {p80.probability:.1%}  "
              f"(z={p80.z_score}, conf={p80.confidence:.2f})")

        # Temperature below 32°F
        p32 = predictor.predict_temperature_below(city_key, 32, tomorrow)
        print(f"  P(T_min < 32°F on {tomorrow}):  {p32.probability:.1%}  "
              f"(z={p32.z_score}, conf={p32.confidence:.2f})")

        # Precipitation
        precip = predictor.predict_precipitation(city_key, tomorrow)
        print(f"  P(Rain on {tomorrow}):           {precip.probability:.1%}  "
              f"(conf={precip.confidence:.2f})")

        # Top temp bucket
        buckets = predictor.predict_temperature_range(city_key, tomorrow)
        if buckets:
            top = buckets[0]
            print(f"  Most likely temp bucket:        {top['bucket']}°F  "
                  f"({top['probability']:.1%})")

        print(f"  Forecast value:                 {p90.forecast_value}°F max")
        print(f"  Historical mean high:           {p90.historical_mean}°F  "
              f"(σ={p90.historical_stddev}°F)")
