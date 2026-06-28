"""
PolyWeather — Risk Manager

Implements the Kelly Criterion for optimal bet sizing and portfolio-level
risk management for the paper-trading system.

The Kelly Criterion formula:
    f* = (p · b − q) / b
where:
    p = our estimated win probability
    q = 1 − p (probability of losing)
    b = net odds received on the bet = (1 / market_price) − 1

We apply "half-Kelly" (multiply by KELLY_FRACTION_CAP = 0.5) as a safety
measure, since real-world edge estimates always have estimation error.

Usage:
    from risk_manager import RiskManager
    rm = RiskManager(bankroll=10000)

    sizing = rm.calculate_position(win_prob=0.65, market_price=0.50, bankroll=10000)
    # => {'kelly_fraction': 0.15, 'position_size': 750.0, 'side': 'YES', ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KellyResult:
    """Output of a Kelly Criterion calculation."""
    win_prob: float
    market_price: float
    implied_prob: float
    edge: float                  # our_prob − implied_prob
    edge_pct: float              # edge as a percentage
    kelly_fraction: float        # raw f*
    adjusted_fraction: float     # after half-Kelly cap
    position_size: float         # dollar amount to bet
    side: str                    # "YES" or "NO"
    tradeable: bool              # whether edge exceeds minimum threshold
    reject_reason: str = ""      # why trade was rejected (if any)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioRisk:
    """Snapshot of current portfolio risk metrics."""
    bankroll: float
    total_exposure: float        # sum of all open position amounts
    exposure_pct: float          # total_exposure / bankroll
    num_positions: int
    max_single_exposure: float
    max_single_pct: float
    city_exposure: dict[str, float] = field(default_factory=dict)
    within_limits: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HedgeSuggestion:
    """A suggested hedge trade to reduce risk."""
    market_id: str
    city: str
    current_side: str
    hedge_side: str
    hedge_amount: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """Portfolio risk management using the Kelly Criterion.

    All monetary amounts are in USD.  All probabilities are floats in [0, 1].
    Market prices represent the YES contract price on a 0-to-1 scale
    (e.g. 0.65 means the market prices YES at 65 ¢).
    """

    def __init__(self, bankroll: float | None = None) -> None:
        self._bankroll = bankroll or config.INITIAL_BALANCE

    # ------------------------------------------------------------------
    # Core Kelly Criterion
    # ------------------------------------------------------------------

    def calculate_kelly_fraction(
        self,
        win_prob: float,
        market_price: float,
    ) -> float:
        """Compute the raw Kelly fraction.

        Parameters
        ----------
        win_prob : float
            Our estimated probability that the event occurs (0..1).
        market_price : float
            Current YES contract price on Polymarket (0..1).
            E.g. 0.60 means "60 cents to buy YES, pays $1 if YES".

        Returns
        -------
        float
            Raw Kelly fraction f*.  Positive → bet YES, negative → bet NO.
            Clamped to [-1, 1].

        Formula
        -------
        For a YES bet:
            b = (1 / market_price) - 1          # net odds
            f* = (p * b - q) / b
               = p - q / b
               = p - (1 - p) * market_price / (1 - market_price)

        For a NO bet (when our prob < market implied):
            Treat as a YES bet on the complementary event.
        """
        if not (0.0 < market_price < 1.0):
            return 0.0
        if not (0.0 < win_prob < 1.0):
            return 0.0

        p = win_prob
        q = 1.0 - p

        # Net odds for a YES bet
        b = (1.0 / market_price) - 1.0
        if b <= 0:
            return 0.0

        # Kelly fraction for YES side
        f_yes = (p * b - q) / b

        return max(-1.0, min(1.0, f_yes))

    # ------------------------------------------------------------------
    # Full position sizing with all guardrails
    # ------------------------------------------------------------------

    def calculate_position(
        self,
        win_prob: float,
        market_price: float,
        bankroll: float | None = None,
        existing_exposure: float = 0.0,
    ) -> KellyResult:
        """Determine the optimal trade given our probability vs market price.

        Applies all safety guardrails:
          1. Half-Kelly scaling (KELLY_FRACTION_CAP)
          2. Max single position size (MAX_SINGLE_POSITION_PCT)
          3. Max total exposure (MAX_TOTAL_EXPOSURE_PCT)
          4. Minimum edge threshold (MIN_EDGE_THRESHOLD)
          5. Probability sanity bounds (MIN/MAX_PROBABILITY)

        Returns a KellyResult with ``tradeable=True`` if the trade passes
        all checks, otherwise ``tradeable=False`` with a ``reject_reason``.
        """
        bankroll = bankroll or self._bankroll
        implied_prob = market_price  # YES price ≈ implied probability

        # Determine direction
        edge = win_prob - implied_prob
        edge_pct = abs(edge)

        if edge >= 0:
            # We think YES is more likely than the market does → buy YES
            side = "YES"
            our_prob = win_prob
            mkt_price = market_price
        else:
            # We think NO is more likely → buy NO (equiv. to selling YES)
            side = "NO"
            our_prob = 1.0 - win_prob
            mkt_price = 1.0 - market_price

        # --- Rejection checks ---

        # 1. Probability sanity
        if win_prob < config.MIN_PROBABILITY or win_prob > config.MAX_PROBABILITY:
            return KellyResult(
                win_prob=win_prob, market_price=market_price,
                implied_prob=implied_prob, edge=edge, edge_pct=edge_pct,
                kelly_fraction=0.0, adjusted_fraction=0.0,
                position_size=0.0, side=side, tradeable=False,
                reject_reason=f"Probability {win_prob:.3f} outside sane bounds "
                              f"[{config.MIN_PROBABILITY}, {config.MAX_PROBABILITY}]",
            )

        # 2. Minimum edge
        if edge_pct < config.MIN_EDGE_THRESHOLD:
            return KellyResult(
                win_prob=win_prob, market_price=market_price,
                implied_prob=implied_prob, edge=edge, edge_pct=edge_pct,
                kelly_fraction=0.0, adjusted_fraction=0.0,
                position_size=0.0, side=side, tradeable=False,
                reject_reason=f"Edge {edge_pct:.1%} below minimum threshold "
                              f"{config.MIN_EDGE_THRESHOLD:.1%}",
            )

        # --- Kelly calculation ---
        raw_kelly = self.calculate_kelly_fraction(our_prob, mkt_price)

        if raw_kelly <= 0:
            return KellyResult(
                win_prob=win_prob, market_price=market_price,
                implied_prob=implied_prob, edge=edge, edge_pct=edge_pct,
                kelly_fraction=raw_kelly, adjusted_fraction=0.0,
                position_size=0.0, side=side, tradeable=False,
                reject_reason="Negative Kelly fraction — no edge",
            )

        # 3. Half-Kelly safety scaling
        adjusted = raw_kelly * config.KELLY_FRACTION_CAP

        # 4. Cap at max single position size
        adjusted = min(adjusted, config.MAX_SINGLE_POSITION_PCT)

        # 5. Cap at remaining exposure room
        remaining_room = max(
            0.0,
            config.MAX_TOTAL_EXPOSURE_PCT - (existing_exposure / bankroll),
        )
        adjusted = min(adjusted, remaining_room)

        position_size = round(adjusted * bankroll, 2)

        if position_size < 1.0:
            return KellyResult(
                win_prob=win_prob, market_price=market_price,
                implied_prob=implied_prob, edge=edge, edge_pct=edge_pct,
                kelly_fraction=raw_kelly, adjusted_fraction=adjusted,
                position_size=0.0, side=side, tradeable=False,
                reject_reason="Position size below $1 minimum",
            )

        return KellyResult(
            win_prob=win_prob, market_price=market_price,
            implied_prob=implied_prob, edge=edge, edge_pct=edge_pct,
            kelly_fraction=round(raw_kelly, 6),
            adjusted_fraction=round(adjusted, 6),
            position_size=position_size,
            side=side, tradeable=True,
        )

    # ------------------------------------------------------------------
    # Edge analysis
    # ------------------------------------------------------------------

    def evaluate_edge(
        self,
        our_prob: float,
        market_prob: float,
    ) -> dict:
        """Compare our probability estimate against the market.

        Returns a dict with edge metrics and a human-readable assessment.
        """
        edge = our_prob - market_prob
        edge_pct = abs(edge)

        if edge > 0:
            direction = "YES underpriced"
            side = "YES"
        elif edge < 0:
            direction = "NO underpriced"
            side = "NO"
        else:
            direction = "fair value"
            side = "NONE"

        # Strength assessment
        if edge_pct >= 0.20:
            strength = "STRONG"
        elif edge_pct >= 0.10:
            strength = "MODERATE"
        elif edge_pct >= config.MIN_EDGE_THRESHOLD:
            strength = "WEAK"
        else:
            strength = "INSUFFICIENT"

        return {
            "our_probability": round(our_prob, 4),
            "market_probability": round(market_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": round(edge_pct, 4),
            "direction": direction,
            "recommended_side": side,
            "strength": strength,
            "tradeable": edge_pct >= config.MIN_EDGE_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Portfolio risk assessment
    # ------------------------------------------------------------------

    def check_portfolio_risk(
        self,
        positions: list[dict],
        bankroll: float | None = None,
    ) -> PortfolioRisk:
        """Analyze current portfolio risk exposure.

        Parameters
        ----------
        positions : list[dict]
            Each dict must have at least: ``amount``, ``city``, ``market_id``.
        bankroll : float, optional
            Current bankroll.  Falls back to ``self._bankroll``.
        """
        bankroll = bankroll or self._bankroll

        if not positions:
            return PortfolioRisk(
                bankroll=bankroll,
                total_exposure=0.0,
                exposure_pct=0.0,
                num_positions=0,
                max_single_exposure=0.0,
                max_single_pct=0.0,
                within_limits=True,
            )

        total_exposure = sum(abs(p.get("amount", 0)) for p in positions)
        exposure_pct = total_exposure / bankroll if bankroll > 0 else 0

        city_exposure: dict[str, float] = {}
        max_single = 0.0
        for p in positions:
            amt = abs(p.get("amount", 0))
            city = p.get("city", "unknown")
            city_exposure[city] = city_exposure.get(city, 0) + amt
            if amt > max_single:
                max_single = amt

        max_single_pct = max_single / bankroll if bankroll > 0 else 0

        warnings: list[str] = []
        within_limits = True

        if exposure_pct > config.MAX_TOTAL_EXPOSURE_PCT:
            warnings.append(
                f"Total exposure {exposure_pct:.1%} exceeds limit "
                f"{config.MAX_TOTAL_EXPOSURE_PCT:.1%}"
            )
            within_limits = False

        if max_single_pct > config.MAX_SINGLE_POSITION_PCT:
            warnings.append(
                f"Max single position {max_single_pct:.1%} exceeds limit "
                f"{config.MAX_SINGLE_POSITION_PCT:.1%}"
            )
            within_limits = False

        # Check per-city concentration (warn if >20% of total in one city)
        for city, amt in city_exposure.items():
            city_pct = amt / bankroll if bankroll > 0 else 0
            if city_pct > 0.20:
                warnings.append(
                    f"City '{city}' concentration {city_pct:.1%} exceeds 20%"
                )

        return PortfolioRisk(
            bankroll=bankroll,
            total_exposure=round(total_exposure, 2),
            exposure_pct=round(exposure_pct, 4),
            num_positions=len(positions),
            max_single_exposure=round(max_single, 2),
            max_single_pct=round(max_single_pct, 4),
            city_exposure={k: round(v, 2) for k, v in city_exposure.items()},
            within_limits=within_limits,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Hedging suggestions
    # ------------------------------------------------------------------

    def suggest_hedges(
        self,
        positions: list[dict],
    ) -> list[HedgeSuggestion]:
        """Identify potential hedge trades to reduce concentrated risk.

        Simple heuristic: if a city has >15% bankroll exposure in one
        direction, suggest a partial offset in the opposite direction.
        """
        city_sides: dict[str, dict] = {}
        for p in positions:
            city = p.get("city", "unknown")
            side = p.get("side", "YES")
            amt = abs(p.get("amount", 0))
            mid = p.get("market_id", "")

            if city not in city_sides:
                city_sides[city] = {"YES": 0.0, "NO": 0.0, "market_ids": []}
            city_sides[city][side] += amt
            if mid not in city_sides[city]["market_ids"]:
                city_sides[city]["market_ids"].append(mid)

        suggestions: list[HedgeSuggestion] = []
        for city, data in city_sides.items():
            yes_exp = data["YES"]
            no_exp = data["NO"]
            net = yes_exp - no_exp

            threshold = self._bankroll * 0.15

            if abs(net) > threshold:
                dominant_side = "YES" if net > 0 else "NO"
                hedge_side = "NO" if net > 0 else "YES"
                hedge_amount = round(abs(net) * 0.3, 2)  # hedge 30% of imbalance

                suggestions.append(HedgeSuggestion(
                    market_id=data["market_ids"][0] if data["market_ids"] else "",
                    city=city,
                    current_side=dominant_side,
                    hedge_side=hedge_side,
                    hedge_amount=hedge_amount,
                    reason=(
                        f"Net {dominant_side} exposure ${abs(net):.0f} in {city} "
                        f"exceeds ${threshold:.0f} threshold. "
                        f"Suggest {hedge_side} ${hedge_amount:.0f} to offset."
                    ),
                ))

        return suggestions

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def get_risk_report(
        self,
        positions: list[dict],
        bankroll: float | None = None,
    ) -> dict:
        """Generate a comprehensive risk report."""
        bankroll = bankroll or self._bankroll
        risk = self.check_portfolio_risk(positions, bankroll)
        hedges = self.suggest_hedges(positions)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_risk": risk.to_dict(),
            "hedge_suggestions": [h.to_dict() for h in hedges],
            "summary": {
                "status": "HEALTHY" if risk.within_limits else "AT_RISK",
                "total_deployed": risk.total_exposure,
                "available_capital": round(bankroll - risk.total_exposure, 2),
                "num_open_positions": risk.num_positions,
                "num_hedge_suggestions": len(hedges),
            },
        }

    # ------------------------------------------------------------------
    # Bankroll management
    # ------------------------------------------------------------------

    @property
    def bankroll(self) -> float:
        return self._bankroll

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update the tracked bankroll after trades resolve."""
        self._bankroll = new_bankroll
        logger.info("Bankroll updated to $%.2f", new_bankroll)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rm = RiskManager(bankroll=10000)

    print("=" * 60)
    print("PolyWeather Risk Manager — Test Suite")
    print("=" * 60)

    # Test 1: Kelly fraction
    print("\n─── Test 1: Raw Kelly Fraction ───")
    for p, mp in [(0.7, 0.5), (0.6, 0.5), (0.55, 0.5), (0.5, 0.5), (0.4, 0.5)]:
        f = rm.calculate_kelly_fraction(p, mp)
        print(f"  P(win)={p:.2f}, market={mp:.2f} → Kelly f*={f:.4f}")

    # Test 2: Full position sizing
    print("\n─── Test 2: Position Sizing with Guardrails ───")
    scenarios = [
        (0.70, 0.50, "Strong YES edge"),
        (0.30, 0.50, "Strong NO edge"),
        (0.52, 0.50, "Weak edge (below threshold)"),
        (0.65, 0.55, "Moderate YES edge"),
        (0.80, 0.40, "Very strong edge"),
        (0.99, 0.50, "Extreme probability"),
    ]
    for p, mp, desc in scenarios:
        result = rm.calculate_position(p, mp)
        status = "✓ TRADE" if result.tradeable else "✗ SKIP"
        print(f"  {status} | {desc}")
        print(f"       P={p}, Mkt={mp}, Side={result.side}, "
              f"Size=${result.position_size:.0f}, "
              f"Edge={result.edge_pct:.1%}, "
              f"Kelly={result.kelly_fraction:.4f}")
        if result.reject_reason:
            print(f"       Reason: {result.reject_reason}")

    # Test 3: Edge analysis
    print("\n─── Test 3: Edge Analysis ───")
    for our, mkt in [(0.65, 0.50), (0.40, 0.60), (0.52, 0.50)]:
        edge = rm.evaluate_edge(our, mkt)
        print(f"  Ours={our}, Mkt={mkt} → {edge['strength']} "
              f"({edge['direction']}, edge={edge['edge_pct']:.1%})")

    # Test 4: Portfolio risk
    print("\n─── Test 4: Portfolio Risk Check ───")
    positions = [
        {"market_id": "m1", "city": "new_york", "side": "YES", "amount": 800},
        {"market_id": "m2", "city": "london", "side": "YES", "amount": 600},
        {"market_id": "m3", "city": "new_york", "side": "NO", "amount": 400},
        {"market_id": "m4", "city": "tokyo", "side": "YES", "amount": 300},
    ]
    risk = rm.check_portfolio_risk(positions)
    print(f"  Exposure: ${risk.total_exposure} ({risk.exposure_pct:.1%} of bankroll)")
    print(f"  Within limits: {risk.within_limits}")
    print(f"  City exposure: {risk.city_exposure}")
    if risk.warnings:
        for w in risk.warnings:
            print(f"  ⚠ {w}")

    # Test 5: Hedge suggestions
    print("\n─── Test 5: Hedge Suggestions ───")
    heavy_positions = [
        {"market_id": "m1", "city": "new_york", "side": "YES", "amount": 2000},
        {"market_id": "m2", "city": "london", "side": "YES", "amount": 500},
    ]
    hedges = rm.suggest_hedges(heavy_positions)
    for h in hedges:
        print(f"  💡 {h.reason}")
    if not hedges:
        print("  No hedges needed")
