"""
PolyWeather — Paper Trading Execution Engine

Simulates placing YES/NO orders on Polymarket weather markets.  All
state is persisted in an SQLite database (``trades.db``).

Features:
  - Market and limit orders
  - Portfolio tracking with unrealised PnL
  - Market resolution and realised PnL calculation
  - Trade history with full audit trail
  - Balance management

Usage:
    from execution_engine import PaperTrader
    trader = PaperTrader()

    trader.place_order("sim_new_york_temp_above_90f_2026-06-30",
                       city="new_york", event_type="temp_above_90f",
                       side="YES", amount=500, order_type="MARKET", price=0.65)

    print(trader.get_portfolio())
    print(trader.get_balance())
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    market_id       TEXT    NOT NULL,
    city            TEXT    NOT NULL DEFAULT '',
    event_type      TEXT    NOT NULL DEFAULT '',
    question        TEXT    NOT NULL DEFAULT '',
    side            TEXT    NOT NULL CHECK(side IN ('YES','NO')),
    order_type      TEXT    NOT NULL DEFAULT 'MARKET' CHECK(order_type IN ('MARKET','LIMIT')),
    amount          REAL    NOT NULL,
    price           REAL    NOT NULL,
    shares          REAL    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'OPEN'
                    CHECK(status IN ('OPEN','FILLED','CANCELLED','RESOLVED')),
    resolved_outcome TEXT   DEFAULT NULL,
    pnl             REAL    DEFAULT NULL,
    resolved_at     TEXT    DEFAULT NULL,
    notes           TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    initial_balance REAL    NOT NULL,
    current_balance REAL    NOT NULL,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    market_id       TEXT    NOT NULL,
    yes_price       REAL    NOT NULL,
    no_price        REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_city   ON trades(city);
CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_id);
"""


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Paper trading engine backed by SQLite.

    All monetary amounts are in USD.  Prices are on a 0–1 scale.
    Shares = amount / price.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or config.DB_PATH)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._ensure_portfolio()
        logger.info("PaperTrader initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def _ensure_portfolio(self) -> None:
        """Create the portfolio row if this is a fresh database."""
        row = self._conn.execute("SELECT COUNT(*) FROM portfolio").fetchone()
        if row[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO portfolio (initial_balance, current_balance, "
                "created_at, updated_at) VALUES (?, ?, ?, ?)",
                (config.INITIAL_BALANCE, config.INITIAL_BALANCE, now, now),
            )
            self._conn.commit()
            logger.info(
                "Portfolio created with initial balance $%.2f",
                config.INITIAL_BALANCE,
            )

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return current available cash balance."""
        row = self._conn.execute(
            "SELECT current_balance FROM portfolio ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return float(row["current_balance"]) if row else 0.0

    def _update_balance(self, delta: float) -> float:
        """Adjust balance by delta and return new balance."""
        current = self.get_balance()
        new_bal = round(current + delta, 2)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE portfolio SET current_balance = ?, updated_at = ? "
            "WHERE id = (SELECT MAX(id) FROM portfolio)",
            (new_bal, now),
        )
        self._conn.commit()
        return new_bal

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        market_id: str,
        city: str = "",
        event_type: str = "",
        side: str = "YES",
        amount: float = 0.0,
        order_type: str = "MARKET",
        price: float = 0.50,
        question: str = "",
        notes: str = "",
    ) -> dict:
        """Place a paper trade.

        Parameters
        ----------
        market_id : str
            Unique market identifier.
        city : str
            City key (e.g. "new_york").
        event_type : str
            Event key (e.g. "temp_above_90f").
        side : str
            "YES" or "NO".
        amount : float
            Dollar amount to spend.
        order_type : str
            "MARKET" or "LIMIT".
        price : float
            Execution price (0–1).  For market orders this is the current
            best price.  For limit orders this is the limit price.
        question : str
            Human-readable market question.
        notes : str
            Any extra metadata.

        Returns
        -------
        dict
            Trade record with generated ``id``.
        """
        side = side.upper()
        if side not in ("YES", "NO"):
            raise ValueError(f"Invalid side '{side}', must be YES or NO")

        order_type = order_type.upper()
        if order_type not in ("MARKET", "LIMIT"):
            raise ValueError(f"Invalid order_type '{order_type}'")

        if amount <= 0:
            raise ValueError("Amount must be positive")

        if not (0.01 <= price <= 0.99):
            raise ValueError(f"Price {price} out of range [0.01, 0.99]")

        balance = self.get_balance()
        if amount > balance:
            raise ValueError(
                f"Insufficient balance: ${balance:.2f} available, "
                f"${amount:.2f} required"
            )

        # Calculate shares
        shares = round(amount / price, 4)

        now = datetime.now(timezone.utc).isoformat()

        # For MARKET orders, fill immediately
        status = "FILLED" if order_type == "MARKET" else "OPEN"

        cursor = self._conn.execute(
            """
            INSERT INTO trades
                (timestamp, market_id, city, event_type, question,
                 side, order_type, amount, price, shares, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now, market_id, city, event_type, question,
             side, order_type, amount, price, shares, status, notes),
        )
        trade_id = cursor.lastrowid

        # Deduct from balance
        if status == "FILLED":
            self._update_balance(-amount)

        self._conn.commit()

        trade = {
            "id": trade_id,
            "timestamp": now,
            "market_id": market_id,
            "city": city,
            "event_type": event_type,
            "question": question,
            "side": side,
            "order_type": order_type,
            "amount": amount,
            "price": price,
            "shares": shares,
            "status": status,
            "notes": notes,
        }

        logger.info(
            "📝 Trade #%d: %s %s $%.2f @ %.3f (%s shares) on %s [%s]",
            trade_id, side, order_type, amount, price, shares,
            market_id, status,
        )

        return trade

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_portfolio(self) -> dict:
        """Return full portfolio overview.

        Returns dict with ``balance``, ``positions``, ``total_value``,
        ``unrealised_pnl``, ``total_invested``.
        """
        balance = self.get_balance()

        # Group filled trades by market_id + side
        rows = self._conn.execute(
            """
            SELECT market_id, city, event_type, question, side,
                   SUM(amount) as total_amount,
                   SUM(shares) as total_shares,
                   AVG(price) as avg_price,
                   COUNT(*) as num_trades
            FROM trades
            WHERE status = 'FILLED'
            GROUP BY market_id, side
            ORDER BY market_id
            """
        ).fetchall()

        positions: list[dict] = []
        total_invested = 0.0

        for row in rows:
            pos = {
                "market_id": row["market_id"],
                "city": row["city"],
                "event_type": row["event_type"],
                "question": row["question"],
                "side": row["side"],
                "total_amount": round(row["total_amount"], 2),
                "total_shares": round(row["total_shares"], 4),
                "avg_price": round(row["avg_price"], 4),
                "num_trades": row["num_trades"],
                # Current value is hypothetical — shares × current price
                # Since we don't have live prices here, show cost basis
                "cost_basis": round(row["total_amount"], 2),
            }
            positions.append(pos)
            total_invested += row["total_amount"]

        return {
            "balance": balance,
            "positions": positions,
            "total_invested": round(total_invested, 2),
            "total_value": round(balance + total_invested, 2),
            "num_positions": len(positions),
        }

    def get_open_positions(self) -> list[dict]:
        """Return only currently open (filled, unresolved) positions."""
        rows = self._conn.execute(
            """
            SELECT market_id, city, event_type, question, side,
                   SUM(amount) as total_amount,
                   SUM(shares) as total_shares,
                   AVG(price) as avg_price
            FROM trades
            WHERE status = 'FILLED'
            GROUP BY market_id, side
            """
        ).fetchall()

        return [
            {
                "market_id": r["market_id"],
                "city": r["city"],
                "event_type": r["event_type"],
                "side": r["side"],
                "amount": round(r["total_amount"], 2),
                "shares": round(r["total_shares"], 4),
                "avg_price": round(r["avg_price"], 4),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trade_history(self, limit: int = 100) -> list[dict]:
        """Return recent trade records, newest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Market resolution
    # ------------------------------------------------------------------

    def resolve_market(
        self,
        market_id: str,
        outcome: str,
    ) -> dict:
        """Settle a market and calculate realised PnL.

        Parameters
        ----------
        market_id : str
            The market to resolve.
        outcome : str
            "YES" or "NO" — the actual outcome.

        Returns
        -------
        dict
            Summary with total PnL across all trades in this market.
        """
        outcome = outcome.upper()
        if outcome not in ("YES", "NO"):
            raise ValueError(f"Invalid outcome '{outcome}'")

        rows = self._conn.execute(
            "SELECT * FROM trades WHERE market_id = ? AND status = 'FILLED'",
            (market_id,),
        ).fetchall()

        if not rows:
            return {
                "market_id": market_id,
                "outcome": outcome,
                "trades_resolved": 0,
                "total_pnl": 0.0,
            }

        now = datetime.now(timezone.utc).isoformat()
        total_pnl = 0.0

        for row in rows:
            trade_side = row["side"]
            amount = row["amount"]
            shares = row["shares"]

            if trade_side == outcome:
                # Winner: shares pay out $1 each
                payout = shares * 1.0
                pnl = payout - amount
            else:
                # Loser: shares worth $0
                pnl = -amount

            total_pnl += pnl

            self._conn.execute(
                """
                UPDATE trades
                SET status = 'RESOLVED',
                    resolved_outcome = ?,
                    pnl = ?,
                    resolved_at = ?
                WHERE id = ?
                """,
                (outcome, round(pnl, 2), now, row["id"]),
            )

        # Credit winnings back to balance
        # (losing trades already had their cost deducted at fill time)
        if total_pnl > 0:
            self._update_balance(total_pnl + sum(r["amount"] for r in rows))
        else:
            # Even on net loss, return any winning sub-trade proceeds
            winners_payout = sum(
                r["shares"] for r in rows if r["side"] == outcome
            )
            if winners_payout > 0:
                self._update_balance(winners_payout)

        self._conn.commit()

        logger.info(
            "🏁 Market %s resolved: %s | PnL: $%.2f (%d trades)",
            market_id, outcome, total_pnl, len(rows),
        )

        return {
            "market_id": market_id,
            "outcome": outcome,
            "trades_resolved": len(rows),
            "total_pnl": round(total_pnl, 2),
            "resolved_at": now,
        }

    # ------------------------------------------------------------------
    # PnL summary
    # ------------------------------------------------------------------

    def get_pnl_summary(self) -> dict:
        """Calculate overall trading performance metrics."""
        # Resolved trades
        resolved = self._conn.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as breakeven,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(pnl), 0) as avg_pnl,
                   COALESCE(MAX(pnl), 0) as best_trade,
                   COALESCE(MIN(pnl), 0) as worst_trade
            FROM trades
            WHERE status = 'RESOLVED'
            """
        ).fetchone()

        total = resolved["total"] or 0
        wins = resolved["wins"] or 0
        win_rate = wins / total if total > 0 else 0.0

        # Open positions
        open_count = self._conn.execute(
            "SELECT COUNT(DISTINCT market_id || side) FROM trades WHERE status = 'FILLED'"
        ).fetchone()[0]

        balance = self.get_balance()

        # Fetch initial balance
        initial = self._conn.execute(
            "SELECT initial_balance FROM portfolio ORDER BY id LIMIT 1"
        ).fetchone()
        initial_balance = float(initial["initial_balance"]) if initial else config.INITIAL_BALANCE

        total_return = ((balance - initial_balance) / initial_balance) if initial_balance > 0 else 0

        return {
            "initial_balance": initial_balance,
            "current_balance": balance,
            "total_return": round(total_return, 4),
            "total_return_pct": f"{total_return:.2%}",
            "total_pnl_resolved": round(resolved["total_pnl"], 2),
            "avg_pnl_per_trade": round(resolved["avg_pnl"], 2),
            "best_trade": round(resolved["best_trade"], 2),
            "worst_trade": round(resolved["worst_trade"], 2),
            "total_resolved": total,
            "wins": wins,
            "losses": resolved["losses"] or 0,
            "win_rate": round(win_rate, 4),
            "win_rate_pct": f"{win_rate:.1%}",
            "open_positions": open_count,
        }

    # ------------------------------------------------------------------
    # Market snapshots (for historical price tracking)
    # ------------------------------------------------------------------

    def record_market_snapshot(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        source: str = "",
    ) -> None:
        """Record a point-in-time price snapshot for a market."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO market_snapshots
                (timestamp, market_id, yes_price, no_price, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now, market_id, yes_price, no_price, source),
        )
        self._conn.commit()

    def get_price_history(
        self,
        market_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Return historical price snapshots for a market."""
        rows = self._conn.execute(
            """
            SELECT * FROM market_snapshots
            WHERE market_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (market_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self, confirm: bool = False) -> None:
        """Wipe all data and start fresh.  Requires ``confirm=True``."""
        if not confirm:
            raise ValueError("Pass confirm=True to reset all trading data")

        self._conn.executescript("""
            DELETE FROM trades;
            DELETE FROM portfolio;
            DELETE FROM market_snapshots;
        """)
        self._ensure_portfolio()
        logger.warning("⚠ All trading data has been reset")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    # Use a temp DB for testing so we don't pollute the real one
    test_db = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_trades.db"
    )

    try:
        trader = PaperTrader(db_path=test_db)

        print("=" * 60)
        print("PolyWeather Execution Engine — Test Suite")
        print("=" * 60)

        # Test 1: Check initial balance
        print(f"\n─── Test 1: Initial Balance ───")
        balance = trader.get_balance()
        print(f"  Balance: ${balance:.2f}")
        assert balance == config.INITIAL_BALANCE, f"Expected ${config.INITIAL_BALANCE}"
        print("  ✓ Correct")

        # Test 2: Place trades
        print(f"\n─── Test 2: Place Orders ───")
        t1 = trader.place_order(
            market_id="sim_new_york_temp_above_90f_2026-06-30",
            city="new_york",
            event_type="temp_above_90f",
            side="YES",
            amount=500,
            order_type="MARKET",
            price=0.65,
            question="Will NYC max temp exceed 90°F on June 30?",
        )
        print(f"  Trade #1: {t1['side']} ${t1['amount']} @ {t1['price']} "
              f"= {t1['shares']} shares [{t1['status']}]")

        t2 = trader.place_order(
            market_id="sim_london_precipitation_2026-06-30",
            city="london",
            event_type="precipitation",
            side="NO",
            amount=300,
            order_type="MARKET",
            price=0.40,
            question="Will London have rain on June 30?",
        )
        print(f"  Trade #2: {t2['side']} ${t2['amount']} @ {t2['price']} "
              f"= {t2['shares']} shares [{t2['status']}]")

        t3 = trader.place_order(
            market_id="sim_new_york_temp_above_90f_2026-06-30",
            city="new_york",
            event_type="temp_above_90f",
            side="YES",
            amount=200,
            order_type="MARKET",
            price=0.70,
            question="Will NYC max temp exceed 90°F on June 30?",
        )
        print(f"  Trade #3: {t3['side']} ${t3['amount']} @ {t3['price']} "
              f"= {t3['shares']} shares [{t3['status']}]")

        # Test 3: Portfolio
        print(f"\n─── Test 3: Portfolio ───")
        portfolio = trader.get_portfolio()
        print(f"  Balance: ${portfolio['balance']:.2f}")
        print(f"  Invested: ${portfolio['total_invested']:.2f}")
        print(f"  Positions: {portfolio['num_positions']}")
        for pos in portfolio['positions']:
            print(f"    {pos['market_id']}: {pos['side']} ${pos['total_amount']:.2f} "
                  f"({pos['total_shares']:.2f} shares @ avg {pos['avg_price']:.3f})")

        # Test 4: Trade history
        print(f"\n─── Test 4: Trade History ───")
        history = trader.get_trade_history()
        print(f"  Total trades: {len(history)}")

        # Test 5: Resolve markets
        print(f"\n─── Test 5: Market Resolution ───")
        # NYC actually hit 90°F → YES wins
        result1 = trader.resolve_market(
            "sim_new_york_temp_above_90f_2026-06-30", "YES"
        )
        print(f"  NYC Temp: {result1['outcome']} | PnL: ${result1['total_pnl']:.2f} "
              f"({result1['trades_resolved']} trades)")

        # London had no rain → NO wins
        result2 = trader.resolve_market(
            "sim_london_precipitation_2026-06-30", "NO"
        )
        print(f"  London Rain: {result2['outcome']} | PnL: ${result2['total_pnl']:.2f} "
              f"({result2['trades_resolved']} trades)")

        # Test 6: PnL summary
        print(f"\n─── Test 6: PnL Summary ───")
        pnl = trader.get_pnl_summary()
        print(f"  Balance: ${pnl['current_balance']:.2f}")
        print(f"  Total return: {pnl['total_return_pct']}")
        print(f"  Resolved PnL: ${pnl['total_pnl_resolved']:.2f}")
        print(f"  Win rate: {pnl['win_rate_pct']}")
        print(f"  Best trade: ${pnl['best_trade']:.2f}")
        print(f"  Worst trade: ${pnl['worst_trade']:.2f}")

        trader.close()
        print(f"\n✅ All tests passed!")

    finally:
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"  Cleaned up {test_db}")
