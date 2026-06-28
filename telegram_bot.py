"""
PolyWeather — Telegram Notification Bot

Thread-safe Telegram integration for trade alerts, daily summaries,
and risk warnings.  Gracefully degrades to a no-op when no bot token
is configured.

Usage:
    from telegram_bot import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send_trade_alert(trade_info_dict)
    notifier.send_daily_summary(portfolio_dict, pnl_dict)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


class TelegramNotifier:
    """Send trade alerts and summaries to Telegram.

    If ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` are not set in
    ``.env``, all methods silently no-op with an INFO log on first call.
    This lets the rest of the system call notification methods freely
    without guarding every call site.
    """

    def __init__(self) -> None:
        self._token = config.TELEGRAM_BOT_TOKEN
        self._chat_id = config.TELEGRAM_CHAT_ID
        self._enabled = bool(self._token and self._chat_id)
        self._lock = threading.Lock()
        self._warned = False
        self._bot = None

        if self._enabled:
            try:
                import telegram
                self._bot = telegram.Bot(token=self._token)
                logger.info("Telegram notifier enabled (chat_id=%s)", self._chat_id)
            except ImportError:
                logger.warning(
                    "python-telegram-bot not installed — Telegram disabled"
                )
                self._enabled = False
            except Exception as exc:
                logger.warning("Telegram bot init failed: %s", exc)
                self._enabled = False
        else:
            logger.info(
                "Telegram notifier disabled — "
                "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable"
            )

    # ------------------------------------------------------------------
    # Internal send
    # ------------------------------------------------------------------

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat.  Thread-safe.

        Returns True on success, False on failure or disabled.
        """
        if not self._enabled:
            if not self._warned:
                logger.debug("Telegram send skipped — notifier not enabled")
                self._warned = True
            return False

        with self._lock:
            try:
                import asyncio

                async def _do_send():
                    await self._bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode=parse_mode,
                    )

                # Handle both running and non-running event loops
                try:
                    loop = asyncio.get_running_loop()
                    # We're inside an async context — schedule as a task
                    loop.create_task(_do_send())
                except RuntimeError:
                    # No running loop — create one
                    asyncio.run(_do_send())

                logger.debug("Telegram message sent (%d chars)", len(text))
                return True

            except Exception as exc:
                logger.warning("Telegram send failed: %s", exc)
                return False

    # ------------------------------------------------------------------
    # Trade alerts
    # ------------------------------------------------------------------

    def send_trade_alert(self, trade_info: dict) -> bool:
        """Send a formatted trade execution alert.

        Expected keys in *trade_info*:
            id, market_id, city, event_type, side, amount, price,
            shares, status, question (optional)
        """
        city_name = config.CITIES.get(
            trade_info.get("city", ""), {}
        ).get("display_name", trade_info.get("city", "?"))

        side = trade_info.get("side", "?")
        side_emoji = "🟢" if side == "YES" else "🔴"

        text = (
            f"{side_emoji} <b>TRADE EXECUTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 City: <b>{city_name}</b>\n"
            f"📊 Event: {trade_info.get('event_type', '?')}\n"
            f"🎯 Side: <b>{side}</b>\n"
            f"💰 Amount: <b>${trade_info.get('amount', 0):.2f}</b>\n"
            f"💲 Price: {trade_info.get('price', 0):.3f}\n"
            f"📈 Shares: {trade_info.get('shares', 0):.2f}\n"
            f"📋 Status: {trade_info.get('status', '?')}\n"
        )

        question = trade_info.get("question", "")
        if question:
            text += f"❓ {question}\n"

        text += f"\n🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

        logger.info(
            "Trade alert: %s %s $%.2f @ %.3f in %s",
            side, trade_info.get("event_type", "?"),
            trade_info.get("amount", 0), trade_info.get("price", 0),
            city_name,
        )

        return self._send_message(text)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def send_daily_summary(
        self,
        portfolio: dict,
        pnl: dict,
    ) -> bool:
        """Send an end-of-day portfolio and PnL summary.

        Parameters
        ----------
        portfolio : dict
            Output of ``PaperTrader.get_portfolio()``.
        pnl : dict
            Output of ``PaperTrader.get_pnl_summary()``.
        """
        balance = pnl.get("current_balance", 0)
        initial = pnl.get("initial_balance", config.INITIAL_BALANCE)
        total_return = pnl.get("total_return_pct", "0.00%")
        win_rate = pnl.get("win_rate_pct", "0%")

        pnl_value = balance - initial
        pnl_emoji = "📈" if pnl_value >= 0 else "📉"

        text = (
            f"📊 <b>DAILY SUMMARY</b> — PolyWeather\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💼 <b>Portfolio</b>\n"
            f"  Balance: <b>${balance:,.2f}</b>\n"
            f"  Invested: ${portfolio.get('total_invested', 0):,.2f}\n"
            f"  Open positions: {portfolio.get('num_positions', 0)}\n\n"
            f"{pnl_emoji} <b>Performance</b>\n"
            f"  Total return: <b>{total_return}</b>\n"
            f"  P&L: ${pnl_value:+,.2f}\n"
            f"  Win rate: {win_rate}\n"
            f"  Resolved trades: {pnl.get('total_resolved', 0)}\n"
            f"  Best trade: ${pnl.get('best_trade', 0):+,.2f}\n"
            f"  Worst trade: ${pnl.get('worst_trade', 0):+,.2f}\n\n"
        )

        # Active positions
        positions = portfolio.get("positions", [])
        if positions:
            text += "📌 <b>Active Positions</b>\n"
            for pos in positions[:8]:  # Limit to avoid message length issues
                city_name = config.CITIES.get(
                    pos.get("city", ""), {}
                ).get("display_name", pos.get("city", "?"))
                text += (
                    f"  • {city_name} {pos.get('side', '?')} "
                    f"${pos.get('total_amount', 0):.0f} "
                    f"@ {pos.get('avg_price', 0):.3f}\n"
                )

        text += f"\n🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

        logger.info("Daily summary: balance=$%.2f, return=%s", balance, total_return)
        return self._send_message(text)

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    def send_risk_alert(self, message: str) -> bool:
        """Send an urgent risk warning message."""
        text = (
            f"⚠️ <b>RISK ALERT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{message}\n\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        logger.warning("Risk alert sent: %s", message[:100])
        return self._send_message(text)

    # ------------------------------------------------------------------
    # Generic message
    # ------------------------------------------------------------------

    def send_message(self, text: str) -> bool:
        """Send a plain text message (no HTML formatting)."""
        return self._send_message(text, parse_mode=None)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    notifier = TelegramNotifier()

    print("=" * 50)
    print("PolyWeather Telegram Bot — Status Check")
    print("=" * 50)
    print(f"  Enabled: {notifier.is_enabled}")
    print(f"  Token set: {bool(config.TELEGRAM_BOT_TOKEN)}")
    print(f"  Chat ID set: {bool(config.TELEGRAM_CHAT_ID)}")

    if notifier.is_enabled:
        print("\n  Sending test message...")
        ok = notifier.send_message("🧪 PolyWeather test — system online!")
        print(f"  Result: {'✓ Sent' if ok else '✗ Failed'}")
    else:
        print("\n  Telegram not configured — all sends will no-op silently.")
        # Verify no-op doesn't raise
        notifier.send_trade_alert({"side": "YES", "amount": 100, "price": 0.5})
        notifier.send_risk_alert("Test risk warning")
        print("  ✓ No-op methods work without errors")
