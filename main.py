"""
PolyWeather — Command Line Entry Point

Master controller for the PolyWeather autonomous trading system.

Usage:
  python main.py run       - Run a single trading cycle and exit
  python main.py daemon    - Run the persistent background loop
"""

import argparse
import logging
import sys

import config
from hermes_orchestrator import HermesOrchestrator

# Setup basic console logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def run_single_cycle() -> None:
    """Execute exactly one trading cycle."""
    logger.info("Starting manual single cycle...")
    orchestrator = HermesOrchestrator()
    try:
        report = orchestrator.run_cycle()
        if report.get("errors"):
            logger.warning("Cycle completed with %d errors.", len(report["errors"]))
        else:
            logger.info("Cycle completed successfully.")
    except KeyboardInterrupt:
        logger.info("Cycle aborted by user.")
    except Exception as exc:
        logger.error("Cycle crashed: %s", exc)
        sys.exit(1)


def run_daemon() -> None:
    """Start the persistent autonomous loop."""
    logger.info("Starting Hermes autonomous daemon (interval=%d min)...", config.AGENT_CYCLE_MINUTES)
    orchestrator = HermesOrchestrator()
    try:
        orchestrator.run_daemon()
    except KeyboardInterrupt:
        logger.info("Daemon stopped gracefully.")
    except Exception as exc:
        logger.error("Daemon crashed: %s", exc)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolyWeather Autonomous Trading System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    subparsers.add_parser(
        "run", help="Run a single trading cycle and exit immediately"
    )

    # Command: daemon
    subparsers.add_parser(
        "daemon", help="Run continuously in the background on a timer"
    )

    args = parser.parse_args()

    if args.command == "run":
        run_single_cycle()
    elif args.command == "daemon":
        run_daemon()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
