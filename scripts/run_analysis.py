#!/usr/bin/env python3
"""
Run capital flow analysis and detect large outflows.

Usage:
    python scripts/run_analysis.py [--hours 1] [--networks ethereum,arbitrum]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import init_db
from src.data.swaps import load_swaps
from src.analytics.capital_flow import detect_large_outflows
from src.signals.telegram import TelegramNotifier
from src.utils.helpers import setup_logging, format_usd


def main():
    parser = argparse.ArgumentParser(description="Run capital flow analysis")
    parser.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Time window in hours (default: 1)",
    )
    parser.add_argument(
        "--networks",
        type=str,
        default=None,
        help="Comma-separated list of networks (default: all)",
    )
    parser.add_argument(
        "--load-swaps",
        action="store_true",
        help="Load recent swaps before analysis",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notifications for alerts",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    # Initialize
    init_db()
    
    # Parse networks
    networks = None
    if args.networks:
        networks = [n.strip() for n in args.networks.split(",")]
    
    # Load swaps if requested
    if args.load_swaps:
        print(f"Loading swaps for last {args.hours * 2}h...")
        swap_count = load_swaps(hours=args.hours * 2, networks=networks)
        print(f"Loaded {swap_count} swaps")
    
    # Run analysis
    print(f"\nAnalyzing capital flows (window={args.hours}h)...")
    alerts = detect_large_outflows(hours=args.hours, networks=networks)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Found {len(alerts)} outflow alerts")
    print(f"{'='*60}")
    
    for alert in alerts:
        severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
        icon = severity_icon.get(alert.severity, "•")
        
        print(f"\n{icon} {alert.token0_symbol}/{alert.token1_symbol} on {alert.network}")
        print(f"   Net outflow: {format_usd(abs(alert.net_flow_usd))}")
        print(f"   TVL: {format_usd(alert.tvl_usd)} ({alert.outflow_percent_of_tvl:.1f}%)")
        print(f"   Swaps: {alert.swap_count}, largest: {format_usd(alert.largest_swap_usd)}")
    
    # Send notifications
    if args.notify and alerts:
        print("\nSending Telegram notifications...")
        notifier = TelegramNotifier()
        notifier.send_pending_signals()


if __name__ == "__main__":
    main()
