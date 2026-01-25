#!/usr/bin/env python3
"""
Analyze LP owners and find top performers.

Usage:
    python scripts/analyze_owners.py [--top 100] [--networks ethereum,arbitrum]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import init_db
from src.analytics.owners import OwnerAnalyzer, get_top_lp_owners
from src.utils.helpers import setup_logging, format_usd


def main():
    parser = argparse.ArgumentParser(description="Analyze LP owners")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top owners to show (default: 20)",
    )
    parser.add_argument(
        "--networks",
        type=str,
        default=None,
        help="Comma-separated list of networks (default: all)",
    )
    parser.add_argument(
        "--min-positions",
        type=int,
        default=5,
        help="Minimum positions for inclusion (default: 5)",
    )
    parser.add_argument(
        "--order-by",
        type=str,
        choices=["pnl", "win_rate", "positions"],
        default="pnl",
        help="Sort by metric (default: pnl)",
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Show success patterns analysis",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save stats to database",
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
    
    analyzer = OwnerAnalyzer()
    
    # Save stats if requested
    if args.save:
        print("Calculating and saving owner stats...")
        count = analyzer.save_owner_stats(networks=networks, min_positions=args.min_positions)
        print(f"Saved stats for {count} owners")
    
    # Get top owners
    print(f"\nTop {args.top} LP owners (sorted by {args.order_by}):")
    print("=" * 80)
    
    top_owners = get_top_lp_owners(limit=args.top, networks=networks)
    
    if not top_owners:
        print("No owners found. Make sure to load position data first.")
        return
    
    for i, owner in enumerate(top_owners, 1):
        pnl_sign = "+" if owner.total_pnl_usd >= 0 else ""
        print(f"\n{i:3}. {owner.address[:10]}...{owner.address[-8:]}")
        print(f"     PnL: {pnl_sign}{format_usd(owner.total_pnl_usd)}")
        print(f"     Positions: {owner.total_positions} (open: {owner.open_positions}, closed: {owner.closed_positions})")
        print(f"     Win rate: {owner.win_rate*100:.1f}% ({owner.profitable_positions}/{owner.closed_positions})")
        print(f"     Avg holding: {owner.avg_holding_days:.1f} days")
        print(f"     Networks: {', '.join(owner.favorite_networks[:3])}")
    
    # Show patterns if requested
    if args.patterns:
        print("\n" + "=" * 80)
        print("SUCCESS PATTERNS ANALYSIS")
        print("=" * 80)
        
        patterns = analyzer.extract_success_patterns(top_count=50, networks=networks)
        
        if "error" in patterns:
            print(f"Error: {patterns['error']}")
            return
        
        print(f"\nSample size: {patterns['sample_size']} top performers")
        print(f"\nRecommended networks: {', '.join(patterns['recommended_networks'])}")
        print(f"Average holding period: {patterns['avg_holding_period_days']} days")
        print(f"Average range width: {patterns['avg_range_width_percent']:.1f}%")
        print(f"Average win rate: {patterns['avg_win_rate']*100:.1f}%")
        
        if patterns.get("top_performer"):
            tp = patterns["top_performer"]
            print(f"\nTop performer:")
            print(f"  Address: {tp['address']}")
            print(f"  PnL: {format_usd(tp['pnl_usd'])}")
            print(f"  Positions: {tp['positions']}")
            print(f"  Win rate: {tp['win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
