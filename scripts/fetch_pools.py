#!/usr/bin/env python3
"""
Fetch pools from all networks.

Usage:
    python scripts/fetch_pools.py [--networks ethereum,arbitrum] [--min-tvl 50000]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import init_db
from src.data.pools import load_pools
from src.utils.helpers import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Fetch pools from subgraphs")
    parser.add_argument(
        "--networks",
        type=str,
        default=None,
        help="Comma-separated list of networks (default: all enabled)",
    )
    parser.add_argument(
        "--min-tvl",
        type=float,
        default=None,
        help="Minimum TVL in USD (default: from settings)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    # Initialize database
    init_db()
    
    # Parse networks
    networks = None
    if args.networks:
        networks = [n.strip() for n in args.networks.split(",")]
    
    # Load pools
    print(f"Loading pools (networks={networks or 'all'}, min_tvl={args.min_tvl or 'default'})...")
    
    results = load_pools(networks=networks, min_tvl=args.min_tvl)
    
    print("\nResults:")
    total = 0
    for network, count in results.items():
        print(f"  {network}: {count} pools")
        total += count
    
    print(f"\nTotal: {total} pools")


if __name__ == "__main__":
    main()
