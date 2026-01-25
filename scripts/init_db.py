#!/usr/bin/env python3
"""
Initialize the database.

Creates all tables defined in models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import init_db
from src.utils.helpers import setup_logging


def main():
    """Initialize database."""
    setup_logging()
    
    print("Initializing database...")
    init_db()
    print("Done!")


if __name__ == "__main__":
    main()
