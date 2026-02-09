"""
Database connection and session management.
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from .models import Base
from config.settings import get_settings


_engine = None
_SessionLocal = None


def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance and concurrency."""
    cursor = dbapi_connection.cursor()
    # WAL mode allows concurrent reads while writing
    cursor.execute("PRAGMA journal_mode=WAL")
    # Faster synchronous mode (still safe with WAL)
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Use memory for temp tables
    cursor.execute("PRAGMA temp_store=MEMORY")
    # Increase cache size (negative = KB, so -64000 = 64MB)
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.close()


def get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Ensure data directory exists for SQLite
        is_sqlite = settings.database.url.startswith("sqlite")
        if is_sqlite:
            db_path = settings.database.url.replace("sqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite-specific connection args
        connect_args = {}
        if is_sqlite:
            connect_args = {
                "check_same_thread": False,
                "timeout": 30,  # Wait up to 30 seconds for lock
            }
        
        _engine = create_engine(
            settings.database.url,
            echo=settings.database.echo,
            connect_args=connect_args,
            # Pool settings for SQLite
            pool_pre_ping=True,  # Check connection validity before use
        )
        
        # Set SQLite pragmas on connect
        if is_sqlite:
            event.listen(_engine, "connect", _set_sqlite_pragma)
    
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def get_session() -> Session:
    """Create a new database session."""
    SessionLocal = get_session_factory()
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.
    
    Usage:
        with session_scope() as session:
            session.add(obj)
            session.commit()
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        # Always close session to release locks
        session.close()


def init_db() -> None:
    """
    Initialize database: create all tables.
    
    Safe to call multiple times - only creates tables that don't exist.
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {engine.url}")


def drop_all_tables() -> None:
    """
    Drop all tables. USE WITH CAUTION.
    
    Only for development/testing.
    """
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped.")


def reset_db() -> None:
    """
    Reset database: drop all tables and recreate.
    
    USE WITH CAUTION - destroys all data.
    """
    drop_all_tables()
    init_db()
    print("Database reset complete.")


def close_all_connections() -> None:
    """
    Close all database connections and release locks.
    
    Useful when you get "database is locked" errors.
    """
    global _engine, _SessionLocal
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        print("All database connections closed.")
