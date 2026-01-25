"""
Database connection and session management.
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base
from config.settings import get_settings


_engine = None
_SessionLocal = None


def get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Ensure data directory exists for SQLite
        if settings.database.url.startswith("sqlite"):
            db_path = settings.database.url.replace("sqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        _engine = create_engine(
            settings.database.url,
            echo=settings.database.echo,
            # SQLite specific settings
            connect_args={"check_same_thread": False} if "sqlite" in settings.database.url else {},
        )
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
