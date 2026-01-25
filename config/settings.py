"""
Main settings for Revert LP Strategy.

Settings can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DatabaseSettings:
    """Database connection settings."""
    url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", 
        "sqlite:///./data/revert_lp.db"
    ))
    echo: bool = False  # Log SQL queries


@dataclass
class TelegramSettings:
    """Telegram bot settings for notifications."""
    bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    enabled: bool = field(default_factory=lambda: bool(os.getenv("TELEGRAM_BOT_TOKEN")))


@dataclass
class PoolFilterSettings:
    """Filters for pool selection."""
    # Minimum TVL in USD to consider a pool
    min_tvl_usd: float = 50_000.0
    
    # Minimum 24h volume in USD
    min_volume_24h_usd: float = 10_000.0
    
    # Minimum pool age in days (for new token analysis)
    min_pool_age_days: int = 1
    
    # Maximum pool age in days for "new pool" classification
    max_new_pool_age_days: int = 30


@dataclass 
class CapitalFlowSettings:
    """Settings for capital flow / large outflow detection (Module 3)."""
    # Absolute threshold: alert if outflow exceeds this USD amount
    large_outflow_usd: float = 100_000.0
    
    # Relative threshold: alert if outflow exceeds this % of pool TVL
    large_outflow_tvl_percent: float = 5.0
    
    # Time window for aggregating outflows (hours)
    outflow_window_hours: int = 1
    
    # Minimum number of historical data points for percentile calculation
    min_history_for_percentile: int = 100


@dataclass
class FlowPriceSettings:
    """Settings for flow vs price correlation (Module 2)."""
    # Time windows to analyze (hours)
    flow_windows_hours: List[int] = field(default_factory=lambda: [1, 4, 24])
    
    # Price change windows after flow (hours)
    price_lag_hours: List[int] = field(default_factory=lambda: [1, 4, 24])
    
    # Percentile threshold for "significant" flow
    significant_flow_percentile: float = 95.0
    
    # Minimum correlation to consider flow-price relationship valid
    min_correlation: float = 0.3


@dataclass
class OwnerAnalysisSettings:
    """Settings for LP owner analysis (Module 4)."""
    # Minimum number of positions to include owner in analysis
    min_positions: int = 5
    
    # Minimum position age to consider "closed" (days)
    min_position_age_days: int = 7
    
    # Number of top owners to track
    top_owners_count: int = 100
    
    # Include smart contracts in analysis (separately from EOA)
    include_contracts: bool = True


@dataclass
class NewTokenSettings:
    """Settings for new token/pool analysis (Module 1)."""
    # Maximum age to consider a pool "new" (days)
    max_age_days: int = 30
    
    # Minimum TVL for new pool to be considered valid
    min_tvl_usd: float = 50_000.0
    
    # Minimum number of positions to analyze
    min_positions_for_analysis: int = 10
    
    # Range width buckets for analysis (% from current price)
    range_width_buckets: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0])


@dataclass
class RiskManagementSettings:
    """Risk management rules."""
    # Maximum % of capital per single position
    max_position_size_percent: float = 10.0
    
    # Maximum number of positions per token
    max_positions_per_token: int = 3
    
    # Maximum number of positions per network
    max_positions_per_network: int = 10
    
    # Portfolio drawdown limit (%) - pause strategy if exceeded
    max_drawdown_percent: float = 20.0


@dataclass
class DataCollectionSettings:
    """Settings for data collection and storage."""
    # How often to fetch new data (minutes)
    fetch_interval_minutes: int = 5
    
    # How many days of history to maintain
    history_days: int = 365
    
    # Batch size for subgraph queries
    batch_size: int = 1000
    
    # Rate limit: max requests per minute per subgraph
    max_requests_per_minute: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


@dataclass
class Settings:
    """Main settings container."""
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    pool_filter: PoolFilterSettings = field(default_factory=PoolFilterSettings)
    capital_flow: CapitalFlowSettings = field(default_factory=CapitalFlowSettings)
    flow_price: FlowPriceSettings = field(default_factory=FlowPriceSettings)
    owner_analysis: OwnerAnalysisSettings = field(default_factory=OwnerAnalysisSettings)
    new_token: NewTokenSettings = field(default_factory=NewTokenSettings)
    risk: RiskManagementSettings = field(default_factory=RiskManagementSettings)
    data_collection: DataCollectionSettings = field(default_factory=DataCollectionSettings)
    
    # Selected networks to analyze (empty = all enabled)
    active_networks: List[str] = field(default_factory=list)
    
    # Debug mode
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() == "true")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
