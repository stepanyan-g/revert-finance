"""
SQLAlchemy models for storing LP strategy data.

Tables:
- tokens: ERC20 token metadata
- pools: Liquidity pools (Uniswap v2/v3, etc.)
- swaps: Individual swap transactions
- positions: LP positions (Uniswap v3 NFTs)
- position_events: Mint/Burn/Collect events for positions
- owners: Wallet addresses that own positions
- owner_stats: Aggregated statistics per owner
- price_snapshots: Historical token prices
- signals: Generated alerts/signals
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    BigInteger,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Token Model
# =============================================================================

class Token(Base):
    """ERC20 token metadata."""
    __tablename__ = "tokens"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Unique identifier: network + address
    network = Column(String(50), nullable=False, index=True)
    address = Column(String(42), nullable=False, index=True)
    
    # Token metadata
    symbol = Column(String(50), nullable=True)
    name = Column(String(200), nullable=True)
    decimals = Column(Integer, nullable=False, default=18)
    
    # Risk indicators
    is_verified = Column(Boolean, default=False)
    is_honeypot = Column(Boolean, default=False)
    liquidity_locked = Column(Boolean, default=False)
    
    # Timestamps
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("network", "address", name="uix_token_network_address"),
        Index("ix_token_symbol", "symbol"),
    )


# =============================================================================
# Pool Model
# =============================================================================

class Pool(Base):
    """Liquidity pool (Uniswap v2/v3, Sushiswap, etc.)."""
    __tablename__ = "pools"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Unique identifier: network + dex + address
    network = Column(String(50), nullable=False, index=True)
    dex = Column(String(50), nullable=False, index=True)  # uniswap_v3, sushiswap, etc.
    address = Column(String(42), nullable=False, index=True)
    
    # Pool tokens
    token0_id = Column(Integer, ForeignKey("tokens.id"), nullable=True)
    token1_id = Column(Integer, ForeignKey("tokens.id"), nullable=True)
    token0_address = Column(String(42), nullable=False)
    token1_address = Column(String(42), nullable=False)
    token0_symbol = Column(String(50), nullable=True)
    token1_symbol = Column(String(50), nullable=True)
    
    # Pool parameters (v3 specific)
    fee_tier = Column(Integer, nullable=True)  # 100, 500, 3000, 10000 (bps)
    tick_spacing = Column(Integer, nullable=True)
    
    # Current state
    tvl_usd = Column(Numeric(30, 2), default=0)
    volume_24h_usd = Column(Numeric(30, 2), default=0)
    volume_7d_usd = Column(Numeric(30, 2), default=0)
    fee_24h_usd = Column(Numeric(30, 2), default=0)
    
    # Current price
    current_tick = Column(Integer, nullable=True)
    sqrt_price_x96 = Column(String(100), nullable=True)
    token0_price = Column(Numeric(30, 18), nullable=True)
    token1_price = Column(Numeric(30, 18), nullable=True)
    
    # Pool creation
    created_at_block = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, nullable=True)
    
    # Tracking
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    token0 = relationship("Token", foreign_keys=[token0_id])
    token1 = relationship("Token", foreign_keys=[token1_id])
    
    __table_args__ = (
        UniqueConstraint("network", "dex", "address", name="uix_pool_network_dex_address"),
        Index("ix_pool_tvl", "tvl_usd"),
        Index("ix_pool_volume", "volume_24h_usd"),
        Index("ix_pool_created", "created_at"),
    )


# =============================================================================
# Swap Model
# =============================================================================

class Swap(Base):
    """Individual swap transaction."""
    __tablename__ = "swaps"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference to pool
    pool_id = Column(Integer, ForeignKey("pools.id"), nullable=False, index=True)
    network = Column(String(50), nullable=False, index=True)
    
    # Transaction details
    tx_hash = Column(String(66), nullable=False, index=True)
    block_number = Column(BigInteger, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Swap details
    sender = Column(String(42), nullable=False, index=True)
    recipient = Column(String(42), nullable=True)
    
    # Amounts (can be negative for direction)
    amount0 = Column(Numeric(40, 0), nullable=False)  # Raw amount (with decimals)
    amount1 = Column(Numeric(40, 0), nullable=False)
    amount_usd = Column(Numeric(30, 2), nullable=True)
    
    # Price at swap
    sqrt_price_x96 = Column(String(100), nullable=True)
    tick = Column(Integer, nullable=True)
    
    # Direction: 'buy' (token0 bought) or 'sell' (token0 sold)
    direction = Column(String(10), nullable=True)
    
    # Gas
    gas_used = Column(BigInteger, nullable=True)
    gas_price = Column(BigInteger, nullable=True)  # in wei
    gas_cost_usd = Column(Numeric(20, 6), nullable=True)
    
    # MEV detection
    is_mev = Column(Boolean, default=False)
    is_arbitrage = Column(Boolean, default=False)
    
    # Relationships
    pool = relationship("Pool")
    
    __table_args__ = (
        Index("ix_swap_pool_time", "pool_id", "timestamp"),
        Index("ix_swap_sender", "sender"),
        Index("ix_swap_amount", "amount_usd"),
    )


# =============================================================================
# Position Model (Uniswap v3 LP NFT)
# =============================================================================

class Position(Base):
    """LP position (Uniswap v3 NFT)."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Position identifier
    network = Column(String(50), nullable=False, index=True)
    dex = Column(String(50), nullable=False)
    token_id = Column(BigInteger, nullable=False)  # NFT token ID
    
    # Reference to pool
    pool_id = Column(Integer, ForeignKey("pools.id"), nullable=True, index=True)
    pool_address = Column(String(42), nullable=False)
    
    # Owner
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=True, index=True)
    owner_address = Column(String(42), nullable=False, index=True)
    
    # Position range
    tick_lower = Column(Integer, nullable=False)
    tick_upper = Column(Integer, nullable=False)
    
    # Liquidity
    liquidity = Column(Numeric(40, 0), nullable=False, default=0)
    
    # Deposited amounts (cumulative)
    deposited_token0 = Column(Numeric(40, 0), nullable=False, default=0)
    deposited_token1 = Column(Numeric(40, 0), nullable=False, default=0)
    deposited_usd = Column(Numeric(30, 2), nullable=True)
    
    # Withdrawn amounts (cumulative)
    withdrawn_token0 = Column(Numeric(40, 0), nullable=False, default=0)
    withdrawn_token1 = Column(Numeric(40, 0), nullable=False, default=0)
    withdrawn_usd = Column(Numeric(30, 2), nullable=True)
    
    # Collected fees (cumulative)
    collected_fees_token0 = Column(Numeric(40, 0), nullable=False, default=0)
    collected_fees_token1 = Column(Numeric(40, 0), nullable=False, default=0)
    collected_fees_usd = Column(Numeric(30, 2), nullable=True)
    
    # Current value (snapshot)
    current_value_usd = Column(Numeric(30, 2), nullable=True)
    
    # PnL calculation
    total_gas_cost_usd = Column(Numeric(20, 6), nullable=True, default=0)
    realized_pnl_usd = Column(Numeric(30, 2), nullable=True)
    unrealized_pnl_usd = Column(Numeric(30, 2), nullable=True)
    
    # Status
    is_closed = Column(Boolean, default=False)
    is_in_range = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pool = relationship("Pool")
    owner = relationship("Owner", back_populates="positions")
    events = relationship("PositionEvent", back_populates="position")
    
    __table_args__ = (
        UniqueConstraint("network", "dex", "token_id", name="uix_position_network_dex_token"),
        Index("ix_position_owner", "owner_address"),
        Index("ix_position_pool", "pool_id"),
        Index("ix_position_created", "created_at"),
    )


# =============================================================================
# Position Event Model
# =============================================================================

class PositionEvent(Base):
    """Events for LP positions: Mint, IncreaseLiquidity, DecreaseLiquidity, Collect."""
    __tablename__ = "position_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference to position
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False, index=True)
    network = Column(String(50), nullable=False)
    
    # Event type
    event_type = Column(String(50), nullable=False)  # mint, increase, decrease, collect
    
    # Transaction
    tx_hash = Column(String(66), nullable=False, index=True)
    block_number = Column(BigInteger, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Amounts
    amount0 = Column(Numeric(40, 0), nullable=True)
    amount1 = Column(Numeric(40, 0), nullable=True)
    amount_usd = Column(Numeric(30, 2), nullable=True)
    liquidity_delta = Column(Numeric(40, 0), nullable=True)
    
    # Gas
    gas_used = Column(BigInteger, nullable=True)
    gas_price = Column(BigInteger, nullable=True)
    gas_cost_usd = Column(Numeric(20, 6), nullable=True)
    
    # Relationships
    position = relationship("Position", back_populates="events")
    
    __table_args__ = (
        Index("ix_event_position_time", "position_id", "timestamp"),
        Index("ix_event_type", "event_type"),
    )


# =============================================================================
# Owner Model
# =============================================================================

class Owner(Base):
    """Wallet address that owns LP positions."""
    __tablename__ = "owners"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Address
    address = Column(String(42), nullable=False, unique=True, index=True)
    
    # Type
    is_contract = Column(Boolean, default=False)
    contract_name = Column(String(200), nullable=True)  # e.g., "Uniswap V3: Positions NFT"
    
    # Labels
    is_mev_bot = Column(Boolean, default=False)
    is_known_whale = Column(Boolean, default=False)
    label = Column(String(200), nullable=True)  # Custom label
    
    # First seen
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = relationship("Position", back_populates="owner")
    stats = relationship("OwnerStats", back_populates="owner")


# =============================================================================
# Owner Stats Model
# =============================================================================

class OwnerStats(Base):
    """Aggregated statistics per owner (per network or global)."""
    __tablename__ = "owner_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference to owner
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False, index=True)
    
    # Scope (network or 'global')
    network = Column(String(50), nullable=False, default="global")
    
    # Position counts
    total_positions = Column(Integer, default=0)
    open_positions = Column(Integer, default=0)
    closed_positions = Column(Integer, default=0)
    
    # Success metrics
    profitable_positions = Column(Integer, default=0)
    losing_positions = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)  # profitable / closed
    
    # PnL
    total_deposited_usd = Column(Numeric(30, 2), default=0)
    total_withdrawn_usd = Column(Numeric(30, 2), default=0)
    total_fees_collected_usd = Column(Numeric(30, 2), default=0)
    total_gas_cost_usd = Column(Numeric(20, 6), default=0)
    total_pnl_usd = Column(Numeric(30, 2), default=0)
    avg_pnl_per_position_usd = Column(Numeric(30, 2), nullable=True)
    
    # Risk-adjusted metrics
    pnl_std_dev = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    
    # Preferences (most common)
    favorite_pools = Column(Text, nullable=True)  # JSON array of pool addresses
    avg_holding_period_days = Column(Float, nullable=True)
    avg_range_width_percent = Column(Float, nullable=True)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("Owner", back_populates="stats")
    
    __table_args__ = (
        UniqueConstraint("owner_id", "network", name="uix_owner_stats_owner_network"),
        Index("ix_owner_stats_pnl", "total_pnl_usd"),
        Index("ix_owner_stats_win_rate", "win_rate"),
    )


# =============================================================================
# Price Snapshot Model
# =============================================================================

class PriceSnapshot(Base):
    """Historical token prices (for PnL calculation)."""
    __tablename__ = "price_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Token reference
    token_address = Column(String(42), nullable=False, index=True)
    network = Column(String(50), nullable=False, index=True)
    
    # Price
    price_usd = Column(Numeric(30, 18), nullable=False)
    
    # Source
    source = Column(String(50), nullable=True)  # dex, chainlink, coingecko
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    block_number = Column(BigInteger, nullable=True)
    
    __table_args__ = (
        Index("ix_price_token_time", "token_address", "network", "timestamp"),
    )


# =============================================================================
# Signal Model
# =============================================================================

class WatchedOwner(Base):
    """Owners being monitored for activity."""
    __tablename__ = "watched_owners"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference to owner
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False, index=True)
    owner_address = Column(String(42), nullable=False, index=True)
    
    # User note
    note = Column(String(500), nullable=True)
    
    # Settings
    notify_new_position = Column(Boolean, default=True)
    notify_close_position = Column(Boolean, default=True)
    notify_telegram = Column(Boolean, default=True)
    
    # Last known state (for detecting changes)
    last_position_count = Column(Integer, default=0)
    last_checked_at = Column(DateTime, nullable=True)
    
    # Timestamps
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("Owner")
    
    __table_args__ = (
        UniqueConstraint("owner_address", name="uix_watched_owner_address"),
    )


class Signal(Base):
    """Generated alerts and signals."""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Signal type
    signal_type = Column(String(50), nullable=False, index=True)
    # Types: large_outflow, large_inflow, new_pool_opportunity, 
    #        top_owner_entry, price_correlation_alert
    
    # Severity
    severity = Column(String(20), nullable=False, default="info")  # info, warning, critical
    
    # Context
    network = Column(String(50), nullable=True)
    pool_address = Column(String(42), nullable=True)
    token_address = Column(String(42), nullable=True)
    owner_address = Column(String(42), nullable=True)
    
    # Signal details
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(Text, nullable=True)  # JSON with additional data
    
    # Value metrics
    amount_usd = Column(Numeric(30, 2), nullable=True)
    percent_change = Column(Float, nullable=True)
    
    # Status
    is_sent = Column(Boolean, default=False)
    sent_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_signal_type_time", "signal_type", "created_at"),
    )
