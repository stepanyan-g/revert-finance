"""
Module 1: New Token Analysis.

Analyzes newly created pools and tokens:
- Monitors new pool creation
- Analyzes LP positions on new pools
- Provides recommendations for range and timing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import func

from config.settings import get_settings
from src.db.models import Pool, Token, Position, Signal
from src.db.database import session_scope
from src.data.pools import PoolLoader


logger = logging.getLogger(__name__)


@dataclass
class NewPoolInfo:
    """Information about a new pool."""
    pool_id: int
    address: str
    network: str
    token0_symbol: str
    token1_symbol: str
    tvl_usd: float
    volume_24h_usd: float
    fee_tier: int
    age_days: float
    created_at: datetime
    
    # Analysis results
    position_count: int = 0
    avg_range_width_percent: float = 0
    avg_holding_days: float = 0
    success_rate: float = 0  # % of positions with positive PnL
    
    # Risk indicators
    is_new_token: bool = False
    token0_is_verified: bool = False
    token1_is_verified: bool = False
    
    # Recommendation
    recommended_range_percent: float = 0
    risk_level: str = "unknown"  # low, medium, high, very_high


@dataclass
class NewTokenAlert:
    """Alert for a new token/pool opportunity."""
    pool: NewPoolInfo
    alert_type: str  # new_pool, high_volume, rapid_growth
    message: str
    
    def to_signal(self) -> dict:
        """Convert to Signal model dict."""
        import json
        return {
            "signal_type": "new_pool_opportunity",
            "severity": "info",
            "network": self.pool.network,
            "pool_address": self.pool.address,
            "title": f"New Pool: {self.pool.token0_symbol}/{self.pool.token1_symbol}",
            "message": self.message,
            "data": json.dumps({
                "pool_id": self.pool.pool_id,
                "tvl_usd": self.pool.tvl_usd,
                "age_days": self.pool.age_days,
                "recommended_range": self.pool.recommended_range_percent,
            }),
            "amount_usd": Decimal(str(self.pool.tvl_usd)),
        }


class NewTokenAnalyzer:
    """
    Analyzes new tokens and pools.
    
    Features:
    - Detect newly created pools
    - Analyze position patterns on new pools
    - Provide range recommendations
    - Risk assessment
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def get_new_pools(
        self,
        max_age_days: Optional[int] = None,
        min_tvl: Optional[float] = None,
        networks: Optional[list[str]] = None,
    ) -> list[NewPoolInfo]:
        """
        Get recently created pools.
        
        Args:
            max_age_days: Maximum pool age (default from settings)
            min_tvl: Minimum TVL (default from settings)
            networks: Networks to include
            
        Returns:
            List of NewPoolInfo objects
        """
        if max_age_days is None:
            max_age_days = self.settings.new_token.max_age_days
        if min_tvl is None:
            min_tvl = self.settings.new_token.min_tvl_usd
        
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        
        new_pools = []
        
        with session_scope() as session:
            query = session.query(Pool).filter(
                Pool.created_at >= cutoff,
                Pool.tvl_usd >= min_tvl,
            )
            
            if networks:
                query = query.filter(Pool.network.in_(networks))
            
            pools = query.order_by(Pool.tvl_usd.desc()).all()
            
            for pool in pools:
                age_days = (datetime.utcnow() - pool.created_at).days if pool.created_at else 0
                
                info = NewPoolInfo(
                    pool_id=pool.id,
                    address=pool.address,
                    network=pool.network,
                    token0_symbol=pool.token0_symbol or "???",
                    token1_symbol=pool.token1_symbol or "???",
                    tvl_usd=float(pool.tvl_usd or 0),
                    volume_24h_usd=float(pool.volume_24h_usd or 0),
                    fee_tier=pool.fee_tier or 0,
                    age_days=age_days,
                    created_at=pool.created_at or datetime.utcnow(),
                )
                
                # Analyze positions on this pool
                self._analyze_pool_positions(session, pool, info)
                
                # Assess risk
                self._assess_risk(session, pool, info)
                
                # Generate recommendation
                self._generate_recommendation(info)
                
                new_pools.append(info)
        
        return new_pools
    
    def _analyze_pool_positions(
        self,
        session: Session,
        pool: Pool,
        info: NewPoolInfo,
    ) -> None:
        """Analyze positions on a pool to extract patterns."""
        positions = session.query(Position).filter(
            Position.pool_id == pool.id
        ).all()
        
        info.position_count = len(positions)
        
        if not positions:
            return
        
        # Calculate average range width
        range_widths = []
        holding_days = []
        profitable = 0
        
        for pos in positions:
            if pos.tick_lower and pos.tick_upper:
                # Approximate range width
                tick_range = pos.tick_upper - pos.tick_lower
                width_percent = tick_range * 0.01  # ~0.01% per tick
                range_widths.append(width_percent)
            
            if pos.created_at:
                end = pos.closed_at or datetime.utcnow()
                days = (end - pos.created_at).days
                holding_days.append(days)
            
            # Check if profitable (simplified)
            fees = float(pos.collected_fees_usd or 0)
            deposited = float(pos.deposited_usd or 0)
            if fees > deposited * 0.01:  # More than 1% in fees
                profitable += 1
        
        if range_widths:
            info.avg_range_width_percent = sum(range_widths) / len(range_widths)
        
        if holding_days:
            info.avg_holding_days = sum(holding_days) / len(holding_days)
        
        if positions:
            info.success_rate = profitable / len(positions)
    
    def _assess_risk(
        self,
        session: Session,
        pool: Pool,
        info: NewPoolInfo,
    ) -> None:
        """Assess risk level for a new pool."""
        risk_score = 0
        
        # Age risk: newer = higher risk
        if info.age_days < 1:
            risk_score += 3
        elif info.age_days < 3:
            risk_score += 2
        elif info.age_days < 7:
            risk_score += 1
        
        # TVL risk: lower TVL = higher risk
        if info.tvl_usd < 100_000:
            risk_score += 3
        elif info.tvl_usd < 500_000:
            risk_score += 2
        elif info.tvl_usd < 1_000_000:
            risk_score += 1
        
        # Position count: fewer positions = less data = higher risk
        if info.position_count < 5:
            risk_score += 2
        elif info.position_count < 20:
            risk_score += 1
        
        # Check token verification
        token0 = session.query(Token).filter(Token.id == pool.token0_id).first()
        token1 = session.query(Token).filter(Token.id == pool.token1_id).first()
        
        info.token0_is_verified = token0.is_verified if token0 else False
        info.token1_is_verified = token1.is_verified if token1 else False
        
        if not info.token0_is_verified and not info.token1_is_verified:
            risk_score += 2
        elif not info.token0_is_verified or not info.token1_is_verified:
            risk_score += 1
            info.is_new_token = True
        
        # Assign risk level
        if risk_score >= 8:
            info.risk_level = "very_high"
        elif risk_score >= 5:
            info.risk_level = "high"
        elif risk_score >= 3:
            info.risk_level = "medium"
        else:
            info.risk_level = "low"
    
    def _generate_recommendation(self, info: NewPoolInfo) -> None:
        """Generate range recommendation based on analysis."""
        # Base recommendation on historical data or defaults
        if info.avg_range_width_percent > 0:
            # Use historical average
            info.recommended_range_percent = info.avg_range_width_percent
        else:
            # Default based on risk
            if info.risk_level == "very_high":
                info.recommended_range_percent = 50.0  # Wide range for volatile
            elif info.risk_level == "high":
                info.recommended_range_percent = 30.0
            elif info.risk_level == "medium":
                info.recommended_range_percent = 20.0
            else:
                info.recommended_range_percent = 10.0
    
    def scan_for_opportunities(
        self,
        networks: Optional[list[str]] = None,
        min_tvl: float = 100_000,
        max_risk: str = "high",  # Filter out very_high risk
    ) -> list[NewTokenAlert]:
        """
        Scan for new pool opportunities.
        
        Args:
            networks: Networks to scan
            min_tvl: Minimum TVL
            max_risk: Maximum acceptable risk level
            
        Returns:
            List of NewTokenAlert objects
        """
        risk_levels = ["low", "medium", "high", "very_high"]
        max_risk_idx = risk_levels.index(max_risk)
        
        new_pools = self.get_new_pools(
            min_tvl=min_tvl,
            networks=networks,
        )
        
        alerts = []
        
        for pool in new_pools:
            pool_risk_idx = risk_levels.index(pool.risk_level)
            
            if pool_risk_idx > max_risk_idx:
                continue  # Skip too risky
            
            # Generate alert
            message = (
                f"New pool on {pool.network}: {pool.token0_symbol}/{pool.token1_symbol}\n"
                f"TVL: ${pool.tvl_usd:,.0f}\n"
                f"Age: {pool.age_days:.1f} days\n"
                f"Risk: {pool.risk_level}\n"
                f"Recommended range: ±{pool.recommended_range_percent:.1f}%"
            )
            
            alert = NewTokenAlert(
                pool=pool,
                alert_type="new_pool",
                message=message,
            )
            alerts.append(alert)
        
        return alerts
    
    def save_opportunities_as_signals(
        self,
        networks: Optional[list[str]] = None,
    ) -> int:
        """
        Scan and save opportunities as signals.
        
        Returns:
            Number of signals created
        """
        alerts = self.scan_for_opportunities(networks=networks)
        
        with session_scope() as session:
            for alert in alerts:
                signal = Signal(**alert.to_signal())
                session.add(signal)
            session.commit()
        
        logger.info(f"Created {len(alerts)} new pool opportunity signals")
        return len(alerts)


def get_new_pools(
    max_age_days: int = 30,
    networks: Optional[list[str]] = None,
) -> list[NewPoolInfo]:
    """
    Convenience function to get new pools.
    
    Args:
        max_age_days: Maximum pool age
        networks: Networks to include
        
    Returns:
        List of NewPoolInfo objects
    """
    analyzer = NewTokenAnalyzer()
    return analyzer.get_new_pools(
        max_age_days=max_age_days,
        networks=networks,
    )
