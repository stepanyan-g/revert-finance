"""
Module 4: LP Owner Analysis.

Analyzes LP position owners to identify successful strategies:
- Track PnL per owner
- Calculate win rates
- Identify top performers
- Extract patterns from successful LPs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from config.settings import get_settings
from src.db.models import Owner, OwnerStats, Position, PositionEvent, Pool
from src.db.database import session_scope


logger = logging.getLogger(__name__)


@dataclass
class OwnerMetrics:
    """Metrics for a single LP owner."""
    address: str
    is_contract: bool
    
    # Position counts
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    
    # PnL
    total_deposited_usd: float = 0
    total_withdrawn_usd: float = 0
    total_fees_usd: float = 0
    total_gas_usd: float = 0
    realized_pnl_usd: float = 0
    unrealized_pnl_usd: float = 0
    total_pnl_usd: float = 0
    
    # Success metrics
    profitable_positions: int = 0
    losing_positions: int = 0
    win_rate: float = 0
    avg_pnl_per_position: float = 0
    
    # Behavior
    favorite_networks: list[str] = field(default_factory=list)
    favorite_pools: list[str] = field(default_factory=list)
    avg_holding_days: float = 0
    avg_range_width_percent: float = 0
    
    # Ranking
    rank_by_pnl: int = 0
    rank_by_win_rate: int = 0
    
    def to_owner_stats(self, owner_id: int, network: str = "global") -> dict:
        """Convert to OwnerStats model dict."""
        return {
            "owner_id": owner_id,
            "network": network,
            "total_positions": self.total_positions,
            "open_positions": self.open_positions,
            "closed_positions": self.closed_positions,
            "profitable_positions": self.profitable_positions,
            "losing_positions": self.losing_positions,
            "win_rate": self.win_rate,
            "total_deposited_usd": Decimal(str(self.total_deposited_usd)),
            "total_withdrawn_usd": Decimal(str(self.total_withdrawn_usd)),
            "total_fees_collected_usd": Decimal(str(self.total_fees_usd)),
            "total_gas_cost_usd": Decimal(str(self.total_gas_usd)),
            "total_pnl_usd": Decimal(str(self.total_pnl_usd)),
            "avg_pnl_per_position_usd": Decimal(str(self.avg_pnl_per_position)),
            "favorite_pools": json.dumps(self.favorite_pools[:10]),
            "avg_holding_period_days": self.avg_holding_days,
            "avg_range_width_percent": self.avg_range_width_percent,
            "calculated_at": datetime.utcnow(),
        }


class OwnerAnalyzer:
    """
    Analyzes LP position owners to identify successful patterns.
    
    Key metrics:
    - Total PnL: withdrawn + fees - deposited - gas
    - Win rate: % of positions with positive PnL
    - Avg holding period
    - Range width preferences
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def calculate_position_pnl(
        self,
        position: Position,
        current_prices: Optional[dict] = None,
    ) -> dict:
        """
        Calculate PnL for a single position.
        
        Args:
            position: Position model instance
            current_prices: Dict of token_address -> price_usd for valuation
            
        Returns:
            Dict with PnL breakdown
        """
        # Realized PnL from closed/collected
        deposited = float(position.deposited_usd or 0)
        withdrawn = float(position.withdrawn_usd or 0)
        fees = float(position.collected_fees_usd or 0)
        gas = float(position.total_gas_cost_usd or 0)
        
        realized_pnl = withdrawn + fees - deposited - gas
        
        # Unrealized PnL for open positions
        unrealized_pnl = 0
        if not position.is_closed and position.current_value_usd:
            current_value = float(position.current_value_usd)
            unrealized_pnl = current_value - deposited + fees - gas
        
        total_pnl = realized_pnl + unrealized_pnl
        
        return {
            "position_id": position.id,
            "deposited_usd": deposited,
            "withdrawn_usd": withdrawn,
            "fees_usd": fees,
            "gas_usd": gas,
            "realized_pnl_usd": realized_pnl,
            "unrealized_pnl_usd": unrealized_pnl,
            "total_pnl_usd": total_pnl,
            "is_profitable": total_pnl > 0,
            "is_closed": position.is_closed,
        }
    
    def calculate_owner_metrics(
        self,
        session: Session,
        owner: Owner,
        network: Optional[str] = None,
    ) -> OwnerMetrics:
        """
        Calculate comprehensive metrics for an owner.
        
        Args:
            session: Database session
            owner: Owner model instance
            network: Filter by network (None = all)
            
        Returns:
            OwnerMetrics with all calculated values
        """
        # Query by owner_address (always set) rather than owner_id (may be NULL)
        query = session.query(Position).filter(
            Position.owner_address == owner.address.lower()
        )
        
        if network:
            query = query.filter(Position.network == network)
        
        positions = query.all()
        
        if not positions:
            return OwnerMetrics(
                address=owner.address,
                is_contract=owner.is_contract,
            )
        
        metrics = OwnerMetrics(
            address=owner.address,
            is_contract=owner.is_contract,
            total_positions=len(positions),
        )
        
        # Track for aggregation
        holding_periods = []
        range_widths = []
        network_counts: dict[str, int] = {}
        pool_counts: dict[str, int] = {}
        
        for pos in positions:
            # Count open/closed
            if pos.is_closed:
                metrics.closed_positions += 1
            else:
                metrics.open_positions += 1
            
            # Calculate PnL
            pnl = self.calculate_position_pnl(pos)
            
            metrics.total_deposited_usd += pnl["deposited_usd"]
            metrics.total_withdrawn_usd += pnl["withdrawn_usd"]
            metrics.total_fees_usd += pnl["fees_usd"]
            metrics.total_gas_usd += pnl["gas_usd"]
            
            if pos.is_closed:
                metrics.realized_pnl_usd += pnl["realized_pnl_usd"]
                if pnl["is_profitable"]:
                    metrics.profitable_positions += 1
                else:
                    metrics.losing_positions += 1
            else:
                metrics.unrealized_pnl_usd += pnl["unrealized_pnl_usd"]
            
            # Track holding period
            if pos.created_at:
                end_date = pos.closed_at or datetime.utcnow()
                days = (end_date - pos.created_at).days
                holding_periods.append(days)
            
            # Track range width
            if pos.tick_lower and pos.tick_upper:
                # Approximate range width (proper calculation needs pool tick)
                tick_range = pos.tick_upper - pos.tick_lower
                # Very rough approximation: each tick ~0.01%
                width_percent = tick_range * 0.01
                range_widths.append(width_percent)
            
            # Track preferences
            network_counts[pos.network] = network_counts.get(pos.network, 0) + 1
            if pos.pool_address:
                pool_counts[pos.pool_address] = pool_counts.get(pos.pool_address, 0) + 1
        
        # Aggregate metrics
        metrics.total_pnl_usd = metrics.realized_pnl_usd + metrics.unrealized_pnl_usd
        
        if metrics.closed_positions > 0:
            metrics.win_rate = metrics.profitable_positions / metrics.closed_positions
            metrics.avg_pnl_per_position = metrics.realized_pnl_usd / metrics.closed_positions
        
        if holding_periods:
            metrics.avg_holding_days = sum(holding_periods) / len(holding_periods)
        
        if range_widths:
            metrics.avg_range_width_percent = sum(range_widths) / len(range_widths)
        
        # Top networks and pools
        metrics.favorite_networks = sorted(
            network_counts.keys(),
            key=lambda x: network_counts[x],
            reverse=True
        )[:5]
        
        metrics.favorite_pools = sorted(
            pool_counts.keys(),
            key=lambda x: pool_counts[x],
            reverse=True
        )[:10]
        
        return metrics
    
    def get_top_owners(
        self,
        networks: Optional[list[str]] = None,
        min_positions: Optional[int] = None,
        limit: int = 100,
        order_by: str = "pnl",  # pnl, win_rate, positions
        exclude_contracts: bool = False,
    ) -> list[OwnerMetrics]:
        """
        Get top performing LP owners.
        
        Args:
            networks: Networks to include
            min_positions: Minimum positions for statistical significance
            limit: Number of top owners to return
            order_by: Ranking metric
            exclude_contracts: Exclude smart contract owners
            
        Returns:
            List of OwnerMetrics sorted by performance
        """
        if min_positions is None:
            min_positions = self.settings.owner_analysis.min_positions
        
        with session_scope() as session:
            # Get owners with enough positions
            query = session.query(Owner)
            
            if exclude_contracts:
                query = query.filter(Owner.is_contract == False)
            
            owners = query.all()
            
            all_metrics = []
            
            for owner in owners:
                try:
                    metrics = self.calculate_owner_metrics(session, owner)
                    
                    # Filter by minimum positions
                    if metrics.total_positions < min_positions:
                        continue
                    
                    # Filter by network if specified
                    if networks:
                        if not any(n in metrics.favorite_networks for n in networks):
                            continue
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error calculating metrics for {owner.address}: {e}")
            
            # Sort by specified metric
            if order_by == "pnl":
                all_metrics.sort(key=lambda x: x.realized_pnl_usd, reverse=True)
            elif order_by == "win_rate":
                # Сортируем по win_rate, затем по количеству закрытых позиций (для значимости)
                # Формула: win_rate * 100 + log(closed_positions) чтобы учесть оба фактора
                import math
                all_metrics.sort(
                    key=lambda x: (
                        x.win_rate,  # Главный критерий — процент успеха
                        x.closed_positions,  # Больше закрытых = статистически значимее
                    ), 
                    reverse=True
                )
            elif order_by == "positions":
                all_metrics.sort(key=lambda x: x.closed_positions, reverse=True)
            
            # Assign ranks
            for i, m in enumerate(all_metrics):
                m.rank_by_pnl = i + 1
            
            return all_metrics[:limit]
    
    def save_owner_stats(
        self,
        networks: Optional[list[str]] = None,
        min_positions: Optional[int] = None,
    ) -> int:
        """
        Calculate and save stats for all owners.
        
        Args:
            networks: Networks to include
            min_positions: Minimum positions
            
        Returns:
            Number of owners processed
        """
        if min_positions is None:
            min_positions = self.settings.owner_analysis.min_positions
        
        count = 0
        
        with session_scope() as session:
            owners = session.query(Owner).all()
            
            for owner in owners:
                try:
                    metrics = self.calculate_owner_metrics(session, owner)
                    
                    if metrics.total_positions < min_positions:
                        continue
                    
                    # Upsert stats
                    stats_dict = metrics.to_owner_stats(owner.id)
                    
                    existing = session.query(OwnerStats).filter(
                        OwnerStats.owner_id == owner.id,
                        OwnerStats.network == "global"
                    ).first()
                    
                    if existing:
                        for key, value in stats_dict.items():
                            if key != "owner_id":
                                setattr(existing, key, value)
                    else:
                        stats = OwnerStats(**stats_dict)
                        session.add(stats)
                    
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Error saving stats for {owner.address}: {e}")
            
            session.commit()
        
        logger.info(f"Saved stats for {count} owners")
        return count
    
    def get_owner_by_address(self, address: str) -> Optional[OwnerMetrics]:
        """
        Get metrics for a specific owner.
        
        Args:
            address: Owner wallet address
            
        Returns:
            OwnerMetrics or None if not found
        """
        with session_scope() as session:
            owner = session.query(Owner).filter(
                Owner.address == address.lower()
            ).first()
            
            if not owner:
                return None
            
            return self.calculate_owner_metrics(session, owner)
    
    def extract_success_patterns(
        self,
        top_count: int = 50,
        networks: Optional[list[str]] = None,
    ) -> dict:
        """
        Extract patterns from top performing owners.
        
        Analyzes what successful LPs have in common:
        - Preferred networks
        - Preferred pools
        - Average holding period
        - Range width preferences
        
        Returns:
            Dict with extracted patterns
        """
        top_owners = self.get_top_owners(
            networks=networks,
            limit=top_count,
            order_by="pnl",
            exclude_contracts=True,
        )
        
        if not top_owners:
            return {"error": "No owners found"}
        
        # Aggregate patterns
        network_scores: dict[str, float] = {}
        pool_scores: dict[str, int] = {}
        holding_periods = []
        range_widths = []
        win_rates = []
        
        for owner in top_owners:
            # Weight by PnL rank (higher rank = more weight)
            weight = (top_count - owner.rank_by_pnl + 1) / top_count
            
            for net in owner.favorite_networks:
                network_scores[net] = network_scores.get(net, 0) + weight
            
            for pool in owner.favorite_pools:
                pool_scores[pool] = pool_scores.get(pool, 0) + 1
            
            holding_periods.append(owner.avg_holding_days)
            range_widths.append(owner.avg_range_width_percent)
            win_rates.append(owner.win_rate)
        
        # Calculate recommendations
        top_networks = sorted(
            network_scores.keys(),
            key=lambda x: network_scores[x],
            reverse=True
        )[:5]
        
        top_pools = sorted(
            pool_scores.keys(),
            key=lambda x: pool_scores[x],
            reverse=True
        )[:10]
        
        avg_holding = sum(holding_periods) / len(holding_periods) if holding_periods else 0
        avg_range = sum(range_widths) / len(range_widths) if range_widths else 0
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
        
        return {
            "sample_size": len(top_owners),
            "recommended_networks": top_networks,
            "popular_pools": top_pools,
            "avg_holding_period_days": round(avg_holding, 1),
            "avg_range_width_percent": round(avg_range, 2),
            "avg_win_rate": round(avg_win_rate, 3),
            "top_performer": {
                "address": top_owners[0].address if top_owners else None,
                "pnl_usd": top_owners[0].total_pnl_usd if top_owners else 0,
                "positions": top_owners[0].total_positions if top_owners else 0,
                "win_rate": top_owners[0].win_rate if top_owners else 0,
            } if top_owners else None,
        }


def get_top_lp_owners(
    limit: int = 100,
    networks: Optional[list[str]] = None,
    order_by: str = "pnl",
) -> list[OwnerMetrics]:
    """
    Convenience function to get top LP owners.
    
    Args:
        limit: Number of owners to return
        networks: Networks to filter
        order_by: Ranking metric (pnl, win_rate, positions)
        
    Returns:
        List of OwnerMetrics
    """
    analyzer = OwnerAnalyzer()
    return analyzer.get_top_owners(networks=networks, limit=limit, order_by=order_by)
