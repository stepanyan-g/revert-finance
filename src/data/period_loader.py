"""
Period-based data loading with statistics tracking.

Supports loading data by month with progress tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import func

from config.networks import NETWORKS, GRAPH_API_KEY
from src.db.models import Pool, Position, LoadingStats
from src.db.database import session_scope
from .subgraph import SubgraphClient
from .pools import PoolLoader
from .positions import PositionLoader

logger = logging.getLogger(__name__)


@dataclass
class PeriodOption:
    """Represents a selectable time period."""
    key: str  # e.g., "2024-01"
    label: str  # e.g., "Январь 2024"
    start: datetime
    end: datetime


@dataclass
class PeriodStats:
    """Statistics for a single period."""
    period_key: str
    period_label: str
    network: str
    data_type: str
    
    # From The Graph (estimated/cached)
    total_available: int
    
    # From our database
    total_loaded: int
    
    # Calculated
    loaded_percent: float
    is_fully_loaded: bool
    
    # Filter context
    min_tvl_usd: float
    min_amount_usd: float


def get_period_options(months_back: int = 12) -> List[PeriodOption]:
    """
    Generate list of period options from current month going back.
    
    Args:
        months_back: How many months to include
        
    Returns:
        List of PeriodOption objects
    """
    options = []
    now = datetime.utcnow()
    
    # Russian month names
    month_names = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
        5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
        9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
    }
    
    for i in range(months_back):
        # Calculate the month
        date = now - relativedelta(months=i)
        
        # Period key: YYYY-MM
        key = date.strftime("%Y-%m")
        
        # Label: "Январь 2024" or "Текущий месяц" for current
        if i == 0:
            label = f"Текущий месяц ({month_names[date.month]} {date.year})"
        elif i == 1:
            label = f"Прошлый месяц ({month_names[date.month]} {date.year})"
        else:
            label = f"{month_names[date.month]} {date.year}"
        
        # Start and end of month
        start = datetime(date.year, date.month, 1)
        if date.month == 12:
            end = datetime(date.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(date.year, date.month + 1, 1) - timedelta(seconds=1)
        
        options.append(PeriodOption(
            key=key,
            label=label,
            start=start,
            end=end,
        ))
    
    return options


def get_multi_period_options() -> List[Tuple[str, List[PeriodOption]]]:
    """
    Get grouped period options for dropdown.
    
    Returns:
        List of (label, periods) tuples
    """
    all_periods = get_period_options(12)
    
    options = [
        ("Текущий месяц", all_periods[:1]),
        ("Последние 2 месяца", all_periods[:2]),
        ("Последние 3 месяца", all_periods[:3]),
        ("Последние 6 месяцев", all_periods[:6]),
        ("Последний год", all_periods[:12]),
    ]
    
    # Also add individual months
    for period in all_periods:
        options.append((period.label, [period]))
    
    return options


# GraphQL query to count positions in a time range
COUNT_MINTS_QUERY = """
query countMints($startTime: BigInt!, $endTime: BigInt!, $minAmount: BigDecimal!) {
    mints(
        first: 1000
        where: { 
            timestamp_gte: $startTime, 
            timestamp_lte: $endTime,
            amountUSD_gte: $minAmount 
        }
    ) {
        id
    }
}
"""

COUNT_POOLS_QUERY = """
query countPools($minTvl: BigDecimal!) {
    pools(
        first: 1000
        where: { totalValueLockedUSD_gte: $minTvl }
    ) {
        id
    }
}
"""


class PeriodDataLoader:
    """
    Loads data by time period with statistics tracking.
    """
    
    def __init__(self):
        self.pool_loader = PoolLoader()
        self.position_loader = PositionLoader()
    
    def get_period_statistics(
        self,
        session: Session,
        periods: List[PeriodOption],
        networks: List[str],
        min_tvl_usd: float = 50000,
        min_amount_usd: float = 100,
        data_type: str = "positions",
    ) -> List[PeriodStats]:
        """
        Get loading statistics for given periods and networks.
        
        Args:
            session: Database session
            periods: List of periods to check
            networks: Networks to include
            min_tvl_usd: Minimum TVL filter for pools
            min_amount_usd: Minimum amount filter for positions
            data_type: "positions" or "pools"
            
        Returns:
            List of PeriodStats objects
        """
        stats = []
        
        for period in periods:
            for network in networks:
                # Get cached stats or estimate
                cached = self._get_cached_stats(
                    session, period, network, data_type, min_tvl_usd, min_amount_usd
                )
                
                if cached:
                    total_available = cached.total_available
                else:
                    # Estimate from The Graph (can be slow, so we cache it)
                    total_available = self._estimate_available(
                        network, period, data_type, min_tvl_usd, min_amount_usd
                    )
                    # Cache the estimate
                    self._save_stats(
                        session, period, network, data_type,
                        min_tvl_usd, min_amount_usd,
                        total_available, 0
                    )
                
                # Count loaded in our database
                total_loaded = self._count_loaded(
                    session, period, network, data_type, min_tvl_usd, min_amount_usd
                )
                
                # Calculate percentage
                if total_available > 0:
                    loaded_percent = (total_loaded / total_available) * 100
                else:
                    loaded_percent = 0 if total_loaded == 0 else 100
                
                stats.append(PeriodStats(
                    period_key=period.key,
                    period_label=period.label,
                    network=network,
                    data_type=data_type,
                    total_available=total_available,
                    total_loaded=total_loaded,
                    loaded_percent=min(loaded_percent, 100),
                    is_fully_loaded=loaded_percent >= 99,
                    min_tvl_usd=min_tvl_usd,
                    min_amount_usd=min_amount_usd,
                ))
        
        return stats
    
    def _get_cached_stats(
        self,
        session: Session,
        period: PeriodOption,
        network: str,
        data_type: str,
        min_tvl_usd: float,
        min_amount_usd: float,
    ) -> Optional[LoadingStats]:
        """Get cached loading stats if available."""
        return session.query(LoadingStats).filter(
            LoadingStats.period_key == period.key,
            LoadingStats.network == network,
            LoadingStats.data_type == data_type,
            LoadingStats.min_tvl_usd == Decimal(str(min_tvl_usd)),
            LoadingStats.min_amount_usd == Decimal(str(min_amount_usd)),
        ).first()
    
    def _save_stats(
        self,
        session: Session,
        period: PeriodOption,
        network: str,
        data_type: str,
        min_tvl_usd: float,
        min_amount_usd: float,
        total_available: int,
        total_loaded: int,
    ) -> LoadingStats:
        """Save or update loading stats."""
        existing = self._get_cached_stats(
            session, period, network, data_type, min_tvl_usd, min_amount_usd
        )
        
        if existing:
            existing.total_available = total_available
            existing.total_loaded = total_loaded
            existing.updated_at = datetime.utcnow()
            return existing
        
        stats = LoadingStats(
            period_key=period.key,
            period_start=period.start,
            period_end=period.end,
            network=network,
            data_type=data_type,
            min_tvl_usd=Decimal(str(min_tvl_usd)),
            min_amount_usd=Decimal(str(min_amount_usd)),
            total_available=total_available,
            total_loaded=total_loaded,
        )
        session.add(stats)
        return stats
    
    def _estimate_available(
        self,
        network: str,
        period: PeriodOption,
        data_type: str,
        min_tvl_usd: float,
        min_amount_usd: float,
    ) -> int:
        """Estimate available data from The Graph."""
        if not GRAPH_API_KEY:
            return 0
        
        try:
            client = SubgraphClient(network)
            
            if data_type == "positions":
                # Count mints in the period
                data = client.query(
                    COUNT_MINTS_QUERY,
                    variables={
                        "startTime": str(int(period.start.timestamp())),
                        "endTime": str(int(period.end.timestamp())),
                        "minAmount": str(min_amount_usd),
                    }
                )
                return len(data.get("mints", []))
            
            elif data_type == "pools":
                # Count pools above TVL threshold
                data = client.query(
                    COUNT_POOLS_QUERY,
                    variables={
                        "minTvl": str(min_tvl_usd),
                    }
                )
                return len(data.get("pools", []))
            
        except Exception as e:
            logger.error(f"Error estimating available for {network}: {e}")
            return 0
        
        return 0
    
    def _count_loaded(
        self,
        session: Session,
        period: PeriodOption,
        network: str,
        data_type: str,
        min_tvl_usd: float,
        min_amount_usd: float,
    ) -> int:
        """Count data already loaded in our database for the period."""
        if data_type == "positions":
            return session.query(Position).filter(
                Position.network == network,
                Position.created_at >= period.start,
                Position.created_at <= period.end,
                Position.deposited_usd >= min_amount_usd,
            ).count()
        
        elif data_type == "pools":
            return session.query(Pool).filter(
                Pool.network == network,
                Pool.tvl_usd >= min_tvl_usd,
            ).count()
        
        return 0
    
    def load_period_data(
        self,
        session: Session,
        periods: List[PeriodOption],
        networks: List[str],
        min_tvl_usd: float = 50000,
        min_amount_usd: float = 100,
        limit_per_period: int = 500,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Load data for given periods.
        
        Args:
            session: Database session
            periods: Periods to load
            networks: Networks to load
            min_tvl_usd: Minimum TVL for pools
            min_amount_usd: Minimum amount for positions
            limit_per_period: Max items per period
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with loading results
        """
        from src.data.swaps import SwapLoader
        
        results = {
            "pools": {},
            "positions": {},
            "swaps": {},
            "errors": [],
        }
        
        # Pools are not period-dependent, so load once per network
        pools_loaded_networks = set()
        
        # Calculate total steps: pools (once per network) + positions (per period+network) + swaps (per period+network)
        total_steps = len(networks) + len(periods) * len(networks) * 2  # pools + positions + swaps
        current_step = 0
        
        # First, load pools for each network (only once per network, not per period)
        for network in networks:
            try:
                if progress_callback:
                    progress_callback(
                        current_step / total_steps,
                        f"Загрузка пулов: {network}"
                    )
                
                pool_count = self.pool_loader.load_pools_for_network(
                    session, network, min_tvl=min_tvl_usd, limit=limit_per_period * len(periods)
                )
                
                results["pools"][network] = pool_count
                pools_loaded_networks.add(network)
                
            except Exception as e:
                results["errors"].append(f"Pools {network}: {e}")
            
            current_step += 1
        
        # Then load positions and swaps for each period
        swap_loader = SwapLoader()
        
        for period in periods:
            for network in networks:
                # Load positions for the period
                try:
                    if progress_callback:
                        progress_callback(
                            current_step / total_steps,
                            f"Загрузка позиций: {network} ({period.label})"
                        )
                    
                    # Load positions for the specific period using start_date and end_date
                    pos_result = self.position_loader.load_positions_from_events(
                        session, network,
                        min_amount_usd=str(min_amount_usd),
                        limit=limit_per_period,
                        start_date=period.start,
                        end_date=period.end,
                    )
                    
                    key = f"{period.key}_{network}"
                    results["positions"][key] = pos_result
                    
                    # Update stats
                    loaded_count = pos_result.get("open", 0) + pos_result.get("closed", 0)
                    self._save_stats(
                        session, period, network, "positions",
                        min_tvl_usd, min_amount_usd,
                        0,  # Will be updated next time we check
                        loaded_count,
                    )
                    
                except Exception as e:
                    results["errors"].append(f"Positions {network} {period.key}: {e}")
                
                current_step += 1
                
                # Load swaps for the period
                try:
                    if progress_callback:
                        progress_callback(
                            current_step / total_steps,
                            f"Загрузка свопов: {network} ({period.label})"
                        )
                    
                    # Load swaps for pools in this network
                    # Calculate hours from period
                    hours_in_period = int((period.end - period.start).total_seconds() / 3600)
                    
                    swap_result = swap_loader.load_swaps_for_period(
                        session, network,
                        start_date=period.start,
                        end_date=period.end,
                        limit=limit_per_period,
                    )
                    
                    key = f"{period.key}_{network}"
                    results["swaps"][key] = swap_result
                    
                except Exception as e:
                    results["errors"].append(f"Swaps {network} {period.key}: {e}")
                
                current_step += 1
        
        session.commit()
        
        if progress_callback:
            progress_callback(1.0, "Загрузка завершена!")
        
        return results
    
    def refresh_statistics(
        self,
        session: Session,
        networks: List[str],
        min_tvl_usd: float = 50000,
        min_amount_usd: float = 100,
    ) -> None:
        """
        Refresh cached statistics from The Graph.
        
        Clears old cached values so they get re-fetched.
        """
        # Delete old stats to force refresh
        session.query(LoadingStats).filter(
            LoadingStats.network.in_(networks),
            LoadingStats.min_tvl_usd == Decimal(str(min_tvl_usd)),
            LoadingStats.min_amount_usd == Decimal(str(min_amount_usd)),
        ).delete(synchronize_session=False)
        session.commit()
