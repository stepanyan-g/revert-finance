"""
Swap data loading and analysis from The Graph subgraphs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from config.networks import NETWORKS, GRAPH_API_KEY
from src.db.models import Pool, Swap
from .subgraph import SubgraphClient, SWAPS_QUERY

logger = logging.getLogger(__name__)


class SwapLoader:
    """Loads swap data from The Graph into the database."""
    
    def __init__(self):
        """Initialize swap loader."""
        self.page_size = 100
    
    def load_swaps_for_pool(
        self,
        session: Session,
        pool: Pool,
        hours: int = 24,
        limit: int = 100,
    ) -> int:
        """
        Load recent swaps for a specific pool.
        
        Args:
            session: SQLAlchemy session
            pool: Pool object
            hours: How many hours back to fetch
            limit: Maximum number of swaps
            
        Returns:
            Number of swaps loaded
        """
        if not GRAPH_API_KEY:
            logger.error("GRAPH_API_KEY not set")
            return 0
        
        try:
            client = SubgraphClient(pool.network)
        except ValueError as e:
            logger.warning(f"Failed to create client for {pool.network}: {e}")
            return 0
        
        start_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
        end_time = int(datetime.utcnow().timestamp())
        loaded_count = 0
        skip = 0
        
        while loaded_count < limit:
            try:
                data = client.query(
                    SWAPS_QUERY,
                    variables={
                        "poolId": pool.address,
                        "first": min(self.page_size, limit - loaded_count),
                        "skip": skip,
                        "startTime": str(start_time),
                        "endTime": str(end_time),
                    }
                )
                
                swaps = data.get("swaps", [])
                if not swaps:
                    break
                
                for swap_data in swaps:
                    self._save_swap(session, pool, swap_data)
                    loaded_count += 1
                
                session.commit()
                skip += len(swaps)
                
                if len(swaps) < self.page_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading swaps for pool {pool.address}: {e}")
                break
        
        return loaded_count
    
    def load_swaps_for_period(
        self,
        session: Session,
        network: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 500,
    ) -> int:
        """
        Load swaps for all pools in a network for a specific date range.
        
        Args:
            session: SQLAlchemy session
            network: Network name
            start_date: Start of period
            end_date: End of period
            limit: Maximum swaps per pool
            
        Returns:
            Total number of swaps loaded
        """
        if not GRAPH_API_KEY:
            logger.error("GRAPH_API_KEY not set")
            return 0
        
        # Get pools for this network that have been loaded
        pools = session.query(Pool).filter(Pool.network == network).all()
        
        if not pools:
            logger.warning(f"No pools found for network {network}. Load pools first.")
            return 0
        
        try:
            client = SubgraphClient(network)
        except ValueError as e:
            logger.warning(f"Failed to create client for {network}: {e}")
            return 0
        
        start_time = int(start_date.timestamp())
        end_time = int(end_date.timestamp())
        
        total_loaded = 0
        
        for pool in pools:
            loaded_count = 0
            skip = 0
            
            while loaded_count < limit:
                try:
                    data = client.query(
                        SWAPS_QUERY,
                        variables={
                            "poolId": pool.address,
                            "first": min(self.page_size, limit - loaded_count),
                            "skip": skip,
                            "startTime": str(start_time),
                            "endTime": str(end_time),
                        }
                    )
                    
                    swaps = data.get("swaps", [])
                    if not swaps:
                        break
                    
                    for swap_data in swaps:
                        self._save_swap(session, pool, swap_data)
                        loaded_count += 1
                    
                    skip += len(swaps)
                    
                    if len(swaps) < self.page_size:
                        break
                        
                except Exception as e:
                    logger.error(f"Error loading swaps for pool {pool.address}: {e}")
                    break
            
            total_loaded += loaded_count
        
        session.commit()
        logger.info(f"Loaded {total_loaded} swaps for {network} from {start_date} to {end_date}")
        return total_loaded
    
    def _save_swap(self, session: Session, pool: Pool, data: dict) -> Swap:
        """Save or update a swap in the database."""
        tx = data.get("transaction", {})
        tx_hash = tx.get("id", data.get("id", "")).split("#")[0].lower()
        
        # Check if swap exists
        existing = session.query(Swap).filter(
            Swap.tx_hash == tx_hash,
            Swap.pool_id == pool.id,
        ).first()
        
        if existing:
            return existing
        
        # Parse timestamp
        timestamp = datetime.fromtimestamp(int(data.get("timestamp", 0)))
        
        # Parse amounts
        amount0 = Decimal(str(data.get("amount0", 0) or 0))
        amount1 = Decimal(str(data.get("amount1", 0) or 0))
        amount_usd = Decimal(str(data.get("amountUSD", 0) or 0))
        
        # Determine direction (positive amount0 = token0 bought = buy)
        direction = "buy" if amount0 > 0 else "sell"
        
        swap = Swap(
            pool_id=pool.id,
            network=pool.network,
            tx_hash=tx_hash,
            block_number=int(tx.get("blockNumber", 0)),
            timestamp=timestamp,
            sender=data.get("sender", "").lower(),
            recipient=data.get("recipient", "").lower() if data.get("recipient") else None,
            amount0=int(amount0 * (10 ** 18)),  # Store as raw amount
            amount1=int(amount1 * (10 ** 18)),
            amount_usd=abs(amount_usd),
            sqrt_price_x96=data.get("sqrtPriceX96"),
            tick=int(data["tick"]) if data.get("tick") else None,
            direction=direction,
            gas_used=int(tx.get("gasUsed", 0)) if tx.get("gasUsed") else None,
            gas_price=int(tx.get("gasPrice", 0)) if tx.get("gasPrice") else None,
        )
        
        session.add(swap)
        return swap


class SwapAnalyzer:
    """Analyzes swap data for flow patterns."""
    
    def get_net_flow(
        self,
        session: Session,
        pool_id: int,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get net flow for a specific pool.
        
        Args:
            session: SQLAlchemy session
            pool_id: Pool ID to analyze
            hours: Time period in hours
            
        Returns:
            Dict with inflow_usd, outflow_usd, net_flow_usd, swap_count
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        swaps = session.query(Swap).filter(
            Swap.pool_id == pool_id,
            Swap.timestamp >= start_time,
        ).all()
        
        inflow = sum(float(s.amount_usd or 0) for s in swaps if s.direction == "buy")
        outflow = sum(float(s.amount_usd or 0) for s in swaps if s.direction == "sell")
        
        return {
            "inflow_usd": inflow,
            "outflow_usd": outflow,
            "net_flow_usd": inflow - outflow,
            "swap_count": len(swaps),
        }
    
    def get_flow_by_token(
        self,
        session: Session,
        hours: int = 24,
        min_volume_usd: float = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get net flow by token over a time period.
        
        Args:
            session: SQLAlchemy session
            hours: Time period in hours
            min_volume_usd: Minimum volume filter
            
        Returns:
            List of flow data per token
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Query swaps grouped by pool
        results = session.query(
            Pool.token0_symbol,
            Pool.token1_symbol,
            Pool.network,
            func.sum(Swap.amount_usd).label("total_volume"),
            func.count(Swap.id).label("swap_count"),
        ).join(Pool).filter(
            Swap.timestamp >= start_time,
        ).group_by(
            Pool.token0_symbol,
            Pool.token1_symbol,
            Pool.network,
        ).having(
            func.sum(Swap.amount_usd) >= min_volume_usd
        ).all()
        
        flows = []
        for r in results:
            # Calculate inflow/outflow based on swap directions
            pool_swaps = session.query(Swap).join(Pool).filter(
                Pool.token0_symbol == r.token0_symbol,
                Pool.token1_symbol == r.token1_symbol,
                Pool.network == r.network,
                Swap.timestamp >= start_time,
            ).all()
            
            inflow = sum(float(s.amount_usd or 0) for s in pool_swaps if s.direction == "buy")
            outflow = sum(float(s.amount_usd or 0) for s in pool_swaps if s.direction == "sell")
            
            flows.append({
                "token_symbol": f"{r.token0_symbol}/{r.token1_symbol}",
                "network": r.network,
                "inflow_usd": inflow,
                "outflow_usd": outflow,
                "net_flow_usd": inflow - outflow,
                "swap_count": r.swap_count,
            })
        
        return sorted(flows, key=lambda x: abs(x["net_flow_usd"]), reverse=True)
    
    def get_large_swaps(
        self,
        session: Session,
        min_amount_usd: float = 50000,
        hours: int = 24,
    ) -> List[Swap]:
        """
        Get large swaps above a threshold.
        
        Args:
            session: SQLAlchemy session
            min_amount_usd: Minimum swap amount
            hours: Time period in hours
            
        Returns:
            List of large swaps
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(Swap).filter(
            Swap.timestamp >= start_time,
            Swap.amount_usd >= min_amount_usd,
        ).order_by(Swap.amount_usd.desc()).all()
