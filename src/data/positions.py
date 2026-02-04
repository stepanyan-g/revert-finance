"""
Position data loading from The Graph subgraphs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from config.networks import NETWORKS, GRAPH_API_KEY
from src.db.models import Pool, Position, Owner
from .subgraph import SubgraphClient, MINTS_QUERY, BURNS_QUERY, POSITIONS_QUERY

logger = logging.getLogger(__name__)


class PositionLoader:
    """Loads position data from The Graph into the database."""
    
    def __init__(self):
        """Initialize position loader."""
        self.page_size = 100
    
    def load_positions_from_events(
        self,
        session: Session,
        network: str,
        min_amount_usd: str = "100",
        limit: int = 200,
        hours: int = 168,  # 7 days
    ) -> Dict[str, int]:
        """
        Load positions by analyzing mint/burn events.
        
        This approach captures both open and closed positions.
        
        Args:
            session: SQLAlchemy session
            network: Network name
            min_amount_usd: Minimum USD value to consider
            limit: Maximum events to process
            hours: How far back to look
            
        Returns:
            Dict with counts of open and closed positions
        """
        if not GRAPH_API_KEY:
            logger.error("GRAPH_API_KEY not set")
            return {"open": 0, "closed": 0}
        
        try:
            client = SubgraphClient(network)
        except ValueError as e:
            logger.warning(f"Failed to create client for {network}: {e}")
            return {"open": 0, "closed": 0}
        
        start_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
        
        # Track positions by (owner, pool, tick_range)
        positions_map: Dict[str, Dict] = {}
        
        # Load mints
        mints = self._load_mints(client, start_time, min_amount_usd, limit)
        for mint in mints:
            key = self._position_key(mint)
            if key not in positions_map:
                positions_map[key] = {
                    "network": network,
                    "pool_address": mint["pool"]["id"].lower(),
                    "pool_data": mint["pool"],
                    "owner_address": mint["owner"].lower(),
                    "tick_lower": int(mint["tickLower"]),
                    "tick_upper": int(mint["tickUpper"]),
                    "deposited_token0": Decimal("0"),
                    "deposited_token1": Decimal("0"),
                    "withdrawn_token0": Decimal("0"),
                    "withdrawn_token1": Decimal("0"),
                    "deposited_usd": Decimal("0"),
                    "withdrawn_usd": Decimal("0"),
                    "created_at": datetime.fromtimestamp(int(mint["timestamp"])),
                    "closed_at": None,
                    "is_closed": False,
                }
            
            pos = positions_map[key]
            pos["deposited_token0"] += Decimal(str(mint.get("amount0", 0) or 0))
            pos["deposited_token1"] += Decimal(str(mint.get("amount1", 0) or 0))
            pos["deposited_usd"] += Decimal(str(mint.get("amountUSD", 0) or 0))
        
        # Load burns
        burns = self._load_burns(client, start_time, min_amount_usd, limit)
        for burn in burns:
            key = self._position_key(burn)
            if key in positions_map:
                pos = positions_map[key]
                pos["withdrawn_token0"] += abs(Decimal(str(burn.get("amount0", 0) or 0)))
                pos["withdrawn_token1"] += abs(Decimal(str(burn.get("amount1", 0) or 0)))
                pos["withdrawn_usd"] += Decimal(str(burn.get("amountUSD", 0) or 0))
                pos["closed_at"] = datetime.fromtimestamp(int(burn["timestamp"]))
                
                # Mark as closed if most liquidity removed
                if pos["withdrawn_usd"] >= pos["deposited_usd"] * Decimal("0.9"):
                    pos["is_closed"] = True
        
        # Save positions to database
        open_count = 0
        closed_count = 0
        
        for key, pos_data in positions_map.items():
            position = self._save_position(session, pos_data)
            if position:
                if pos_data["is_closed"]:
                    closed_count += 1
                else:
                    open_count += 1
        
        session.commit()
        
        logger.info(f"Loaded {open_count} open and {closed_count} closed positions from {network}")
        return {"open": open_count, "closed": closed_count}
    
    def _position_key(self, event: dict) -> str:
        """Create unique key for position identification."""
        return f"{event['owner'].lower()}_{event['pool']['id'].lower()}_{event['tickLower']}_{event['tickUpper']}"
    
    def _load_mints(
        self, 
        client: SubgraphClient, 
        start_time: int,
        min_amount_usd: str,
        limit: int,
    ) -> List[dict]:
        """Load mint events from subgraph."""
        mints = []
        skip = 0
        
        while len(mints) < limit:
            try:
                data = client.query(
                    MINTS_QUERY,
                    variables={
                        "first": min(self.page_size, limit - len(mints)),
                        "skip": skip,
                        "startTime": str(start_time),
                        "minAmount": min_amount_usd,
                    }
                )
                
                batch = data.get("mints", [])
                if not batch:
                    break
                
                mints.extend(batch)
                skip += len(batch)
                
                if len(batch) < self.page_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading mints: {e}")
                break
        
        return mints
    
    def _load_burns(
        self, 
        client: SubgraphClient, 
        start_time: int,
        min_amount_usd: str,
        limit: int,
    ) -> List[dict]:
        """Load burn events from subgraph."""
        burns = []
        skip = 0
        
        while len(burns) < limit:
            try:
                data = client.query(
                    BURNS_QUERY,
                    variables={
                        "first": min(self.page_size, limit - len(burns)),
                        "skip": skip,
                        "startTime": str(start_time),
                        "minAmount": min_amount_usd,
                    }
                )
                
                batch = data.get("burns", [])
                if not batch:
                    break
                
                burns.extend(batch)
                skip += len(batch)
                
                if len(batch) < self.page_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading burns: {e}")
                break
        
        return burns
    
    def _save_position(self, session: Session, data: dict) -> Optional[Position]:
        """Save or update a position in the database."""
        # Ensure owner exists
        owner = self._ensure_owner(session, data["owner_address"])
        
        # Find or create pool
        pool = session.query(Pool).filter(
            Pool.network == data["network"],
            Pool.address == data["pool_address"],
        ).first()
        
        pool_id = pool.id if pool else None
        
        # Check if position exists
        position = session.query(Position).filter(
            Position.network == data["network"],
            Position.owner_address == data["owner_address"],
            Position.pool_address == data["pool_address"],
            Position.tick_lower == data["tick_lower"],
            Position.tick_upper == data["tick_upper"],
        ).first()
        
        if position:
            # Update existing
            position.deposited_token0 = int(data["deposited_token0"] * (10 ** 18))
            position.deposited_token1 = int(data["deposited_token1"] * (10 ** 18))
            position.withdrawn_token0 = int(data["withdrawn_token0"] * (10 ** 18))
            position.withdrawn_token1 = int(data["withdrawn_token1"] * (10 ** 18))
            position.deposited_usd = data["deposited_usd"]
            position.withdrawn_usd = data["withdrawn_usd"]
            position.is_closed = data["is_closed"]
            position.closed_at = data["closed_at"] if data["is_closed"] else None
            position.updated_at = datetime.utcnow()
        else:
            # Create new position
            # Generate a pseudo token_id from the position key
            token_id = hash(f"{data['owner_address']}_{data['pool_address']}_{data['tick_lower']}_{data['tick_upper']}") % (10 ** 15)
            
            position = Position(
                network=data["network"],
                dex="uniswap_v3",
                token_id=abs(token_id),
                pool_id=pool_id,
                pool_address=data["pool_address"],
                owner_id=owner.id if owner else None,
                owner_address=data["owner_address"],
                tick_lower=data["tick_lower"],
                tick_upper=data["tick_upper"],
                liquidity=0,  # Would need separate query for this
                deposited_token0=int(data["deposited_token0"] * (10 ** 18)),
                deposited_token1=int(data["deposited_token1"] * (10 ** 18)),
                withdrawn_token0=int(data["withdrawn_token0"] * (10 ** 18)),
                withdrawn_token1=int(data["withdrawn_token1"] * (10 ** 18)),
                deposited_usd=data["deposited_usd"],
                withdrawn_usd=data["withdrawn_usd"],
                is_closed=data["is_closed"],
                created_at=data["created_at"],
                closed_at=data["closed_at"] if data["is_closed"] else None,
            )
            session.add(position)
        
        return position
    
    def _ensure_owner(self, session: Session, address: str) -> Owner:
        """Ensure owner exists in database."""
        address = address.lower()
        
        owner = session.query(Owner).filter(Owner.address == address).first()
        
        if not owner:
            owner = Owner(
                address=address,
                is_contract=address.startswith("0x000000"),
            )
            session.add(owner)
            session.flush()
        
        return owner
    
    def load_all_positions(
        self,
        networks: Optional[List[str]] = None,
        limit_per_network: int = 200,
    ) -> Dict[str, Dict[str, int]]:
        """
        Load positions from multiple networks.
        
        Args:
            networks: List of networks to load from (None = all enabled)
            limit_per_network: Max positions per network
            
        Returns:
            Dict mapping network -> {"open": count, "closed": count}
        """
        from src.db.database import session_scope
        
        if networks is None:
            networks = [n for n, c in NETWORKS.items() if c.enabled]
        
        results = {}
        
        for network in networks:
            try:
                with session_scope() as session:
                    result = self.load_positions_from_events(
                        session, network, limit=limit_per_network
                    )
                    results[network] = result
            except Exception as e:
                logger.error(f"Error loading positions from {network}: {e}")
                results[network] = {"open": 0, "closed": 0, "error": str(e)}
        
        return results
