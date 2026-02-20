"""
Pool data loading from The Graph subgraphs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy.orm import Session

from config.networks import NETWORKS, GRAPH_API_KEY
from src.db.models import Pool, Token
from .subgraph import SubgraphClient, POOLS_QUERY

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class PoolLoader:
    """Loads pool data from The Graph into the database."""
    
    def __init__(self):
        """Initialize pool loader."""
        self.page_size = 100
        self._token_cache = {}  # Track tokens we've added: {(network, address): Token}
    
    def load_pools_for_network(
        self,
        session: Session,
        network: str,
        min_tvl: float = 50000,
        limit: int = 500,
    ) -> int:
        """
        Load pools from a specific network.
        
        Args:
            session: SQLAlchemy session
            network: Network name (ethereum, arbitrum, etc.)
            min_tvl: Minimum TVL in USD
            limit: Maximum number of pools to load
            
        Returns:
            Number of pools loaded
        """
        if not GRAPH_API_KEY:
            logger.error("GRAPH_API_KEY not set")
            return 0
        
        network_config = NETWORKS.get(network)
        if not network_config or not network_config.enabled:
            logger.warning(f"Network {network} not available or disabled")
            return 0
        
        try:
            client = SubgraphClient(network)
        except ValueError as e:
            logger.warning(f"Failed to create client for {network}: {e}")
            return 0
        
        loaded_count = 0
        skip = 0
        
        while loaded_count < limit:
            try:
                query_vars = {
                    "first": min(self.page_size, limit - loaded_count),
                    "skip": skip,
                    "minTvl": str(min_tvl),
                }
                
                logger.info(f"Querying The Graph for {network} pools: {query_vars}")
                data = client.query(
                    POOLS_QUERY,
                    variables=query_vars
                )
                
                # Check for GraphQL errors
                if not data:
                    logger.error(f"Empty response from The Graph for {network}")
                    break
                
                pools = data.get("pools", [])
                logger.info(f"Received {len(pools)} pools from The Graph for {network}")
                
                if not pools:
                    if skip == 0:
                        logger.warning(f"No pools found for {network} with TVL >= {min_tvl}")
                    break
                
                logger.info(f"Received {len(pools)} pools from The Graph (skip={skip})")
                
                for pool_data in pools:
                    try:
                        pool = self._save_pool(session, network, pool_data)
                        if pool:
                            loaded_count += 1
                        else:
                            logger.warning(f"Failed to save pool {pool_data.get('id', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Error saving pool {pool_data.get('id', 'unknown')}: {e}", exc_info=True)
                        # Continue with next pool instead of breaking
                        continue
                
                try:
                    session.commit()
                    logger.info(f"Committed {loaded_count} pools so far")
                except Exception as e:
                    logger.error(f"Error committing pools: {e}", exc_info=True)
                    session.rollback()
                    break
                
                skip += len(pools)
                
                if len(pools) < self.page_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading pools from {network}: {e}")
                break
        
        logger.info(f"Loaded {loaded_count} pools from {network}")
        return loaded_count
    
    def _save_pool(self, session: Session, network: str, data: dict) -> Optional[Pool]:
        """Save or update a pool in the database."""
        try:
            address = data.get("id", "").lower()
            if not address or len(address) != 42:
                logger.error(f"Invalid pool address: {data.get('id', 'missing')}")
                return None
            
            # Check if pool exists
            pool = session.query(Pool).filter(
                Pool.network == network,
                Pool.address == address,
            ).first()
            
            # Parse token data
            token0 = data.get("token0", {})
            token1 = data.get("token1", {})
            
            # Ensure tokens exist
            self._ensure_token(session, network, token0)
            self._ensure_token(session, network, token1)
            
            # Parse timestamps
            created_at = None
            if data.get("createdAtTimestamp"):
                try:
                    created_at = datetime.fromtimestamp(int(data["createdAtTimestamp"]))
                except:
                    pass
            
            if pool:
                # Update existing pool
                pool.tvl_usd = Decimal(str(data.get("totalValueLockedUSD", 0) or 0))
                pool.volume_24h_usd = Decimal(str(data.get("volumeUSD", 0) or 0))
                pool.current_tick = int(data["tick"]) if data.get("tick") else None
                pool.sqrt_price_x96 = data.get("sqrtPrice")
                pool.token0_price = Decimal(str(data.get("token0Price", 0) or 0))
                pool.token1_price = Decimal(str(data.get("token1Price", 0) or 0))
                pool.updated_at = datetime.utcnow()
            else:
                # Create new pool
                pool = Pool(
                    network=network,
                    dex="uniswap_v3",
                    address=address,
                    token0_address=token0.get("id", "").lower(),
                    token1_address=token1.get("id", "").lower(),
                    token0_symbol=token0.get("symbol"),
                    token1_symbol=token1.get("symbol"),
                    fee_tier=int(data.get("feeTier", 0)),
                    tick_spacing=None,
                    tvl_usd=Decimal(str(data.get("totalValueLockedUSD", 0) or 0)),
                    volume_24h_usd=Decimal(str(data.get("volumeUSD", 0) or 0)),
                    current_tick=int(data["tick"]) if data.get("tick") else None,
                    sqrt_price_x96=data.get("sqrtPrice"),
                    token0_price=Decimal(str(data.get("token0Price", 0) or 0)),
                    token1_price=Decimal(str(data.get("token1Price", 0) or 0)),
                    created_at_block=int(data.get("createdAtBlockNumber", 0)) if data.get("createdAtBlockNumber") else None,
                    created_at=created_at,
                )
                session.add(pool)
                logger.debug(f"Added new pool: {network}/{address} ({pool.token0_symbol}/{pool.token1_symbol})")
            
            return pool
        except Exception as e:
            logger.error(f"Error in _save_pool for {data.get('id', 'unknown')}: {e}", exc_info=True)
            return None
    
    def _ensure_token(self, session: Session, network: str, token_data: dict) -> Token:
        """Ensure token exists in database."""
        if not token_data or not token_data.get("id"):
            return None
        
        address = token_data["id"].lower()
        cache_key = (network, address)
        
        # Check our cache first (tokens we've added in this loader instance)
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        # Try to find existing token in database
        token = session.query(Token).filter(
            Token.network == network,
            Token.address == address,
        ).first()
        
        if token:
            self._token_cache[cache_key] = token
            return token
        
        # Create new token
        token = Token(
            network=network,
            address=address,
            symbol=token_data.get("symbol"),
            name=token_data.get("name"),
            decimals=int(token_data.get("decimals", 18)),
        )
        session.add(token)
        self._token_cache[cache_key] = token
        
        return token
    
    def load_all_pools(
        self,
        networks: Optional[List[str]] = None,
        min_tvl: float = 50000,
        limit_per_network: int = 200,
    ) -> dict:
        """
        Load pools from multiple networks.
        
        Args:
            networks: List of networks to load from (None = all enabled)
            min_tvl: Minimum TVL in USD
            limit_per_network: Max pools per network
            
        Returns:
            Dict mapping network -> count of loaded pools
        """
        from src.db.database import session_scope
        
        if networks is None:
            networks = [n for n, c in NETWORKS.items() if c.enabled]
        
        results = {}
        
        for network in networks:
            try:
                # Clear token cache between networks (each network uses a separate session)
                self._token_cache.clear()
                
                with session_scope() as session:
                    count = self.load_pools_for_network(
                        session, network, min_tvl, limit_per_network
                    )
                    results[network] = count
            except Exception as e:
                logger.error(f"Error loading pools from {network}: {e}")
                results[network] = 0
        
        return results
