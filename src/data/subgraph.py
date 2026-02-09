"""
The Graph Subgraph Client for Uniswap V3 queries.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import requests

from config.networks import NETWORKS, GRAPH_API_KEY, SUBGRAPH_IDS

logger = logging.getLogger(__name__)


class SubgraphClient:
    """Client for querying The Graph subgraphs."""
    
    def __init__(self, network: str = "ethereum"):
        """
        Initialize subgraph client.
        
        Args:
            network: Network name (ethereum, arbitrum, etc.)
        """
        self.network = network
        self.api_key = GRAPH_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "GRAPH_API_KEY not set. Get your free API key at https://thegraph.com/studio/"
            )
        
        # Build endpoint URL
        subgraph_id = SUBGRAPH_IDS.get(network, {}).get("uniswap_v3")
        if not subgraph_id:
            raise ValueError(f"No Uniswap V3 subgraph configured for network: {network}")
        
        self.endpoint = f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/{subgraph_id}"
        self.session = requests.Session()
        
    def query(
        self, 
        query: str, 
        variables: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Query result data
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        for attempt in range(retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    raise Exception(f"GraphQL error: {error_msg}")
                
                return result.get("data", {})
                
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Query failed (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise
        
        return {}


# =============================================================================
# Query Templates
# =============================================================================

POOLS_QUERY = """
query getPools($first: Int!, $skip: Int!, $minTvl: BigDecimal!) {
    pools(
        first: $first
        skip: $skip
        orderBy: totalValueLockedUSD
        orderDirection: desc
        where: { totalValueLockedUSD_gte: $minTvl }
    ) {
        id
        token0 {
            id
            symbol
            name
            decimals
        }
        token1 {
            id
            symbol
            name
            decimals
        }
        feeTier
        tickSpacing
        liquidity
        sqrtPrice
        tick
        totalValueLockedUSD
        volumeUSD
        token0Price
        token1Price
        createdAtTimestamp
        createdAtBlockNumber
    }
}
"""

SWAPS_QUERY = """
query getSwaps($poolId: String!, $first: Int!, $skip: Int!, $startTime: BigInt!, $endTime: BigInt) {
    swaps(
        first: $first
        skip: $skip
        orderBy: timestamp
        orderDirection: desc
        where: { pool: $poolId, timestamp_gte: $startTime, timestamp_lte: $endTime }
    ) {
        id
        transaction {
            id
            blockNumber
            timestamp
            gasUsed
            gasPrice
        }
        timestamp
        sender
        recipient
        amount0
        amount1
        amountUSD
        sqrtPriceX96
        tick
    }
}
"""

MINTS_QUERY = """
query getMints($first: Int!, $skip: Int!, $startTime: BigInt!, $endTime: BigInt!, $minAmount: BigDecimal!) {
    mints(
        first: $first
        skip: $skip
        orderBy: timestamp
        orderDirection: desc
        where: { timestamp_gte: $startTime, timestamp_lte: $endTime, amountUSD_gte: $minAmount }
    ) {
        id
        transaction {
            id
            blockNumber
            timestamp
            gasUsed
            gasPrice
        }
        timestamp
        pool {
            id
            token0 { id, symbol, decimals }
            token1 { id, symbol, decimals }
        }
        owner
        sender
        origin
        amount0
        amount1
        amountUSD
        tickLower
        tickUpper
        logIndex
    }
}
"""

BURNS_QUERY = """
query getBurns($first: Int!, $skip: Int!, $startTime: BigInt!, $endTime: BigInt!, $minAmount: BigDecimal!) {
    burns(
        first: $first
        skip: $skip
        orderBy: timestamp
        orderDirection: desc
        where: { timestamp_gte: $startTime, timestamp_lte: $endTime, amountUSD_gte: $minAmount }
    ) {
        id
        transaction {
            id
            blockNumber
            timestamp
            gasUsed
            gasPrice
        }
        timestamp
        pool {
            id
            token0 { id, symbol, decimals }
            token1 { id, symbol, decimals }
        }
        owner
        origin
        amount0
        amount1
        amountUSD
        tickLower
        tickUpper
        logIndex
    }
}
"""

POSITIONS_QUERY = """
query getPositions($first: Int!, $skip: Int!, $minLiquidity: BigInt!) {
    positions(
        first: $first
        skip: $skip
        orderBy: liquidity
        orderDirection: desc
        where: { liquidity_gt: $minLiquidity }
    ) {
        id
        owner
        pool {
            id
            token0 { id, symbol, decimals }
            token1 { id, symbol, decimals }
        }
        tickLower {
            tickIdx
        }
        tickUpper {
            tickIdx
        }
        liquidity
        depositedToken0
        depositedToken1
        withdrawnToken0
        withdrawnToken1
        collectedFeesToken0
        collectedFeesToken1
    }
}
"""
