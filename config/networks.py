"""
Network configurations for all supported chains.

Each network contains:
- chain_id: Unique identifier
- name: Human-readable name
- rpc_url: Default RPC endpoint (can be overridden via env)
- subgraphs: URLs for Uniswap v2/v3 subgraphs
- block_time: Average block time in seconds (for rate limiting)
- native_token: Native token symbol (for gas calculations)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict, Optional, Dict
from dataclasses import dataclass

# Load .env file
try:
    from dotenv import load_dotenv
    # Find .env in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use environment variables directly


@dataclass
class SubgraphConfig:
    """Subgraph endpoints for a network."""
    uniswap_v3: Optional[str] = None
    uniswap_v2: Optional[str] = None
    sushiswap: Optional[str] = None
    aerodrome: Optional[str] = None
    pancakeswap: Optional[str] = None


@dataclass
class NetworkConfig:
    """Configuration for a single network."""
    chain_id: int
    name: str
    rpc_url: str
    subgraphs: SubgraphConfig
    block_time: float  # seconds
    native_token: str
    explorer_url: str
    enabled: bool = True


# =============================================================================
# The Graph Decentralized Network
# =============================================================================
# 
# The Graph has migrated to decentralized network. Endpoints require API key.
# Get your API key at: https://thegraph.com/studio/
# 
# Subgraph IDs (used with gateway):
# - Ethereum Uniswap V3: 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV
# - Arbitrum Uniswap V3: FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM
# - Polygon Uniswap V3: 3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm
# - Optimism Uniswap V3: Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj
# - Base Uniswap V3: 43Hwfi3dJSoGpyas9VwXc7PJHGioL7KZibojPE4QJk2J
# - BNB Uniswap V3: F85MNzUGYqgSHSHRGgeVMNsdnW1KtZSVgFULumXRZTw2
#
# Gateway URL format: https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}

# Get API key from environment
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "")

def _build_gateway_url(subgraph_id: str) -> Optional[str]:
    """Build The Graph Gateway URL with API key."""
    if not GRAPH_API_KEY:
        return None
    return f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{subgraph_id}"


# Subgraph IDs
SUBGRAPH_IDS = {
    "ethereum": {
        "uniswap_v3": "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
    },
    "arbitrum": {
        "uniswap_v3": "FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM",
    },
    "polygon": {
        "uniswap_v3": "3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm",
    },
    "optimism": {
        "uniswap_v3": "Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj",
    },
    "base": {
        "uniswap_v3": "43Hwfi3dJSoGpyas9VwXc7PJHGioL7KZibojPE4QJk2J",
    },
    "bnb": {
        "uniswap_v3": "F85MNzUGYqgSHSHRGgeVMNsdnW1KtZSVgFULumXRZTw2",
    },
}


def _get_subgraph_url(network: str, dex: str) -> Optional[str]:
    """Get subgraph URL for network and DEX."""
    subgraph_id = SUBGRAPH_IDS.get(network, {}).get(dex)
    if subgraph_id:
        return _build_gateway_url(subgraph_id)
    return None


# =============================================================================
# Network Configurations
# =============================================================================

NETWORKS: Dict[str, NetworkConfig] = {
    "ethereum": NetworkConfig(
        chain_id=1,
        name="Ethereum Mainnet",
        rpc_url="https://eth.llamarpc.com",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("ethereum", "uniswap_v3"),
        ),
        block_time=12.0,
        native_token="ETH",
        explorer_url="https://etherscan.io",
    ),
    
    "arbitrum": NetworkConfig(
        chain_id=42161,
        name="Arbitrum One",
        rpc_url="https://arb1.arbitrum.io/rpc",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("arbitrum", "uniswap_v3"),
        ),
        block_time=0.25,
        native_token="ETH",
        explorer_url="https://arbiscan.io",
    ),
    
    "polygon": NetworkConfig(
        chain_id=137,
        name="Polygon",
        rpc_url="https://polygon-rpc.com",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("polygon", "uniswap_v3"),
        ),
        block_time=2.0,
        native_token="MATIC",
        explorer_url="https://polygonscan.com",
    ),
    
    "optimism": NetworkConfig(
        chain_id=10,
        name="Optimism",
        rpc_url="https://mainnet.optimism.io",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("optimism", "uniswap_v3"),
        ),
        block_time=2.0,
        native_token="ETH",
        explorer_url="https://optimistic.etherscan.io",
    ),
    
    "base": NetworkConfig(
        chain_id=8453,
        name="Base",
        rpc_url="https://mainnet.base.org",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("base", "uniswap_v3"),
        ),
        block_time=2.0,
        native_token="ETH",
        explorer_url="https://basescan.org",
    ),
    
    "bnb": NetworkConfig(
        chain_id=56,
        name="BNB Chain",
        rpc_url="https://bsc-dataseed.binance.org",
        subgraphs=SubgraphConfig(
            uniswap_v3=_get_subgraph_url("bnb", "uniswap_v3"),
        ),
        block_time=3.0,
        native_token="BNB",
        explorer_url="https://bscscan.com",
    ),
    
    "unichain": NetworkConfig(
        chain_id=130,
        name="Unichain",
        rpc_url="https://rpc.unichain.org",
        subgraphs=SubgraphConfig(
            uniswap_v3=None,
        ),
        block_time=1.0,
        native_token="ETH",
        explorer_url="https://uniscan.xyz",
        enabled=False,
    ),
}


def get_network_config(network: str) -> NetworkConfig:
    """
    Get configuration for a specific network.
    
    Args:
        network: Network identifier (e.g., 'ethereum', 'arbitrum')
        
    Returns:
        NetworkConfig for the specified network
        
    Raises:
        ValueError: If network is not found
    """
    if network not in NETWORKS:
        available = ", ".join(NETWORKS.keys())
        raise ValueError(f"Unknown network: {network}. Available: {available}")
    return NETWORKS[network]


def get_enabled_networks() -> Dict[str, NetworkConfig]:
    """Get all enabled networks."""
    return {k: v for k, v in NETWORKS.items() if v.enabled}


def get_networks_with_subgraph(dex: str) -> Dict[str, NetworkConfig]:
    """
    Get networks that have a specific DEX subgraph configured.
    
    Args:
        dex: DEX name (e.g., 'uniswap_v3', 'sushiswap')
        
    Returns:
        Dictionary of networks with that DEX subgraph
    """
    result = {}
    for name, config in NETWORKS.items():
        if config.enabled and getattr(config.subgraphs, dex, None):
            result[name] = config
    return result
