"""
Utility functions for LP strategy.
"""

import logging
import math
import sys
from decimal import Decimal
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom format string
    """
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def tick_to_price(tick: int, decimals0: int = 18, decimals1: int = 18) -> float:
    """
    Convert Uniswap v3 tick to price.
    
    Price = 1.0001^tick * 10^(decimals0 - decimals1)
    
    Args:
        tick: Uniswap v3 tick value
        decimals0: Decimals of token0
        decimals1: Decimals of token1
        
    Returns:
        Price of token0 in terms of token1
    """
    return (1.0001 ** tick) * (10 ** (decimals0 - decimals1))


def price_to_tick(price: float, decimals0: int = 18, decimals1: int = 18) -> int:
    """
    Convert price to Uniswap v3 tick.
    
    tick = log(price / 10^(decimals0 - decimals1)) / log(1.0001)
    
    Args:
        price: Price of token0 in terms of token1
        decimals0: Decimals of token0
        decimals1: Decimals of token1
        
    Returns:
        Nearest tick value
    """
    adjusted_price = price / (10 ** (decimals0 - decimals1))
    if adjusted_price <= 0:
        raise ValueError("Price must be positive")
    
    tick = math.log(adjusted_price) / math.log(1.0001)
    return int(round(tick))


def calculate_range_width_percent(
    tick_lower: int,
    tick_upper: int,
    current_tick: Optional[int] = None,
) -> float:
    """
    Calculate the width of a LP range as a percentage.
    
    For a range [P_low, P_high], width = (P_high - P_low) / P_mid * 100
    
    Args:
        tick_lower: Lower tick of range
        tick_upper: Upper tick of range
        current_tick: Current pool tick (for relative calculation)
        
    Returns:
        Range width as percentage
    """
    price_lower = tick_to_price(tick_lower)
    price_upper = tick_to_price(tick_upper)
    
    if current_tick is not None:
        mid_price = tick_to_price(current_tick)
    else:
        mid_price = (price_lower + price_upper) / 2
    
    width_percent = (price_upper - price_lower) / mid_price * 100
    return width_percent


def sqrt_price_x96_to_price(
    sqrt_price_x96: str,
    decimals0: int = 18,
    decimals1: int = 18,
) -> float:
    """
    Convert sqrtPriceX96 to actual price.
    
    price = (sqrtPriceX96 / 2^96)^2 * 10^(decimals0 - decimals1)
    
    Args:
        sqrt_price_x96: sqrtPriceX96 value from pool
        decimals0: Decimals of token0
        decimals1: Decimals of token1
        
    Returns:
        Price of token0 in terms of token1
    """
    sqrt_price = int(sqrt_price_x96) / (2 ** 96)
    price = sqrt_price ** 2 * (10 ** (decimals0 - decimals1))
    return price


def format_usd(amount: float, decimals: int = 0) -> str:
    """
    Format amount as USD string.
    
    Args:
        amount: Amount in USD
        decimals: Decimal places to show
        
    Returns:
        Formatted string like "$1,234.56"
    """
    if decimals > 0:
        return f"${amount:,.{decimals}f}"
    return f"${amount:,.0f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value (0.1 = 10%)
        decimals: Decimal places
        
    Returns:
        Formatted string like "10.00%"
    """
    return f"{value * 100:.{decimals}f}%"


def wei_to_ether(wei: int) -> float:
    """Convert wei to ether."""
    return wei / 10**18


def gwei_to_wei(gwei: float) -> int:
    """Convert gwei to wei."""
    return int(gwei * 10**9)


def calculate_gas_cost_usd(
    gas_used: int,
    gas_price_wei: int,
    eth_price_usd: float,
) -> float:
    """
    Calculate gas cost in USD.
    
    Args:
        gas_used: Gas units used
        gas_price_wei: Gas price in wei
        eth_price_usd: Current ETH price
        
    Returns:
        Gas cost in USD
    """
    gas_cost_eth = (gas_used * gas_price_wei) / 10**18
    return gas_cost_eth * eth_price_usd


def is_stablecoin(symbol: str) -> bool:
    """
    Check if token is a stablecoin.
    
    Args:
        symbol: Token symbol
        
    Returns:
        True if stablecoin
    """
    stablecoins = {
        "USDC", "USDT", "DAI", "FRAX", "BUSD", "TUSD", "USDP",
        "GUSD", "LUSD", "sUSD", "USDD", "DOLA", "MIM", "UST",
        "EURC", "EURS", "agEUR", "EURT",
    }
    return symbol.upper() in stablecoins


def is_wrapped_native(symbol: str, network: str) -> bool:
    """
    Check if token is wrapped native token.
    
    Args:
        symbol: Token symbol
        network: Network name
        
    Returns:
        True if wrapped native
    """
    wrapped_native = {
        "ethereum": "WETH",
        "arbitrum": "WETH",
        "optimism": "WETH",
        "polygon": "WMATIC",
        "bnb": "WBNB",
        "base": "WETH",
    }
    return symbol.upper() == wrapped_native.get(network, "").upper()


def categorize_pair(symbol0: str, symbol1: str, network: str) -> str:
    """
    Categorize a trading pair.
    
    Categories:
    - stable-stable: Both stablecoins
    - stable-native: Stablecoin + wrapped native
    - stable-other: Stablecoin + other token
    - native-other: Wrapped native + other token
    - other-other: Two non-standard tokens
    
    Args:
        symbol0: First token symbol
        symbol1: Second token symbol
        network: Network name
        
    Returns:
        Category string
    """
    is_stable0 = is_stablecoin(symbol0)
    is_stable1 = is_stablecoin(symbol1)
    is_native0 = is_wrapped_native(symbol0, network)
    is_native1 = is_wrapped_native(symbol1, network)
    
    if is_stable0 and is_stable1:
        return "stable-stable"
    elif (is_stable0 and is_native1) or (is_native0 and is_stable1):
        return "stable-native"
    elif is_stable0 or is_stable1:
        return "stable-other"
    elif is_native0 or is_native1:
        return "native-other"
    else:
        return "other-other"
