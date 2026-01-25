from __future__ import annotations

"""
Module 3: Capital Flow Analysis - Large Outflow Detection.

Monitors for significant capital outflows from pools/tokens and generates alerts.
"""

import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import func

from config.settings import get_settings
from src.db.models import Pool, Swap, Signal
from src.db.database import session_scope
from src.data.swaps import SwapAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class OutflowAlert:
    """Represents a large outflow detection."""
    pool_id: int
    pool_address: str
    token0_symbol: str
    token1_symbol: str
    network: str
    
    outflow_usd: float
    inflow_usd: float
    net_flow_usd: float
    
    tvl_usd: float
    outflow_percent_of_tvl: float
    
    swap_count: int
    largest_swap_usd: float
    
    time_window_hours: int
    severity: str  # info, warning, critical
    
    def to_signal(self) -> dict:
        """Convert to Signal model dict."""
        return {
            "signal_type": "large_outflow",
            "severity": self.severity,
            "network": self.network,
            "pool_address": self.pool_address,
            "title": f"Large outflow: {self.token0_symbol}/{self.token1_symbol}",
            "message": self._format_message(),
            "data": json.dumps(self._to_dict()),
            "amount_usd": Decimal(str(abs(self.net_flow_usd))),
            "percent_change": self.outflow_percent_of_tvl,
        }
    
    def _format_message(self) -> str:
        """Format human-readable message."""
        return (
            f"Pool {self.token0_symbol}/{self.token1_symbol} on {self.network}\n"
            f"Net outflow: ${abs(self.net_flow_usd):,.0f} ({self.outflow_percent_of_tvl:.1f}% of TVL)\n"
            f"Outflow: ${self.outflow_usd:,.0f} | Inflow: ${self.inflow_usd:,.0f}\n"
            f"TVL: ${self.tvl_usd:,.0f}\n"
            f"Swaps: {self.swap_count} | Largest: ${self.largest_swap_usd:,.0f}\n"
            f"Window: {self.time_window_hours}h"
        )
    
    def _to_dict(self) -> dict:
        """Convert to dict for JSON storage."""
        return {
            "pool_id": self.pool_id,
            "pool_address": self.pool_address,
            "token0_symbol": self.token0_symbol,
            "token1_symbol": self.token1_symbol,
            "network": self.network,
            "outflow_usd": self.outflow_usd,
            "inflow_usd": self.inflow_usd,
            "net_flow_usd": self.net_flow_usd,
            "tvl_usd": self.tvl_usd,
            "outflow_percent_of_tvl": self.outflow_percent_of_tvl,
            "swap_count": self.swap_count,
            "largest_swap_usd": self.largest_swap_usd,
            "time_window_hours": self.time_window_hours,
            "severity": self.severity,
        }


class CapitalFlowAnalyzer:
    """
    Analyzes capital flows and detects anomalous outflows.
    
    Detection methods:
    1. Absolute threshold: outflow > X USD
    2. Relative threshold: outflow > Y% of TVL
    3. Statistical: outflow > Z percentile of historical outflows
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.swap_analyzer = SwapAnalyzer()
    
    def analyze_pool(
        self,
        session: Session,
        pool: Pool,
        hours: int = None,
    ) -> Optional[OutflowAlert]:
        """
        Analyze capital flow for a single pool.
        
        Args:
            session: Database session
            pool: Pool to analyze
            hours: Time window (uses settings if not specified)
            
        Returns:
            OutflowAlert if significant outflow detected, None otherwise
        """
        if hours is None:
            hours = self.settings.capital_flow.outflow_window_hours
        
        # Get flow data
        flow = self.swap_analyzer.get_net_flow(
            session,
            pool_id=pool.id,
            hours=hours,
        )
        
        net_flow = flow["net_flow_usd"]
        outflow = flow["outflow_usd"]
        inflow = flow["inflow_usd"]
        
        # Skip if net inflow
        if net_flow >= 0:
            return None
        
        abs_outflow = abs(net_flow)
        tvl = float(pool.tvl_usd or 0)
        
        # Calculate percent of TVL
        outflow_percent = (abs_outflow / tvl * 100) if tvl > 0 else 0
        
        # Check thresholds
        absolute_threshold = self.settings.capital_flow.large_outflow_usd
        relative_threshold = self.settings.capital_flow.large_outflow_tvl_percent
        
        is_large_absolute = abs_outflow >= absolute_threshold
        is_large_relative = outflow_percent >= relative_threshold
        
        if not (is_large_absolute or is_large_relative):
            return None
        
        # Determine severity
        if is_large_absolute and is_large_relative:
            if outflow_percent >= relative_threshold * 2:
                severity = "critical"
            else:
                severity = "warning"
        else:
            severity = "info"
        
        # Get largest swap
        since = datetime.utcnow() - timedelta(hours=hours)
        largest_swap = session.query(func.max(Swap.amount_usd)).filter(
            Swap.pool_id == pool.id,
            Swap.timestamp >= since,
            Swap.direction == "sell"
        ).scalar() or 0
        
        return OutflowAlert(
            pool_id=pool.id,
            pool_address=pool.address,
            token0_symbol=pool.token0_symbol or "???",
            token1_symbol=pool.token1_symbol or "???",
            network=pool.network,
            outflow_usd=outflow,
            inflow_usd=inflow,
            net_flow_usd=net_flow,
            tvl_usd=tvl,
            outflow_percent_of_tvl=outflow_percent,
            swap_count=flow["swap_count"],
            largest_swap_usd=float(largest_swap),
            time_window_hours=hours,
            severity=severity,
        )
    
    def scan_all_pools(
        self,
        networks: Optional[list[str]] = None,
        hours: int = None,
        save_signals: bool = True,
    ) -> list[OutflowAlert]:
        """
        Scan all pools for large outflows.
        
        Args:
            networks: Networks to scan (None = all)
            hours: Time window
            save_signals: Whether to save signals to database
            
        Returns:
            List of OutflowAlert objects
        """
        if hours is None:
            hours = self.settings.capital_flow.outflow_window_hours
        
        alerts = []
        
        with session_scope() as session:
            query = session.query(Pool).filter(
                Pool.tvl_usd >= self.settings.pool_filter.min_tvl_usd
            )
            
            if networks:
                query = query.filter(Pool.network.in_(networks))
            
            pools = query.all()
            logger.info(f"Scanning {len(pools)} pools for outflows")
            
            for pool in pools:
                try:
                    alert = self.analyze_pool(session, pool, hours)
                    if alert:
                        alerts.append(alert)
                        logger.info(
                            f"Outflow detected: {alert.token0_symbol}/{alert.token1_symbol} "
                            f"on {alert.network}: ${abs(alert.net_flow_usd):,.0f} "
                            f"({alert.severity})"
                        )
                        
                        if save_signals:
                            signal = Signal(**alert.to_signal())
                            session.add(signal)
                            
                except Exception as e:
                    logger.error(f"Error analyzing pool {pool.address}: {e}")
            
            session.commit()
        
        # Sort by severity and amount
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: (severity_order[x.severity], -abs(x.net_flow_usd)))
        
        logger.info(f"Found {len(alerts)} outflow alerts")
        return alerts
    
    def get_token_outflows(
        self,
        hours: int = 24,
        min_outflow_usd: float = 50000,
        networks: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get aggregated outflows by token.
        
        Returns list of tokens with significant net outflows.
        """
        with session_scope() as session:
            flow_data = self.swap_analyzer.get_flow_by_token(
                session,
                hours=hours,
                min_volume_usd=min_outflow_usd,
            )
            
            # Filter to outflows only
            outflows = [
                d for d in flow_data
                if d["net_flow_usd"] < -min_outflow_usd
            ]
            
            return outflows


def detect_large_outflows(
    hours: int = 1,
    networks: Optional[list[str]] = None,
    save_signals: bool = True,
) -> list[OutflowAlert]:
    """
    Convenience function to detect large outflows.
    
    Args:
        hours: Time window to analyze
        networks: Networks to scan
        save_signals: Save to database
        
    Returns:
        List of OutflowAlert objects
    """
    analyzer = CapitalFlowAnalyzer()
    return analyzer.scan_all_pools(
        networks=networks,
        hours=hours,
        save_signals=save_signals,
    )
