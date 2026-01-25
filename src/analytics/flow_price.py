"""
Module 2: Flow vs Price Correlation.

Analyzes relationship between capital flows and price movements:
- Tracks net flow per token/pool
- Correlates with price changes
- Generates signals when significant flows occur
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass
import json

from sqlalchemy.orm import Session
from sqlalchemy import func

from config.settings import get_settings
from src.db.models import Pool, Swap, Signal, PriceSnapshot
from src.db.database import session_scope


logger = logging.getLogger(__name__)


@dataclass
class FlowPriceData:
    """Flow and price data for a token."""
    token_address: str
    token_symbol: str
    network: str
    
    # Flow data
    inflow_usd: float
    outflow_usd: float
    net_flow_usd: float
    swap_count: int
    
    # Flow as percentage of volume
    flow_volume_ratio: float  # net_flow / total_volume
    
    # Price data (if available)
    price_start: float
    price_end: float
    price_change_percent: float
    
    # Historical comparison
    flow_percentile: float  # How unusual is this flow (0-100)
    
    # Signal
    is_significant: bool
    signal_type: str  # large_inflow, large_outflow, none


@dataclass
class FlowPriceAlert:
    """Alert for significant flow-price relationship."""
    token_symbol: str
    token_address: str
    network: str
    
    flow_type: str  # inflow, outflow
    net_flow_usd: float
    flow_percentile: float
    
    expected_price_change: float  # Based on historical correlation
    
    message: str
    
    def to_signal(self) -> dict:
        """Convert to Signal model dict."""
        return {
            "signal_type": "flow_price_alert",
            "severity": "warning" if self.flow_percentile > 95 else "info",
            "network": self.network,
            "token_address": self.token_address,
            "title": f"{self.flow_type.title()}: {self.token_symbol}",
            "message": self.message,
            "data": json.dumps({
                "net_flow_usd": self.net_flow_usd,
                "flow_percentile": self.flow_percentile,
                "expected_price_change": self.expected_price_change,
            }),
            "amount_usd": Decimal(str(abs(self.net_flow_usd))),
            "percent_change": self.expected_price_change,
        }


class FlowPriceAnalyzer:
    """
    Analyzes correlation between capital flows and price movements.
    
    Features:
    - Track net flow per token
    - Calculate flow percentiles
    - Correlate with price changes
    - Generate predictive signals
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def get_token_flows(
        self,
        session: Session,
        hours: int = 24,
        min_volume_usd: float = 10000,
        network: Optional[str] = None,
    ) -> list[FlowPriceData]:
        """
        Get flow data for all tokens.
        
        Args:
            session: Database session
            hours: Time window
            min_volume_usd: Minimum volume to include
            network: Filter by network
            
        Returns:
            List of FlowPriceData objects
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all swaps with pool info
        query = session.query(Swap, Pool).join(Pool).filter(
            Swap.timestamp >= since
        )
        
        if network:
            query = query.filter(Swap.network == network)
        
        swaps = query.all()
        
        # Aggregate by token
        token_data: dict[tuple[str, str], dict] = {}
        
        for swap, pool in swaps:
            # Process both tokens in the pair
            for token_addr, token_sym in [
                (pool.token0_address, pool.token0_symbol),
                (pool.token1_address, pool.token1_symbol),
            ]:
                key = (token_addr, pool.network)
                
                if key not in token_data:
                    token_data[key] = {
                        "token_address": token_addr,
                        "token_symbol": token_sym or "???",
                        "network": pool.network,
                        "inflow_usd": 0,
                        "outflow_usd": 0,
                        "swap_count": 0,
                    }
                
                amount = float(swap.amount_usd or 0) / 2  # Split between tokens
                token_data[key]["swap_count"] += 1
                
                if swap.direction == "buy":
                    token_data[key]["inflow_usd"] += amount
                else:
                    token_data[key]["outflow_usd"] += amount
        
        # Convert to FlowPriceData objects
        results = []
        
        for key, data in token_data.items():
            total_volume = data["inflow_usd"] + data["outflow_usd"]
            
            if total_volume < min_volume_usd:
                continue
            
            net_flow = data["inflow_usd"] - data["outflow_usd"]
            flow_ratio = net_flow / total_volume if total_volume > 0 else 0
            
            # Calculate percentile (simplified - compare to threshold)
            flow_percentile = self._calculate_flow_percentile(abs(net_flow), total_volume)
            
            # Determine if significant
            threshold_pct = self.settings.flow_price.significant_flow_percentile
            is_significant = flow_percentile >= threshold_pct
            
            if net_flow > 0:
                signal_type = "large_inflow" if is_significant else "none"
            else:
                signal_type = "large_outflow" if is_significant else "none"
            
            results.append(FlowPriceData(
                token_address=data["token_address"],
                token_symbol=data["token_symbol"],
                network=data["network"],
                inflow_usd=data["inflow_usd"],
                outflow_usd=data["outflow_usd"],
                net_flow_usd=net_flow,
                swap_count=data["swap_count"],
                flow_volume_ratio=flow_ratio,
                price_start=0,  # Would need price oracle
                price_end=0,
                price_change_percent=0,
                flow_percentile=flow_percentile,
                is_significant=is_significant,
                signal_type=signal_type,
            ))
        
        # Sort by absolute net flow
        results.sort(key=lambda x: abs(x.net_flow_usd), reverse=True)
        
        return results
    
    def _calculate_flow_percentile(
        self,
        net_flow: float,
        total_volume: float,
    ) -> float:
        """
        Calculate percentile of flow compared to typical.
        
        This is a simplified implementation. In production, would compare
        to historical data distribution.
        """
        # Simplified: use ratio thresholds
        if total_volume == 0:
            return 0
        
        ratio = abs(net_flow) / total_volume
        
        # Map ratio to percentile (empirical thresholds)
        if ratio > 0.8:  # 80%+ net flow = very unusual
            return 99
        elif ratio > 0.6:
            return 95
        elif ratio > 0.4:
            return 90
        elif ratio > 0.3:
            return 80
        elif ratio > 0.2:
            return 70
        elif ratio > 0.1:
            return 50
        else:
            return 30
    
    def get_historical_correlation(
        self,
        token_address: str,
        network: str,
        lookback_days: int = 30,
    ) -> dict:
        """
        Calculate historical correlation between flow and price.
        
        Returns:
            Dict with correlation coefficient and expected impact
        """
        # Placeholder - would need historical price data
        # In production, would:
        # 1. Get daily flow and price data
        # 2. Calculate correlation coefficient
        # 3. Build regression model: price_change = f(flow)
        
        return {
            "correlation": 0.3,  # Placeholder
            "expected_impact_per_100k": 0.5,  # 0.5% price change per $100k flow
        }
    
    def scan_for_flow_signals(
        self,
        hours: int = 24,
        min_percentile: float = 90,
        networks: Optional[list[str]] = None,
    ) -> list[FlowPriceAlert]:
        """
        Scan for significant flow events.
        
        Args:
            hours: Time window
            min_percentile: Minimum flow percentile to alert
            networks: Networks to scan
            
        Returns:
            List of FlowPriceAlert objects
        """
        alerts = []
        
        with session_scope() as session:
            if networks:
                for network in networks:
                    flows = self.get_token_flows(session, hours, network=network)
                    alerts.extend(self._generate_alerts(flows, min_percentile))
            else:
                flows = self.get_token_flows(session, hours)
                alerts.extend(self._generate_alerts(flows, min_percentile))
        
        return alerts
    
    def _generate_alerts(
        self,
        flows: list[FlowPriceData],
        min_percentile: float,
    ) -> list[FlowPriceAlert]:
        """Generate alerts from flow data."""
        alerts = []
        
        for flow in flows:
            if flow.flow_percentile < min_percentile:
                continue
            
            flow_type = "inflow" if flow.net_flow_usd > 0 else "outflow"
            
            # Estimate expected price change
            correlation = self.get_historical_correlation(
                flow.token_address, flow.network
            )
            expected_change = (
                abs(flow.net_flow_usd) / 100000 
                * correlation["expected_impact_per_100k"]
            )
            if flow.net_flow_usd < 0:
                expected_change = -expected_change
            
            message = (
                f"Significant {flow_type} detected for {flow.token_symbol}\n"
                f"Network: {flow.network}\n"
                f"Net flow: ${flow.net_flow_usd:+,.0f}\n"
                f"Flow percentile: {flow.flow_percentile:.0f}%\n"
                f"Inflow: ${flow.inflow_usd:,.0f} | Outflow: ${flow.outflow_usd:,.0f}\n"
                f"Expected price impact: {expected_change:+.2f}%"
            )
            
            alert = FlowPriceAlert(
                token_symbol=flow.token_symbol,
                token_address=flow.token_address,
                network=flow.network,
                flow_type=flow_type,
                net_flow_usd=flow.net_flow_usd,
                flow_percentile=flow.flow_percentile,
                expected_price_change=expected_change,
                message=message,
            )
            alerts.append(alert)
        
        return alerts
    
    def save_flow_signals(
        self,
        hours: int = 24,
        networks: Optional[list[str]] = None,
    ) -> int:
        """
        Scan and save flow signals.
        
        Returns:
            Number of signals created
        """
        alerts = self.scan_for_flow_signals(hours=hours, networks=networks)
        
        with session_scope() as session:
            for alert in alerts:
                signal = Signal(**alert.to_signal())
                session.add(signal)
            session.commit()
        
        logger.info(f"Created {len(alerts)} flow-price signals")
        return len(alerts)


def analyze_token_flows(
    hours: int = 24,
    networks: Optional[list[str]] = None,
) -> list[FlowPriceData]:
    """
    Convenience function to analyze token flows.
    
    Args:
        hours: Time window
        networks: Networks to analyze
        
    Returns:
        List of FlowPriceData objects
    """
    analyzer = FlowPriceAnalyzer()
    
    with session_scope() as session:
        if networks:
            results = []
            for network in networks:
                results.extend(analyzer.get_token_flows(session, hours, network=network))
            return results
        else:
            return analyzer.get_token_flows(session, hours)
