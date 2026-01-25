#!/usr/bin/env python3
"""
Automatic scheduler for LP strategy analysis.

Runs periodic tasks:
- Pool updates (every 6 hours)
- Swap loading (every hour)
- Outflow analysis (every 30 minutes)
- Flow analysis (every hour)
- Watched owner monitoring (every 10 minutes)
- Telegram notifications (continuous)
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from threading import Thread

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import schedule

from src.db.database import init_db, session_scope
from src.db.models import Pool, Swap, Position, WatchedOwner
from src.data.pools import PoolLoader
from src.data.swaps import SwapLoader
from src.analytics.capital_flow import detect_large_outflows
from src.analytics.flow_price import FlowPriceAnalyzer
from src.analytics.new_tokens import NewTokenAnalyzer
from src.signals.telegram import TelegramNotifier, send_telegram_message
from src.signals.bot import TelegramBot
from src.utils.helpers import setup_logging
from config.settings import get_settings


logger = logging.getLogger(__name__)


def job_update_pools():
    """Update pool data from all networks."""
    logger.info("Running job: update_pools")
    try:
        loader = PoolLoader()
        results = loader.load_all_pools(min_tvl=100000)
        total = sum(results.values())
        logger.info(f"Updated {total} pools")
    except Exception as e:
        logger.error(f"job_update_pools failed: {e}")


def job_load_swaps():
    """Load recent swaps for top pools."""
    logger.info("Running job: load_swaps")
    try:
        with session_scope() as session:
            pools = session.query(Pool).filter(
                Pool.tvl_usd >= 500000
            ).order_by(Pool.tvl_usd.desc()).limit(50).all()
            
            loader = SwapLoader()
            total = 0
            
            for pool in pools:
                try:
                    count = loader.load_swaps_for_pool(session, pool, limit=50)
                    total += count
                except Exception:
                    pass
            
            logger.info(f"Loaded {total} swaps")
    except Exception as e:
        logger.error(f"job_load_swaps failed: {e}")


def job_analyze_outflows():
    """Analyze capital outflows and send alerts."""
    logger.info("Running job: analyze_outflows")
    try:
        alerts = detect_large_outflows(hours=1, save_signals=True)
        
        if alerts:
            logger.info(f"Found {len(alerts)} outflow alerts")
            
            # Send critical/warning alerts via Telegram
            notifier = TelegramNotifier()
            for alert in alerts:
                if alert.severity in ("critical", "warning"):
                    message = (
                        f"⚠️ <b>Outflow Alert</b>\n\n"
                        f"{alert.token0_symbol}/{alert.token1_symbol} ({alert.network})\n"
                        f"Net outflow: ${abs(alert.net_flow_usd):,.0f}\n"
                        f"TVL: ${alert.tvl_usd:,.0f} ({alert.outflow_percent_of_tvl:.1f}%)"
                    )
                    notifier.send_message(message)
        else:
            logger.info("No significant outflows")
    except Exception as e:
        logger.error(f"job_analyze_outflows failed: {e}")


def job_analyze_flows():
    """Analyze token flows and send alerts."""
    logger.info("Running job: analyze_flows")
    try:
        analyzer = FlowPriceAnalyzer()
        count = analyzer.save_flow_signals(hours=1)
        logger.info(f"Created {count} flow signals")
    except Exception as e:
        logger.error(f"job_analyze_flows failed: {e}")


def job_scan_new_pools():
    """Scan for new pool opportunities."""
    logger.info("Running job: scan_new_pools")
    try:
        analyzer = NewTokenAnalyzer()
        count = analyzer.save_opportunities_as_signals()
        logger.info(f"Created {count} new pool signals")
    except Exception as e:
        logger.error(f"job_scan_new_pools failed: {e}")


def job_send_pending_signals():
    """Send pending Telegram notifications."""
    try:
        notifier = TelegramNotifier()
        if notifier.enabled:
            sent = notifier.send_pending_signals(max_count=5)
            if sent > 0:
                logger.info(f"Sent {sent} pending signals")
    except Exception as e:
        logger.error(f"job_send_pending_signals failed: {e}")


def job_monitor_watched_owners():
    """
    Monitor watched owners for position changes every 10 minutes.
    
    For each watched owner:
    - Count current positions
    - Compare with last_position_count
    - If different: detect new/closed positions and send Telegram alert
    - Update last_position_count and last_checked_at
    """
    logger.info("Running job: monitor_watched_owners")
    
    try:
        settings = get_settings()
        notifier = TelegramNotifier()
        changes_detected = 0
        
        with session_scope() as session:
            # Get all watched owners
            watched_list = session.query(WatchedOwner).all()
            
            if not watched_list:
                logger.debug("No watched owners to monitor")
                return
            
            for watched in watched_list:
                try:
                    # Count current positions for this owner
                    current_total = session.query(Position).filter(
                        Position.owner_address == watched.owner_address
                    ).count()
                    
                    current_open = session.query(Position).filter(
                        Position.owner_address == watched.owner_address,
                        Position.is_closed == False
                    ).count()
                    
                    current_closed = session.query(Position).filter(
                        Position.owner_address == watched.owner_address,
                        Position.is_closed == True
                    ).count()
                    
                    last_count = watched.last_position_count or 0
                    
                    # Check if there's a change
                    if current_total != last_count:
                        changes_detected += 1
                        diff = current_total - last_count
                        
                        # Determine what happened
                        if diff > 0:
                            # New positions opened
                            action_text = f"🆕 открыл {diff} новых позиций"
                            action_type = "new"
                        else:
                            # Positions closed
                            action_text = f"🔒 закрыл {abs(diff)} позиций"
                            action_type = "closed"
                        
                        # Get owner address short format
                        addr_short = f"{watched.owner_address[:10]}...{watched.owner_address[-6:]}"
                        
                        # Build message
                        message_lines = [
                            f"👁️ <b>LP Мониторинг</b>",
                            "",
                            f"Владелец: <code>{addr_short}</code>",
                            f"Действие: {action_text}",
                            "",
                            f"📊 Текущие позиции:",
                            f"• Всего: {current_total}",
                            f"• Открытых: {current_open}",
                            f"• Закрытых: {current_closed}",
                        ]
                        
                        if watched.note:
                            message_lines.insert(3, f"📝 Заметка: {watched.note}")
                        
                        message = "\n".join(message_lines)
                        
                        logger.info(
                            f"Watched owner {addr_short}: {action_text} "
                            f"(was {last_count}, now {current_total})"
                        )
                        
                        # Send Telegram notification if enabled
                        if watched.notify_telegram and notifier.enabled:
                            success = notifier.send_message(message)
                            if success:
                                logger.info(f"Telegram alert sent for {addr_short}")
                            else:
                                logger.warning(f"Failed to send Telegram alert for {addr_short}")
                        
                        # Update the watched owner record
                        watched.last_position_count = current_total
                        watched.last_checked_at = datetime.now()
                    else:
                        # No change, just update last_checked_at
                        watched.last_checked_at = datetime.now()
                
                except Exception as e:
                    logger.error(f"Error monitoring owner {watched.owner_address}: {e}")
                    continue
            
            # Commit all updates
            session.commit()
        
        if changes_detected > 0:
            logger.info(f"Detected changes for {changes_detected} watched owners")
        else:
            logger.debug("No changes detected for watched owners")
    
    except Exception as e:
        logger.error(f"job_monitor_watched_owners failed: {e}")
        import traceback
        traceback.print_exc()


def run_scheduler():
    """Run the scheduler."""
    setup_logging()
    init_db()
    
    settings = get_settings()
    
    logger.info("Starting scheduler...")
    
    # Schedule jobs
    schedule.every(6).hours.do(job_update_pools)
    schedule.every(1).hour.do(job_load_swaps)
    schedule.every(30).minutes.do(job_analyze_outflows)
    schedule.every(1).hour.do(job_analyze_flows)
    schedule.every(6).hours.do(job_scan_new_pools)
    schedule.every(5).minutes.do(job_send_pending_signals)
    schedule.every(10).minutes.do(job_monitor_watched_owners)  # Monitor watched owners
    
    # Run initial jobs
    logger.info("Running initial jobs...")
    job_update_pools()
    job_load_swaps()
    job_analyze_outflows()
    job_monitor_watched_owners()  # Check watched owners on startup
    
    # Start Telegram bot in background
    if settings.telegram.enabled:
        logger.info("Starting Telegram bot...")
        bot = TelegramBot()
        bot_thread = bot.start_polling_thread()
    
    # Run scheduler loop
    logger.info("Scheduler running. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    run_scheduler()
