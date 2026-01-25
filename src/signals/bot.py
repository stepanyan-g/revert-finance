"""
Telegram bot with commands for LP strategy.

Commands:
- /start - Welcome message
- /status - System status
- /pools - Pool statistics
- /outflows - Check for large outflows
- /flows - Token flow analysis
- /new_pools - New pool opportunities
- /top_owners - Top LP owners
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional

import requests

from config.settings import get_settings
from src.db.database import session_scope, init_db
from src.db.models import Pool, Swap, Signal, Position, Owner


logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot with commands.
    
    Uses polling (not webhooks) for simplicity.
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.bot_token = bot_token or settings.telegram.bot_token
        self.chat_id = chat_id or settings.telegram.chat_id
        self.enabled = bool(self.bot_token)
        
        self._running = False
        self._last_update_id = 0
        
        if not self.enabled:
            logger.warning("Telegram bot disabled. Set TELEGRAM_BOT_TOKEN.")
    
    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send message to chat."""
        if not self.enabled:
            return False
        
        target_chat = chat_id or self.chat_id
        if not target_chat:
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": target_chat,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            return response.json().get("ok", False)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def get_updates(self, offset: int = 0) -> list:
        """Get updates from Telegram."""
        try:
            response = requests.get(
                f"{self.api_url}/getUpdates",
                params={
                    "offset": offset,
                    "timeout": 30,
                },
                timeout=35,
            )
            result = response.json()
            return result.get("result", [])
        except Exception as e:
            logger.error(f"Failed to get updates: {e}")
            return []
    
    def handle_command(self, message: dict) -> Optional[str]:
        """Handle incoming command."""
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        
        if not text.startswith("/"):
            return None
        
        command = text.split()[0].lower()
        args = text.split()[1:] if len(text.split()) > 1 else []
        
        handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/pools": self._cmd_pools,
            "/outflows": self._cmd_outflows,
            "/flows": self._cmd_flows,
            "/new_pools": self._cmd_new_pools,
            "/top_owners": self._cmd_top_owners,
        }
        
        handler = handlers.get(command)
        if handler:
            return handler(args)
        else:
            return "Unknown command. Use /help to see available commands."
    
    def _cmd_start(self, args: list) -> str:
        """Welcome message."""
        return (
            "🔄 <b>Revert LP Strategy Bot</b>\n\n"
            "I monitor DeFi liquidity pools and alert you about:\n"
            "• Large capital outflows\n"
            "• New pool opportunities\n"
            "• Token flow patterns\n"
            "• Top performing LP owners\n\n"
            "Use /help to see available commands."
        )
    
    def _cmd_help(self, args: list) -> str:
        """Help message."""
        return (
            "📋 <b>Available Commands</b>\n\n"
            "/status - System status\n"
            "/pools [network] - Pool statistics\n"
            "/outflows [hours] - Check for large outflows\n"
            "/flows [hours] - Token flow analysis\n"
            "/new_pools - New pool opportunities\n"
            "/top_owners [n] - Top N LP owners\n"
            "/help - This message"
        )
    
    def _cmd_status(self, args: list) -> str:
        """System status."""
        with session_scope() as session:
            pool_count = session.query(Pool).count()
            swap_count = session.query(Swap).count()
            position_count = session.query(Position).count()
            owner_count = session.query(Owner).count()
            signal_count = session.query(Signal).count()
        
        return (
            "📊 <b>System Status</b>\n\n"
            f"Pools: {pool_count:,}\n"
            f"Swaps: {swap_count:,}\n"
            f"Positions: {position_count:,}\n"
            f"Owners: {owner_count:,}\n"
            f"Signals: {signal_count:,}"
        )
    
    def _cmd_pools(self, args: list) -> str:
        """Pool statistics."""
        from sqlalchemy import func
        
        network_filter = args[0] if args else None
        
        with session_scope() as session:
            query = session.query(
                Pool.network,
                func.count(Pool.id),
                func.sum(Pool.tvl_usd)
            ).group_by(Pool.network)
            
            if network_filter:
                query = query.filter(Pool.network == network_filter)
            
            results = query.all()
        
        if not results:
            return "No pools found."
        
        lines = ["📊 <b>Pool Statistics</b>\n"]
        for network, count, tvl in results:
            tvl_str = f"${float(tvl or 0):,.0f}"
            lines.append(f"• {network}: {count} pools, TVL: {tvl_str}")
        
        return "\n".join(lines)
    
    def _cmd_outflows(self, args: list) -> str:
        """Check for large outflows."""
        from src.analytics.capital_flow import detect_large_outflows
        
        hours = int(args[0]) if args else 24
        
        alerts = detect_large_outflows(hours=hours, save_signals=False)
        
        if not alerts:
            return f"✅ No significant outflows detected in the last {hours}h."
        
        lines = [f"⚠️ <b>Outflows ({hours}h)</b>\n"]
        
        for alert in alerts[:10]:
            severity_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(alert.severity, "⚪")
            lines.append(
                f"{severity_icon} {alert.token0_symbol}/{alert.token1_symbol} ({alert.network})\n"
                f"   ${abs(alert.net_flow_usd):,.0f} ({alert.outflow_percent_of_tvl:.1f}% TVL)"
            )
        
        if len(alerts) > 10:
            lines.append(f"\n... and {len(alerts) - 10} more")
        
        return "\n".join(lines)
    
    def _cmd_flows(self, args: list) -> str:
        """Token flow analysis."""
        from src.analytics.flow_price import analyze_token_flows
        
        hours = int(args[0]) if args else 24
        
        flows = analyze_token_flows(hours=hours)
        
        if not flows:
            return "No flow data available."
        
        # Top 10 by absolute flow
        top_flows = sorted(flows, key=lambda x: abs(x.net_flow_usd), reverse=True)[:10]
        
        lines = [f"💰 <b>Token Flows ({hours}h)</b>\n"]
        
        for flow in top_flows:
            direction = "📈" if flow.net_flow_usd > 0 else "📉"
            lines.append(
                f"{direction} {flow.token_symbol} ({flow.network})\n"
                f"   Net: ${flow.net_flow_usd:+,.0f}"
            )
        
        return "\n".join(lines)
    
    def _cmd_new_pools(self, args: list) -> str:
        """New pool opportunities."""
        from src.analytics.new_tokens import get_new_pools
        
        pools = get_new_pools(max_age_days=7)
        
        if not pools:
            return "No new pools found (last 7 days)."
        
        lines = ["🆕 <b>New Pools (7d)</b>\n"]
        
        for pool in pools[:10]:
            risk_icon = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "very_high": "🔴"
            }.get(pool.risk_level, "⚪")
            
            lines.append(
                f"{risk_icon} {pool.token0_symbol}/{pool.token1_symbol} ({pool.network})\n"
                f"   TVL: ${pool.tvl_usd:,.0f}, Age: {pool.age_days:.1f}d"
            )
        
        return "\n".join(lines)
    
    def _cmd_top_owners(self, args: list) -> str:
        """Top LP owners."""
        from src.analytics.owners import get_top_lp_owners
        
        limit = int(args[0]) if args else 10
        
        owners = get_top_lp_owners(limit=limit)
        
        if not owners:
            return "No owner data available."
        
        lines = ["🏆 <b>Top LP Owners</b>\n"]
        
        for i, owner in enumerate(owners, 1):
            pnl_sign = "+" if owner.total_pnl_usd >= 0 else ""
            lines.append(
                f"{i}. {owner.address[:8]}...{owner.address[-4:]}\n"
                f"   PnL: {pnl_sign}${owner.total_pnl_usd:,.0f}, "
                f"Win: {owner.win_rate*100:.0f}%"
            )
        
        return "\n".join(lines)
    
    def start_polling(self) -> None:
        """Start polling for updates."""
        if not self.enabled:
            logger.error("Bot not enabled. Set TELEGRAM_BOT_TOKEN.")
            return
        
        self._running = True
        logger.info("Starting Telegram bot polling...")
        
        while self._running:
            try:
                updates = self.get_updates(offset=self._last_update_id + 1)
                
                for update in updates:
                    self._last_update_id = update.get("update_id", 0)
                    
                    message = update.get("message", {})
                    if not message:
                        continue
                    
                    chat_id = message.get("chat", {}).get("id")
                    
                    response = self.handle_command(message)
                    if response:
                        self.send_message(response, chat_id=str(chat_id))
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(5)
    
    def stop(self) -> None:
        """Stop polling."""
        self._running = False
    
    def start_polling_thread(self) -> threading.Thread:
        """Start polling in a background thread."""
        thread = threading.Thread(target=self.start_polling, daemon=True)
        thread.start()
        return thread


def run_bot():
    """Run the Telegram bot."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.helpers import setup_logging
    setup_logging()
    
    init_db()
    
    bot = TelegramBot()
    
    if not bot.enabled:
        print("Telegram bot not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
    
    print("Starting Telegram bot... Press Ctrl+C to stop.")
    
    try:
        bot.start_polling()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()


if __name__ == "__main__":
    run_bot()
