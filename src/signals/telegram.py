from __future__ import annotations

"""
Telegram notification service.

Sends alerts and signals to Telegram chat.
"""

import logging
from datetime import datetime
from typing import Optional

import requests

from config.settings import get_settings
from src.db.models import Signal
from src.db.database import session_scope


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends notifications to Telegram.
    
    Setup:
    1. Create bot via @BotFather
    2. Get chat ID (send message to bot, then check getUpdates)
    3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.bot_token = bot_token or settings.telegram.bot_token
        self.chat_id = chat_id or settings.telegram.chat_id
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning(
                "Telegram notifications disabled. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable."
            )
    
    @property
    def api_url(self) -> str:
        """Telegram Bot API URL."""
        return f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message to the configured chat.
        
        Args:
            text: Message text (supports HTML or Markdown)
            parse_mode: "HTML" or "Markdown"
            disable_notification: Send silently
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {text[:100]}...")
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_notification": disable_notification,
                },
                timeout=10,
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get("ok"):
                logger.error(f"Telegram API error: {result}")
                return False
            
            logger.info(f"Telegram message sent: {text[:50]}...")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_signal(self, signal: Signal) -> bool:
        """
        Send a Signal object as Telegram message.
        
        Args:
            signal: Signal model instance
            
        Returns:
            True if sent successfully
        """
        # Format message with emoji based on severity
        severity_emoji = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️",
        }
        emoji = severity_emoji.get(signal.severity, "📊")
        
        # Build message
        lines = [
            f"{emoji} <b>{signal.title}</b>",
            "",
            signal.message,
        ]
        
        if signal.amount_usd:
            lines.append(f"\n💰 Amount: ${float(signal.amount_usd):,.0f}")
        
        if signal.network:
            lines.append(f"🌐 Network: {signal.network}")
        
        text = "\n".join(lines)
        
        success = self.send_message(text)
        
        if success:
            # Update signal status
            with session_scope() as session:
                db_signal = session.query(Signal).get(signal.id)
                if db_signal:
                    db_signal.is_sent = True
                    db_signal.sent_at = datetime.utcnow()
                    session.commit()
        
        return success
    
    def send_pending_signals(self, max_count: int = 10) -> int:
        """
        Send all pending (unsent) signals.
        
        Args:
            max_count: Maximum signals to send in one batch
            
        Returns:
            Number of signals sent
        """
        sent = 0
        
        with session_scope() as session:
            signals = session.query(Signal).filter(
                Signal.is_sent == False
            ).order_by(
                Signal.created_at.desc()
            ).limit(max_count).all()
            
            for signal in signals:
                if self.send_signal(signal):
                    sent += 1
        
        if sent > 0:
            logger.info(f"Sent {sent} pending signals")
        
        return sent
    
    def send_daily_summary(
        self,
        pools_count: int,
        swaps_count: int,
        alerts_count: int,
        top_outflows: list[dict],
    ) -> bool:
        """
        Send daily summary report.
        
        Args:
            pools_count: Number of pools tracked
            swaps_count: Number of swaps in last 24h
            alerts_count: Number of alerts generated
            top_outflows: Top outflow events
        """
        lines = [
            "📊 <b>Daily Summary</b>",
            "",
            f"• Pools tracked: {pools_count:,}",
            f"• Swaps (24h): {swaps_count:,}",
            f"• Alerts: {alerts_count}",
        ]
        
        if top_outflows:
            lines.append("\n<b>Top Outflows:</b>")
            for i, outflow in enumerate(top_outflows[:5], 1):
                lines.append(
                    f"{i}. {outflow.get('token_symbol', '???')} on {outflow.get('network', '???')}: "
                    f"${abs(outflow.get('net_flow_usd', 0)):,.0f}"
                )
        
        return self.send_message("\n".join(lines))


def send_telegram_message(text: str) -> bool:
    """
    Convenience function to send a Telegram message.
    
    Args:
        text: Message text
        
    Returns:
        True if sent successfully
    """
    notifier = TelegramNotifier()
    return notifier.send_message(text)
