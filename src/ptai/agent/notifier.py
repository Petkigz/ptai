"""
Notifier - Premium notifications: Desktop, Discord, Telegram, Email
Local only, no cloud required, but supports webhooks for product
"""
import platform
import os
import requests
from typing import Optional, List
from loguru import logger
from pathlib import Path

class Notifier:
    def __init__(self, enabled: bool = True, vault=None):
        self.enabled = enabled
        self.system = platform.system()
        self.vault = vault
        # Load webhook URLs from env or vault
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        logger.info(f"Notifier enabled={enabled} system={self.system} discord={bool(self.discord_webhook)} telegram={bool(self.telegram_bot_token)}")
    
    def notify(self, title: str, message: str, urgency: str = "normal", channels: List[str] = None):
        """Send notification to all enabled channels"""
        if not self.enabled:
            return
        
        channels = channels or ["desktop", "log"]
        if self.discord_webhook:
            channels.append("discord")
        if self.telegram_bot_token and self.telegram_chat_id:
            channels.append("telegram")
        
        # Desktop
        if "desktop" in channels:
            self._notify_desktop(title, message, urgency)
        
        # Discord webhook
        if "discord" in channels and self.discord_webhook:
            self._notify_discord(title, message, urgency)
        
        # Telegram
        if "telegram" in channels and self.telegram_bot_token:
            self._notify_telegram(title, message)
        
        # Always log
        logger.info(f"NOTIFY [{urgency}] {title}: {message} via {channels}")
        
        # Save to notification history
        try:
            Path("./data").mkdir(exist_ok=True)
            with open("./data/notifications.log", "a") as f:
                f.write(f"{title} | {message} | {urgency} | {channels}\n")
        except:
            pass
    
    def _notify_desktop(self, title: str, message: str, urgency: str):
        try:
            if self.system == "Linux":
                import subprocess
                subprocess.run(["notify-send", title, message, "-u", urgency], timeout=2)
            elif self.system == "Darwin":
                import subprocess
                script = f'display notification "{message}" with title "{title}"'
                subprocess.run(["osascript", "-e", script], timeout=2)
            elif self.system == "Windows":
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=5, threaded=True)
                except ImportError:
                    logger.debug("win10toast not installed")
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")
    
    def _notify_discord(self, title: str, message: str, urgency: str):
        try:
            color = 0x10b981 if urgency == "normal" else 0xef4444 if urgency == "critical" else 0xf59e0b
            payload = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color,
                    "footer": {"text": "PTAI - Autonomous Trading Agent"}
                }]
            }
            requests.post(self.discord_webhook, json=payload, timeout=5)
            logger.debug("Discord notified")
        except Exception as e:
            logger.debug(f"Discord notify failed: {e}")
    
    def _notify_telegram(self, title: str, message: str):
        try:
            text = f"*{title}*\n{message}"
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"}
            requests.post(url, json=payload, timeout=5)
            logger.debug("Telegram notified")
        except Exception as e:
            logger.debug(f"Telegram notify failed: {e}")
    
    def opportunity_found(self, question: str, edge: float, size_usd: float):
        self.notify(
            title="🎯 PTAI Opportunity",
            message=f"{question[:80]}\nEdge {edge:.1%} | Size ${size_usd:.2f}\nKelly 6% max prevents wipeout",
            urgency="normal"
        )
    
    def trade_executed(self, question: str, side: str, amount: float, status: str):
        self.notify(
            title=f"⚡ PTAI Trade {status}",
            message=f"{side} ${amount:.2f} on {question[:60]}\nStatus: {status}",
            urgency="normal" if "executed" in status else "critical"
        )
    
    def shutdown_alert(self, reason: str, bankroll: float):
        self.notify(
            title="🚨 PTAI SHUTDOWN",
            message=f"Shutdown: {reason}\nBankroll ${bankroll:.2f}\nSelf-preservation triggered - earn $5/day or shutdown",
            urgency="critical"
        )
    
    def error_alert(self, error: str):
        self.notify(
            title="❌ PTAI Error",
            message=error[:300],
            urgency="critical"
        )
    
    def teammate_update(self, teammate: str, task: str, result: str):
        self.notify(
            title=f"🤖 {teammate} Finished Work",
            message=f"Task: {task[:80]}\nResult: {result[:200]}",
            urgency="normal",
            channels=["log", "discord"]  # don't spam desktop for teammate updates
        )
