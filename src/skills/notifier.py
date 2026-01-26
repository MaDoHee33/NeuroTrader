"""
NeuroTrader Notifier
====================
Notification system for training events.

Supports:
- Telegram Bot
- Discord Webhook
- File-based logs (always on)
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of training events."""
    TRAINING_STARTED = "training_started"
    CHECKPOINT_SAVED = "checkpoint_saved"
    TRAINING_COMPLETE = "training_complete"
    MODEL_PROMOTED = "model_promoted"
    TRAINING_FAILED = "training_failed"
    EVALUATION_COMPLETE = "evaluation_complete"


@dataclass
class TrainingEvent:
    """A training event to notify about."""
    event_type: EventType
    role: str
    symbol: str
    timeframe: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class Notifier:
    """
    Multi-channel notification system.
    
    Usage:
        notifier = Notifier()
        notifier.send(TrainingEvent(
            event_type=EventType.TRAINING_COMPLETE,
            role="scalper",
            symbol="XAUUSD",
            timeframe="M5",
            message="Training completed!",
            metrics={"return": 15.5, "sharpe": 1.2}
        ))
    """
    
    def __init__(self, config_path: str = "config/notifications.json"):
        self.config_path = Path(config_path)
        self.log_dir = Path("logs/notifications")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load notification config from file or environment."""
        config = {
            'telegram': {
                'enabled': False,
                'bot_token': os.environ.get('TELEGRAM_BOT_TOKEN', ''),
                'chat_id': os.environ.get('TELEGRAM_CHAT_ID', '')
            },
            'discord': {
                'enabled': False,
                'webhook_url': os.environ.get('DISCORD_WEBHOOK_URL', '')
            },
            'file_log': {
                'enabled': True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                for channel, settings in file_config.items():
                    if channel in config:
                        config[channel].update(settings)
                        
        return config
    
    def send(self, event: TrainingEvent) -> bool:
        """
        Send notification through all enabled channels.
        
        Returns:
            True if at least one channel succeeded
        """
        success = False
        
        # Always log to file
        if self.config['file_log']['enabled']:
            self._log_to_file(event)
            success = True
        
        # Telegram
        if self.config['telegram']['enabled']:
            if self._send_telegram(event):
                success = True
                
        # Discord
        if self.config['discord']['enabled']:
            if self._send_discord(event):
                success = True
                
        return success
    
    def _format_message(self, event: TrainingEvent) -> str:
        """Format event into readable message."""
        emoji_map = {
            EventType.TRAINING_STARTED: "🚀",
            EventType.CHECKPOINT_SAVED: "💾",
            EventType.TRAINING_COMPLETE: "✅",
            EventType.MODEL_PROMOTED: "🏆",
            EventType.TRAINING_FAILED: "❌",
            EventType.EVALUATION_COMPLETE: "📊"
        }
        
        emoji = emoji_map.get(event.event_type, "📢")
        
        lines = [
            f"{emoji} *NeuroTrader Alert*",
            f"Event: {event.event_type.value.replace('_', ' ').title()}",
            f"Role: {event.role.upper()}",
            f"Symbol: {event.symbol} ({event.timeframe})",
            "",
            event.message
        ]
        
        if event.metrics:
            lines.append("")
            lines.append("📈 Metrics:")
            for key, value in event.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  • {key}: {value:.2f}")
                else:
                    lines.append(f"  • {key}: {value}")
        
        lines.append(f"\n🕐 {event.timestamp[:19]}")
        
        return "\n".join(lines)
    
    def _log_to_file(self, event: TrainingEvent):
        """Log event to file."""
        log_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        log_entry = {
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'role': event.role,
            'symbol': event.symbol,
            'timeframe': event.timeframe,
            'message': event.message,
            'metrics': event.metrics
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _send_telegram(self, event: TrainingEvent) -> bool:
        """Send via Telegram Bot API."""
        token = self.config['telegram']['bot_token']
        chat_id = self.config['telegram']['chat_id']
        
        if not token or not chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        message = self._format_message(event)
        
        try:
            response = requests.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def _send_discord(self, event: TrainingEvent) -> bool:
        """Send via Discord Webhook."""
        webhook_url = self.config['discord']['webhook_url']
        
        if not webhook_url:
            return False
        
        message = self._format_message(event).replace('*', '**')  # Discord bold
        
        try:
            response = requests.post(webhook_url, json={
                'content': message
            }, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            print(f"Discord error: {e}")
            return False
    
    # Convenience methods
    def training_started(self, role: str, symbol: str, timeframe: str, steps: int):
        """Send training started notification."""
        self.send(TrainingEvent(
            event_type=EventType.TRAINING_STARTED,
            role=role,
            symbol=symbol,
            timeframe=timeframe,
            message=f"Training started for {steps:,} steps"
        ))
    
    def training_complete(self, role: str, symbol: str, timeframe: str, 
                         metrics: Dict[str, Any]):
        """Send training complete notification."""
        ret = metrics.get('total_return', 0)
        msg = f"Training complete! Return: {ret:.2f}%"
        
        self.send(TrainingEvent(
            event_type=EventType.TRAINING_COMPLETE,
            role=role,
            symbol=symbol,
            timeframe=timeframe,
            message=msg,
            metrics=metrics
        ))
    
    def model_promoted(self, role: str, version: int, metric: str, value: float):
        """Send model promoted notification."""
        self.send(TrainingEvent(
            event_type=EventType.MODEL_PROMOTED,
            role=role,
            symbol="",
            timeframe="",
            message=f"🏆 New BEST model! Version {version}",
            metrics={metric: value}
        ))
    
    def training_failed(self, role: str, symbol: str, timeframe: str, error: str):
        """Send training failed notification."""
        self.send(TrainingEvent(
            event_type=EventType.TRAINING_FAILED,
            role=role,
            symbol=symbol,
            timeframe=timeframe,
            message=f"Training failed: {error}"
        ))


# Quick test
if __name__ == "__main__":
    notifier = Notifier()
    
    # Test notification
    notifier.training_complete(
        role="scalper",
        symbol="XAUUSD",
        timeframe="M5",
        metrics={
            'total_return': 15.5,
            'sharpe_ratio': 1.23,
            'win_rate': 65.0
        }
    )
    print("✅ Test notification logged to file")
