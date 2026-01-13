"""
Alert System for Real-Time Fraud Detection
Handles notifications, alerts, and communication for fraud events
"""

import smtplib
import json
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    LOG = "log"

@dataclass
class AlertConfig:
    """Configuration for alert system"""
    email_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = False
    dashboard_enabled: bool = True
    log_enabled: bool = True
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    # SMS settings (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    sms_recipients: List[str] = None
    
    # Webhook settings
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = None
    
    # Alert thresholds
    min_confidence_threshold: float = 0.5
    min_amount_threshold: float = 100.0
    rate_limit_per_minute: int = 10

class FraudAlert:
    """Fraud alert data structure"""
    
    def __init__(self, transaction_id: str, user_id: str, amount: float, 
                 risk_level: str, confidence_score: float, risk_factors: List[str],
                 timestamp: datetime = None):
        self.alert_id = f"alert_{int(time.time())}_{transaction_id}"
        self.transaction_id = transaction_id
        self.user_id = user_id
        self.amount = amount
        self.risk_level = risk_level
        self.confidence_score = confidence_score
        self.risk_factors = risk_factors
        self.timestamp = timestamp or datetime.now()
        self.status = "active"
        self.notifications_sent = []
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'transaction_id': self.transaction_id,
            'user_id': self.user_id,
            'amount': self.amount,
            'risk_level': self.risk_level,
            'confidence_score': self.confidence_score,
            'risk_factors': self.risk_factors,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'notifications_sent': self.notifications_sent
        }
    
    def to_email_html(self) -> str:
        """Convert alert to HTML email format"""
        risk_color = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }.get(self.risk_level, '#6c757d')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ border-left: 5px solid {risk_color}; padding: 15px; margin: 10px 0; }}
                .header {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
                .risk-factors {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>[ALERT] Fraud Alert - {self.risk_level.upper()}</h2>
                <p><strong>Alert ID:</strong> {self.alert_id}</p>
                <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="alert">
                <h3>Transaction Details</h3>
                <ul>
                    <li><strong>Transaction ID:</strong> {self.transaction_id}</li>
                    <li><strong>User ID:</strong> {self.user_id}</li>
                    <li><strong>Amount:</strong> ${self.amount:,.2f}</li>
                    <li><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{self.risk_level.upper()}</span></li>
                    <li><strong>Confidence Score:</strong> {self.confidence_score:.3f}</li>
                </ul>
            </div>
            
            <div class="risk-factors">
                <h4>Risk Factors Identified:</h4>
                <ul>
                    {''.join([f'<li>{factor}</li>' for factor in self.risk_factors])}
                </ul>
            </div>
            
            <div class="footer">
                <p>This is an automated alert from the Real-Time Fraud Detection System.</p>
                <p>Please review this transaction and take appropriate action if necessary.</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def to_sms_text(self) -> str:
        """Convert alert to SMS text format"""
        return f"""[ALERT] FRAUD ALERT
TX: {self.transaction_id}
User: {self.user_id}
Amount: ${self.amount:,.2f}
Risk: {self.risk_level.upper()}
Confidence: {self.confidence_score:.2f}
Time: {self.timestamp.strftime('%H:%M:%S')}
"""

class AlertManager:
    """Manages fraud alerts and notifications"""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alert_history = []
        self.rate_limiter = {}
        self.callbacks = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.DASHBOARD: self._log_dashboard_alert,
            AlertChannel.LOG: self._log_alert
        }
    
    def create_alert(self, transaction_id: str, user_id: str, amount: float,
                    risk_level: str, confidence_score: float, risk_factors: List[str]) -> FraudAlert:
        """Create a new fraud alert"""
        alert = FraudAlert(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            risk_level=risk_level,
            confidence_score=confidence_score,
            risk_factors=risk_factors
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.info(f"Fraud alert created: {alert.alert_id}")
        return alert
    
    def send_alert(self, alert: FraudAlert) -> Dict[str, bool]:
        """Send alert through all configured channels"""
        results = {}
        
        # Check rate limiting
        if not self._check_rate_limit(alert):
            logger.warning(f"Alert rate limited: {alert.alert_id}")
            return {channel.value: False for channel in AlertChannel}
        
        # Check if alert meets minimum thresholds
        if not self._meets_thresholds(alert):
            logger.info(f"Alert below thresholds: {alert.alert_id}")
            return {channel.value: False for channel in AlertChannel}
        
        # Send through each enabled channel
        for channel in AlertChannel:
            if self._is_channel_enabled(channel):
                try:
                    success = self.callbacks[channel](alert)
                    results[channel.value] = success
                    if success:
                        alert.notifications_sent.append(channel.value)
                except Exception as e:
                    logger.error(f"Error sending {channel.value} alert: {e}")
                    results[channel.value] = False
            else:
                results[channel.value] = False
        
        return results
    
    def _check_rate_limit(self, alert: FraudAlert) -> bool:
        """Check if alert is within rate limits"""
        current_time = datetime.now()
        minute_key = current_time.strftime('%Y-%m-%d-%H-%M')
        
        if minute_key not in self.rate_limiter:
            self.rate_limiter[minute_key] = 0
        
        if self.rate_limiter[minute_key] >= self.config.rate_limit_per_minute:
            return False
        
        self.rate_limiter[minute_key] += 1
        return True
    
    def _meets_thresholds(self, alert: FraudAlert) -> bool:
        """Check if alert meets minimum thresholds"""
        return (alert.confidence_score >= self.config.min_confidence_threshold and
                alert.amount >= self.config.min_amount_threshold)
    
    def _is_channel_enabled(self, channel: AlertChannel) -> bool:
        """Check if channel is enabled"""
        if channel == AlertChannel.EMAIL:
            return self.config.email_enabled
        elif channel == AlertChannel.SMS:
            return self.config.sms_enabled
        elif channel == AlertChannel.WEBHOOK:
            return self.config.webhook_enabled
        elif channel == AlertChannel.DASHBOARD:
            return self.config.dashboard_enabled
        elif channel == AlertChannel.LOG:
            return self.config.log_enabled
        return False
    
    def _send_email_alert(self, alert: FraudAlert) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[ALERT] Fraud Alert - {alert.risk_level.upper()} - {alert.transaction_id}"
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients or [])
            
            # Create HTML content
            html_content = alert.to_email_html()
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_sms_alert(self, alert: FraudAlert) -> bool:
        """Send SMS alert using Twilio"""
        try:
            from twilio.rest import Client
            
            client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
            message_body = alert.to_sms_text()
            
            for recipient in self.config.sms_recipients or []:
                message = client.messages.create(
                    body=message_body,
                    from_=self.config.twilio_phone_number,
                    to=recipient
                )
                logger.info(f"SMS alert sent: {alert.alert_id} to {recipient}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False
    
    def _send_webhook_alert(self, alert: FraudAlert) -> bool:
        """Send webhook alert"""
        try:
            payload = alert.to_dict()
            headers = self.config.webhook_headers or {'Content-Type': 'application/json'}
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _log_dashboard_alert(self, alert: FraudAlert) -> bool:
        """Log alert for dashboard display"""
        try:
            # This would typically push to a real-time database or message queue
            # For now, we'll just log it
            logger.info(f"Dashboard alert logged: {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log dashboard alert: {e}")
            return False
    
    def _log_alert(self, alert: FraudAlert) -> bool:
        """Log alert to file/system logs"""
        try:
            alert_data = alert.to_dict()
            logger.info(f"FRAUD ALERT: {json.dumps(alert_data, indent=2)}")
            return True
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False
    
    def get_recent_alerts(self, limit: int = 50) -> List[FraudAlert]:
        """Get recent fraud alerts"""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alerts_by_risk_level(self, risk_level: str) -> List[FraudAlert]:
        """Get alerts filtered by risk level"""
        return [alert for alert in self.alert_history if alert.risk_level == risk_level]
    
    def get_alerts_by_user(self, user_id: str) -> List[FraudAlert]:
        """Get alerts for a specific user"""
        return [alert for alert in self.alert_history if alert.user_id == user_id]
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'alerts_by_risk_level': {},
                'alerts_last_hour': 0,
                'alerts_last_day': 0
            }
        
        # Count by risk level
        risk_counts = {}
        for alert in self.alert_history:
            risk_counts[alert.risk_level] = risk_counts.get(alert.risk_level, 0) + 1
        
        # Count recent alerts
        current_time = datetime.now()
        last_hour = len([a for a in self.alert_history 
                        if (current_time - a.timestamp).total_seconds() < 3600])
        last_day = len([a for a in self.alert_history 
                       if (current_time - a.timestamp).total_seconds() < 86400])
        
        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_risk_level': risk_counts,
            'alerts_last_hour': last_hour,
            'alerts_last_day': last_day,
            'average_confidence': sum(a.confidence_score for a in self.alert_history) / len(self.alert_history)
        }
    
    def update_config(self, new_config: AlertConfig):
        """Update alert configuration"""
        self.config = new_config
        logger.info("Alert configuration updated")
    
    def add_callback(self, channel: AlertChannel, callback: Callable):
        """Add custom callback for alert channel"""
        self.callbacks[channel] = callback
        logger.info(f"Custom callback added for {channel.value}")

# Example usage and testing
def test_alert_system():
    """Test the alert system"""
    print("Testing Alert System...")
    
    # Create alert configuration
    config = AlertConfig(
        email_enabled=False,  # Disable email for testing
        sms_enabled=False,    # Disable SMS for testing
        webhook_enabled=False, # Disable webhook for testing
        dashboard_enabled=True,
        log_enabled=True,
        min_confidence_threshold=0.5,
        min_amount_threshold=50.0
    )
    
    # Create alert manager
    alert_manager = AlertManager(config)
    
    # Create test alert
    alert = alert_manager.create_alert(
        transaction_id="tx_test_001",
        user_id="user_001",
        amount=1500.0,
        risk_level="high",
        confidence_score=0.85,
        risk_factors=["High transaction amount", "Unusual transaction time", "High velocity"]
    )
    
    print(f"Created alert: {alert.alert_id}")
    
    # Send alert
    results = alert_manager.send_alert(alert)
    print(f"Alert sending results: {results}")
    
    # Get statistics
    stats = alert_manager.get_alert_statistics()
    print(f"Alert statistics: {stats}")
    
    # Get recent alerts
    recent_alerts = alert_manager.get_recent_alerts(5)
    print(f"Recent alerts: {len(recent_alerts)}")

if __name__ == "__main__":
    test_alert_system()
