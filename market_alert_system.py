#!/usr/bin/env python3
"""
Real-time market pattern monitoring and alert system
Checks for extreme patterns every 15 minutes and sends alerts
"""

import time
import json
import requests
import logging
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import os
from pathlib import Path
from market_pattern_detector import MarketPatternDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketAlertSystem:
    """
    Real-time market monitoring and alert system
    """
    
    def __init__(self, config_path='alert_config.json'):
        self.config = self.load_config(config_path)
        self.detector = MarketPatternDetector(
            model_path=self.config.get('model_path', 'market_pattern_model.pth'),
            use_cuda=False  # Cloud deployment typically doesn't have GPU
        )
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(hours=1)  # Prevent spam alerts
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "data_source": {
                    "type": "csv_file",  # or "api"
                    "path": "current_market_data.csv",
                    "api_url": "your_data_api_endpoint"
                },
                "alerts": {
                    "email": {
                        "enabled": True,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "your_email@gmail.com",
                        "password": "your_app_password",
                        "recipients": ["your_email@gmail.com"]
                    },
                    "webhook": {
                        "enabled": False,
                        "url": "https://your-webhook-url.com/alert"
                    },
                    "discord": {
                        "enabled": False,
                        "webhook_url": "your_discord_webhook_url"
                    }
                },
                "monitoring": {
                    "confidence_threshold": 0.7,
                    "check_interval_minutes": 15,
                    "alert_cooldown_hours": 1
                },
                "model_path": "market_pattern_model.pth"
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default config at {config_path}. Please update it with your settings.")
            return default_config
    
    def get_latest_data(self):
        """Get the latest market data"""
        data_source = self.config['data_source']
        
        if data_source['type'] == 'csv_file':
            # Read from CSV file
            csv_path = data_source['path']
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")

            df = pd.read_csv(csv_path)
            # Get sufficient historical data for accurate technical indicators
            # Need at least 50 rows: 20 for sequence_length + 20 for indicators + buffer
            min_rows = 50
            if len(df) < min_rows:
                logger.warning(f"CSV has only {len(df)} rows, need at least {min_rows} for reliable predictions")
                return df  # Return all available data
            return df.tail(min_rows)
            
        elif data_source['type'] == 'api':
            # Fetch from API
            response = requests.get(data_source['api_url'])
            if response.status_code == 200:
                data = response.json()
                # Convert API response to DataFrame
                # This will depend on your API format
                return pd.DataFrame(data)
            else:
                raise Exception(f"API request failed: {response.status_code}")
        
        else:
            raise ValueError(f"Unknown data source type: {data_source['type']}")
    
    def check_for_patterns(self):
        """Check for extreme market patterns"""
        try:
            # Get latest data
            latest_data = self.get_latest_data()
            
            # Predict patterns
            result = self.detector.predict_pattern(latest_data)
            
            # Log the prediction
            logger.info(f"Pattern check: {result['prediction']} (confidence: {result['confidence']:.3f})")
            
            # Check if alert should be sent
            if result['alert'] and result['confidence'] >= self.config['monitoring']['confidence_threshold']:
                self.send_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during pattern check: {e}")
            return None
    
    def send_alert(self, prediction_result):
        """Send alert through configured channels"""
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        # Check cooldown
        now = datetime.now()
        if prediction in self.last_alert_time:
            time_since_last = now - self.last_alert_time[prediction]
            if time_since_last < self.alert_cooldown:
                logger.info(f"Alert cooldown active for {prediction}. Skipping alert.")
                return
        
        # Update last alert time
        self.last_alert_time[prediction] = now
        
        # Create alert message
        alert_message = self.create_alert_message(prediction_result)
        
        # Send through configured channels
        alerts_config = self.config['alerts']
        
        if alerts_config['email']['enabled']:
            self.send_email_alert(alert_message, prediction_result)
        
        if alerts_config['webhook']['enabled']:
            self.send_webhook_alert(alert_message, prediction_result)
        
        if alerts_config['discord']['enabled']:
            self.send_discord_alert(alert_message, prediction_result)
        
        logger.info(f"Alert sent for {prediction} pattern (confidence: {confidence:.3f})")
    
    def create_alert_message(self, result):
        """Create formatted alert message"""
        prediction = result['prediction']
        confidence = result['confidence']
        timestamp = result['timestamp']
        probabilities = result['probabilities']
        
        if prediction == 'extreme_high':
            message = f"🔴 MARKET EXTREME HIGH DETECTED!\n"
            message += f"The market may be at a turning point - potential reversal downward.\n"
        elif prediction == 'extreme_low':
            message = f"🟢 MARKET EXTREME LOW DETECTED!\n"
            message += f"The market may be at a turning point - potential reversal upward.\n"
        else:
            message = f"ℹ️ Market Pattern Alert\n"
        
        message += f"\n📊 Details:\n"
        message += f"• Prediction: {prediction.replace('_', ' ').title()}\n"
        message += f"• Confidence: {confidence:.1%}\n"
        message += f"• Time: {timestamp}\n"
        message += f"\n📈 Probabilities:\n"
        message += f"• Normal: {probabilities['normal']:.1%}\n"
        message += f"• Extreme High: {probabilities['extreme_high']:.1%}\n"
        message += f"• Extreme Low: {probabilities['extreme_low']:.1%}\n"
        
        return message
    
    def send_email_alert(self, message, result):
        """Send email alert"""
        try:
            email_config = self.config['alerts']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['Subject'] = f"Market Pattern Alert: {result['prediction'].replace('_', ' ').title()}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            for recipient in email_config['recipients']:
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']
            
            server.quit()
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, message, result):
        """Send webhook alert"""
        try:
            webhook_url = self.config['alerts']['webhook']['url']
            
            payload = {
                'message': message,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'timestamp': result['timestamp'],
                'probabilities': result['probabilities']
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def send_discord_alert(self, message, result):
        """Send Discord alert"""
        try:
            webhook_url = self.config['alerts']['discord']['webhook_url']
            
            # Discord webhook payload
            payload = {
                'content': message,
                'embeds': [{
                    'title': f"Market Pattern: {result['prediction'].replace('_', ' ').title()}",
                    'color': 0xff0000 if result['prediction'] == 'extreme_high' else 0x00ff00,
                    'fields': [
                        {'name': 'Confidence', 'value': f"{result['confidence']:.1%}", 'inline': True},
                        {'name': 'Time', 'value': result['timestamp'], 'inline': True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Discord alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting market pattern monitoring system...")
        
        # Schedule pattern checks
        interval = self.config['monitoring']['check_interval_minutes']
        schedule.every(interval).minutes.do(self.check_for_patterns)
        
        # Run an initial check
        logger.info("Running initial pattern check...")
        self.check_for_patterns()
        
        # Main monitoring loop
        logger.info(f"Monitoring started. Checking every {interval} minutes...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled tasks

def main():
    """Main function to start the alert system"""
    try:
        alert_system = MarketAlertSystem()
        alert_system.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Alert system error: {e}")
        raise

if __name__ == "__main__":
    main()
