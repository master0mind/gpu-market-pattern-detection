#!/usr/bin/env python3
"""
Real-time pattern detection demo - monitors your market data
Run this to see the system in action with live monitoring
"""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
from market_pattern_detector import MarketPatternDetector
import os

def simulate_real_time_monitoring(csv_file, update_interval=5):
    """
    Simulate real-time monitoring by processing your CSV data in chunks
    """
    print("🚨 STARTING REAL-TIME MARKET PATTERN MONITORING")
    print("=" * 60)
    
    # Load trained model
    detector = MarketPatternDetector()
    detector.load_model('market_pattern_model.pth')
    print("✅ Pattern detection model loaded")

    # Load your market data
    df = pd.read_csv(csv_file)
    
    # Keep only the required columns to match training data
    required_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
    df = df[required_cols].copy()
    
    df = detector.parse_csv_data(df)
    print(f"📊 Loaded {len(df)} data points from {csv_file}")
    
    # Configuration
    min_data_points = 50  # Need enough data for pattern detection
    alert_threshold = 0.8  # Confidence threshold for alerts
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Update interval: {update_interval} seconds")
    print(f"   • Minimum data points: {min_data_points}")
    print(f"   • Alert threshold: {alert_threshold}")
    print(f"   • Monitoring: {len(df)} total data points")
    
    print(f"\n🔄 Starting monitoring... (Press Ctrl+C to stop)")
    print("-" * 60)
    
    try:
        # Simulate real-time by processing data chunks
        for i in range(min_data_points, len(df), 5):  # Process 5 new points at a time
            current_data = df.iloc[:i]
            current_time = current_data['DateTime'].iloc[-1]
            current_price = current_data['Close'].iloc[-1]
            
            try:
                # Analyze current market state
                result = detector.predict_pattern(current_data)
                
                # Display current status
                status_emoji = "🟢" if result['prediction'] == 'normal' else "🔴" if result['prediction'] == 'extreme_high' else "🔵"
                
                print(f"{status_emoji} {current_time} | Price: ${current_price:.2f} | "
                      f"Pattern: {result['prediction'].upper()} | "
                      f"Confidence: {result['confidence']:.3f}")
                
                # Check for alerts
                if result['prediction'] != 'normal' and result['confidence'] >= alert_threshold:
                    print("🚨" * 20)
                    print(f"⚠️  EXTREME PATTERN ALERT!")
                    print(f"   Time: {current_time}")
                    print(f"   Price: ${current_price:.2f}")
                    print(f"   Pattern: {result['prediction'].upper()}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    print(f"   🔮 POTENTIAL MARKET REVERSAL INCOMING!")
                    print("🚨" * 20)
                    
                    # Save alert to file
                    alert_data = {
                        'timestamp': current_time,
                        'price': float(current_price),
                        'pattern': result['prediction'],
                        'confidence': result['confidence'],
                        'alert_time': datetime.now().isoformat()
                    }
                    
                    with open('pattern_alerts.json', 'a') as f:
                        f.write(json.dumps(alert_data) + '\n')
                
            except Exception as e:
                print(f"⚠️  Error analyzing data at {current_time}: {e}")
            
            # Wait before next update
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped by user")
        print("📝 Check 'pattern_alerts.json' for any alerts generated")

def setup_real_time_alerts():
    """
    Setup guide for real-time alerts
    """
    print("📋 REAL-TIME ALERT SETUP GUIDE")
    print("=" * 50)
    
    print("\n1️⃣  EMAIL ALERTS")
    print("   Update alert_config.json with your email settings:")
    print("   {")
    print('     "email": {')
    print('       "smtp_server": "smtp.gmail.com",')
    print('       "smtp_port": 587,')
    print('       "username": "your-email@gmail.com",')
    print('       "password": "your-app-password",')
    print('       "to_email": "alerts@yourdomain.com"')
    print('     }')
    print("   }")
    
    print("\n2️⃣  DISCORD ALERTS")
    print("   Add Discord webhook URL to alert_config.json:")
    print("   {")
    print('     "discord": {')
    print('       "webhook_url": "https://discord.com/api/webhooks/..."')
    print('     }')
    print("   }")
    
    print("\n3️⃣  LIVE DATA FEED")
    print("   For real-time monitoring, integrate with:")
    print("   • Binance API for crypto")
    print("   • Alpha Vantage for stocks")
    print("   • Your broker's API")
    print("   • Custom data feed")
    
    print("\n4️⃣  DEPLOYMENT OPTIONS")
    print("   • Local: Run on your Windows machine")
    print("   • Cloud: Deploy to Google Cloud Functions")
    print("   • VPS: Run on a virtual private server")
    print("   • Raspberry Pi: Run on edge device")

if __name__ == "__main__":
    csv_file = r"C:\Users\and\Documents\1012-5M-5D.csv"
    
    print("🎯 MARKET PATTERN DETECTION SYSTEM")
    print("=" * 50)
    print("Choose an option:")
    print("1. 📊 Demo real-time monitoring (using your CSV data)")
    print("2. ⚙️  Setup guide for real alerts")
    print("3. 🔍 Single pattern analysis")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 50)
        simulate_real_time_monitoring(csv_file, update_interval=2)
    
    elif choice == "2":
        print("\n" + "=" * 50)
        setup_real_time_alerts()
    
    elif choice == "3":
        print("\n" + "=" * 50)
        detector = MarketPatternDetector()
        detector.load_model('market_pattern_model.pth')
        
        df = pd.read_csv(csv_file)
        result = detector.predict_pattern(df)
        
        print(f"🔮 Current Market Analysis:")
        print(f"   Pattern: {result['prediction'].upper()}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Alert: {'YES' if result['alert'] else 'NO'}")
        
        if result['prediction'] != 'normal':
            print(f"\n⚠️  EXTREME PATTERN DETECTED!")
            print(f"🔮 Market may be at a turning point!")
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
