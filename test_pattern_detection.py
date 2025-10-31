#!/usr/bin/env python3
"""
Test script to demonstrate real-time pattern detection
"""

import pandas as pd
import numpy as np
import torch
from market_pattern_detector import MarketPatternDetector
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def test_pattern_detection(csv_file):
    """Test pattern detection on your market data"""
    
    print("🔍 Testing Market Pattern Detection System")
    print("=" * 50)
    
    # Load the trained model
    detector = MarketPatternDetector()
    
    try:
        detector.load_model('market_pattern_model.pth')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load and process your data
    print(f"\n📊 Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Keep only the required columns to match training data
    required_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
    df = df[required_cols].copy()
    
    # Parse European format data
    df = detector.parse_csv_data(df)
    
    # Show data info
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # Create features and get predictions
    print("\n🔧 Creating technical features...")
    df_features = detector.create_technical_features(df)
    
    print("\n🤖 Running pattern detection...")
    predictions = detector.predict_all_patterns(df_features)
    
    # Analyze results
    pattern_counts = {
        'Normal': np.sum(predictions == 0),
        'Extreme High': np.sum(predictions == 1), 
        'Extreme Low': np.sum(predictions == 2)
    }
    
    print("\n📈 Pattern Detection Results:")
    print("-" * 30)
    for pattern, count in pattern_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"{pattern}: {count} ({percentage:.1f}%)")
    
    # Find extreme patterns with timestamps
    extreme_high_indices = np.where(predictions == 1)[0]
    extreme_low_indices = np.where(predictions == 2)[0]
    
    if len(extreme_high_indices) > 0:
        print(f"\n🔴 EXTREME HIGH patterns detected at:")
        for idx in extreme_high_indices[-5:]:  # Show last 5
            if idx < len(df):
                timestamp = df['DateTime'].iloc[idx]
                price = df['Close'].iloc[idx]
                print(f"  📅 {timestamp} - Price: ${price:.2f}")
    
    if len(extreme_low_indices) > 0:
        print(f"\n🔵 EXTREME LOW patterns detected at:")
        for idx in extreme_low_indices[-5:]:  # Show last 5
            if idx < len(df):
                timestamp = df['DateTime'].iloc[idx]
                price = df['Close'].iloc[idx]
                print(f"  📅 {timestamp} - Price: ${price:.2f}")
    
    # Get the latest prediction
    if len(predictions) > 0:
        latest_prediction = predictions[-1]
        latest_price = df['Close'].iloc[-1]
        latest_time = df['DateTime'].iloc[-1]
        
        pattern_names = {0: 'Normal', 1: 'Extreme High', 2: 'Extreme Low'}
        current_pattern = pattern_names[latest_prediction]
        
        print(f"\n🚨 CURRENT MARKET STATUS:")
        print(f"Time: {latest_time}")
        print(f"Price: ${latest_price:.2f}")
        print(f"Pattern: {current_pattern}")
        
        if latest_prediction in [1, 2]:
            print("⚠️  EXTREME PATTERN DETECTED - POTENTIAL REVERSAL IMMINENT!")
        else:
            print("✅ Normal market conditions")
    
    # Create visualization
    create_pattern_visualization(df, predictions)
    
    return predictions

def create_pattern_visualization(df, predictions):
    """Create a visualization of the pattern detection results"""
    
    plt.figure(figsize=(15, 10))
    
    # Price chart with patterns
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], color='black', linewidth=1, label='Price')
    
    # Color code the patterns
    for i, pred in enumerate(predictions):
        if pred == 1:  # Extreme High
            plt.scatter(i, df['Close'].iloc[i], color='red', s=30, alpha=0.7)
        elif pred == 2:  # Extreme Low
            plt.scatter(i, df['Close'].iloc[i], color='blue', s=30, alpha=0.7)
    
    plt.title('Market Pattern Detection - Price Chart with Extreme Points', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Pattern distribution
    plt.subplot(2, 1, 2)
    pattern_names = ['Normal', 'Extreme High', 'Extreme Low']
    pattern_counts = [np.sum(predictions == i) for i in range(3)]
    colors = ['gray', 'red', 'blue']
    
    bars = plt.bar(pattern_names, pattern_counts, color=colors, alpha=0.7)
    plt.title('Pattern Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, pattern_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pattern_detection_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'pattern_detection_results.png'")

if __name__ == "__main__":
    # Test with your CSV file
    csv_file = r"C:\Users\and\Documents\1012-5M-5D.csv"
    
    try:
        predictions = test_pattern_detection(csv_file)
        print(f"\n✅ Pattern detection test completed successfully!")
        print(f"📝 Check the generated visualization file for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
