#!/usr/bin/env python3
"""
Quick test to verify GPU training and pattern detection
"""

import pandas as pd
import torch
from market_pattern_detector import MarketPatternDetector

def main():
    print("🔍 QUICK GPU AND PATTERN DETECTION TEST")
    print("=" * 50)
    
    # Check GPU
    print(f"🖥️  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
    
    # Load model
    print("\n📥 Loading trained model...")
    detector = MarketPatternDetector()
    detector.load_model('market_pattern_model.pth')
    print("✅ Model loaded successfully!")
    
    # Load data
    print("\n📊 Loading market data...")
    df = pd.read_csv(r"C:\Users\and\Documents\1012-5M-5D.csv")
    
    # Keep required columns plus optional orderflow columns if available
    required_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
    optional_orderflow_cols = [
        'Delta',
        'Cumulative delta (By volume)_Cumulative open',
        'Cumulative delta (By volume)_Cumulative high',
        'Cumulative delta (By volume)_Cumulative low',
        'Cumulative delta (By volume)_Cumulative close'
    ]

    cols_to_keep = required_cols.copy()
    for col in optional_orderflow_cols:
        if col in df.columns:
            cols_to_keep.append(col)

    df = df[cols_to_keep].copy()
    
    # Parse data
    df = detector.parse_csv_data(df)
    print(f"✅ Loaded {len(df)} data points")
    
    # Test single prediction
    print("\n🤖 Testing single pattern prediction...")
    latest_data = df.tail(50)  # Use last 50 points for prediction
    
    result = detector.predict_pattern(latest_data)
    
    print(f"\n📈 PREDICTION RESULT:")
    print(f"   Time: {df['DateTime'].iloc[-1]}")
    print(f"   Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"   Pattern: {result['prediction'].upper()}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Alert: {'🚨 YES' if result['alert'] else '✅ NO'}")
    
    # Test batch predictions
    print("\n🎯 Testing batch pattern detection...")
    df_features = detector.create_technical_features(df)
    predictions = detector.predict_all_patterns(df_features)
    
    # Count patterns
    normal_count = sum(predictions == 0)
    extreme_high_count = sum(predictions == 1)
    extreme_low_count = sum(predictions == 2)
    
    print(f"📊 BATCH RESULTS:")
    print(f"   Normal: {normal_count} ({normal_count/len(predictions)*100:.1f}%)")
    print(f"   Extreme High: {extreme_high_count} ({extreme_high_count/len(predictions)*100:.1f}%)")
    print(f"   Extreme Low: {extreme_low_count} ({extreme_low_count/len(predictions)*100:.1f}%)")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
