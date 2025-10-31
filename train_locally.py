#!/usr/bin/env python3
"""
Local training script for market pattern detection using NVIDIA GPU
Run this on your local machine with CUDA support
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from market_pattern_detector import MarketPatternDetector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU available: {gpu_name}")
        logger.info(f"Number of GPUs: {gpu_count}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        logger.warning("CUDA GPU not available. Training will use CPU.")
        return False

def load_and_validate_data(csv_path):
    """Load and validate CSV data"""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Keep only the required columns to avoid feature mismatch
    df = df[required_cols].copy()
    logger.info(f"Using columns: {list(df.columns)}")
    
    logger.info("Data validation passed")
    return df

def visualize_training_data(df, labels):
    """Create visualizations of the training data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Validate data and labels alignment
    if len(df) != len(labels):
        logger.warning(f"Data length ({len(df)}) doesn't match labels length ({len(labels)}). Truncating to shorter length.")
        min_len = min(len(df), len(labels))
        df = df.iloc[:min_len]
        labels = labels[:min_len]

    # Price chart with labels
    axes[0, 0].plot(df.index, df['Close'], label='Close Price', alpha=0.7)
    extreme_high = np.where(labels == 1)[0]
    extreme_low = np.where(labels == 2)[0]

    # Filter indices to ensure they're within bounds
    extreme_high = extreme_high[extreme_high < len(df)]
    extreme_low = extreme_low[extreme_low < len(df)]

    if len(extreme_high) > 0:
        axes[0, 0].scatter(extreme_high, df.iloc[extreme_high]['Close'],
                          color='red', s=50, label='Extreme High', alpha=0.8)
    if len(extreme_low) > 0:
        axes[0, 0].scatter(extreme_low, df.iloc[extreme_low]['Close'],
                          color='green', s=50, label='Extreme Low', alpha=0.8)
    
    axes[0, 0].set_title('Price Chart with Extreme Labels')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    
    # Label distribution
    label_counts = np.bincount(labels.astype(int))
    label_names = ['Normal', 'Extreme High', 'Extreme Low']
    axes[0, 1].bar(label_names, label_counts)
    axes[0, 1].set_title('Label Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # Volume analysis
    axes[1, 0].plot(df.index, df['Volume(from bar)'], alpha=0.7)
    axes[1, 0].set_title('Volume Over Time')
    axes[1, 0].set_xlabel('Time Index')
    axes[1, 0].set_ylabel('Volume')
    
    # Price volatility
    df['returns'] = df['Close'].pct_change()
    axes[1, 1].hist(df['returns'].dropna(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Return Distribution')
    axes[1, 1].set_xlabel('Returns')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Training data visualization saved as 'training_data_analysis.png'")

def train_model_local(csv_path, model_save_path='market_pattern_model.pth', 
                     epochs=200, batch_size=64, learning_rate=0.001):
    """
    Train the market pattern detection model locally
    """
    # Check GPU
    use_cuda = check_gpu_availability()
    
    # Initialize detector
    detector = MarketPatternDetector(use_cuda=use_cuda)
    
    # Load data
    logger.info("Loading training data...")
    df = load_and_validate_data(csv_path)
    
    # Parse data
    df = detector.parse_csv_data(df)
    
    # Create features
    logger.info("Creating technical features...")
    df_features = detector.create_technical_features(df)
    
    # Create labels
    logger.info("Creating extreme pattern labels...")
    labels = detector.label_extremes(df_features, lookback=15, threshold=0.3)
    
    # Visualize training data
    visualize_training_data(df, labels)
    
    # Train model
    logger.info("Starting model training...")
    detector.train_model_with_data(df_features, labels, epochs=epochs, 
                                 batch_size=batch_size, lr=learning_rate)
    
    # Save model
    detector.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return detector

def test_model_predictions(model_path, test_csv_path):
    """Test the trained model on new data"""
    logger.info("Testing model predictions...")
    
    # Load model
    detector = MarketPatternDetector(model_path=model_path)
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Get predictions
    result = detector.predict_pattern(test_df)
    
    logger.info("Prediction results:")
    logger.info(f"Prediction: {result['prediction']}")
    logger.info(f"Confidence: {result['confidence']:.3f}")
    logger.info(f"Alert: {result['alert']}")
    
    return result

if __name__ == "__main__":
    import sys
      # Configuration
    CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\and\Documents\1012-5M-5D.csv"
    MODEL_SAVE_PATH = "market_pattern_model.pth"
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    try:
        # Train the model
        logger.info("="*50)
        logger.info("STARTING MARKET PATTERN DETECTION TRAINING")
        logger.info("="*50)
        
        detector = train_model_local(
            csv_path=CSV_FILE,
            model_save_path=MODEL_SAVE_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        # Test the model (optional)
        # test_result = test_model_predictions(MODEL_SAVE_PATH, "test_data.csv")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please make sure your CSV file exists and update CSV_FILE variable")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
