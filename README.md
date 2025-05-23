# 🤖 Market Pattern Detection System

An AI-powered market pattern detection system using LSTM neural networks with GPU acceleration for real-time extreme market condition alerts.

## 🚀 Features

- **LSTM + Attention Neural Network** for advanced pattern recognition
- **GPU Acceleration** with CUDA support (NVIDIA RTX series)
- **Real-time Monitoring** every 15 minutes
- **Multiple Alert Channels** (Email, Discord, Webhooks)
- **Technical Analysis** with RSI, MACD, Bollinger Bands, ATR
- **Pattern Classification**: Normal, Extreme High, Extreme Low
- **European CSV Format** support (comma decimal separators)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│  LSTM + Attn    │───▶│  Pattern Alerts │
│   (OHLCV + TA)  │    │  Neural Network │    │  (Multi-channel)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (optional but recommended)
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/master0mind/market-pattern-detection.git
cd market-pattern-detection
```

2. **Install dependencies**:
```bash
pip install -r pattern_requirements.txt
```

3. **For GPU support** (NVIDIA users):
```bash
# Verify CUDA installation
nvidia-smi

# PyTorch with CUDA should install automatically
# If not, install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 🎯 Quick Start

### 1. Train the Model
```bash
python train_locally.py
```

### 2. Test Pattern Detection
```bash
python test_pattern_detection.py
```

### 3. Start Real-time Monitoring
```bash
python market_alert_system.py
```

## 📁 Key Files

| File | Description |
|------|-------------|
| `market_pattern_detector.py` | Core LSTM neural network model |
| `train_locally.py` | GPU-accelerated training script |
| `market_alert_system.py` | 15-minute interval monitoring |
| `test_pattern_detection.py` | Pattern analysis and testing |
| `alert_config.json` | Alert configuration settings |
| `market_pattern_model.pth` | Trained model weights |

⚠️ **Disclaimer**: This system is for educational and research purposes. Always conduct your own research before making financial decisions.
