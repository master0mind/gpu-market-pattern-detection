# Market Pattern Detection and Alert System
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketPatternLSTM(nn.Module):
    """
    LSTM Neural Network for detecting market extreme patterns
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(MarketPatternLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # 3 classes: 0=Normal, 1=Extreme_High, 2=Extreme_Low
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context_vector)
        return output, attention_weights

class MarketPatternDetector:
    """
    Main class for detecting market extreme patterns
    """
    def __init__(self, model_path=None, use_cuda=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.sequence_length = 20  # Look at last 20 periods (5 hours of 15-min data)
        self.feature_columns = []
        
        if model_path:
            self.load_model(model_path)
            
        logger.info(f"Using device: {self.device}")
    
    def create_technical_features(self, df):
        """
        Create technical analysis features for pattern detection
        """
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume(from bar)']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Basic price features
        data['price_range'] = data['High'] - data['Low']
        data['upper_shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
        data['lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        data['body_size'] = np.abs(data['Close'] - data['Open'])
        data['body_ratio'] = data['body_size'] / (data['price_range'] + 1e-8)
        
        # Moving averages
        for period in [5, 10, 20]:
            data[f'sma_{period}'] = ta.trend.SMAIndicator(data['Close'], window=period).sma_indicator()
            data[f'ema_{period}'] = ta.trend.EMAIndicator(data['Close'], window=period).ema_indicator()
        
        # RSI (Relative Strength Index)
        data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'], window=20)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # Volume indicators
        data['volume_sma'] = data['Volume(from bar)'].rolling(window=10).mean()
        data['volume_ratio'] = data['Volume(from bar)'] / (data['volume_sma'] + 1e-8)
        
        # Volatility indicators
        data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        
        # Price momentum
        for period in [3, 5, 10]:
            data[f'momentum_{period}'] = data['Close'].pct_change(period)
        
        # Support/Resistance levels
        data['local_high'] = data['High'].rolling(window=5, center=True).max() == data['High']
        data['local_low'] = data['Low'].rolling(window=5, center=True).min() == data['Low']
        
        # Market structure
        data['higher_high'] = (data['High'] > data['High'].shift(1)) & (data['High'].shift(1) > data['High'].shift(2))
        data['lower_low'] = (data['Low'] < data['Low'].shift(1)) & (data['Low'].shift(1) < data['Low'].shift(2))
        
        return data
    
    def label_extremes(self, df, lookback=10, threshold=0.5):
        """
        Label extreme market conditions for training
        
        Args:
            df: DataFrame with market data
            lookback: Number of periods to look ahead/behind for extremes
            threshold: Percentage threshold for extreme moves
        
        Returns:
            labels: 0=Normal, 1=Extreme_High, 2=Extreme_Low
        """
        labels = np.zeros(len(df))
        
        for i in range(lookback, len(df) - lookback):
            current_close = df.iloc[i]['Close']

            # Skip if current_close is zero or too small (avoid division by zero)
            if abs(current_close) < 1e-8:
                continue

            # Look at future prices to identify reversals
            future_prices = df.iloc[i+1:i+lookback+1]['Close']
            past_prices = df.iloc[i-lookback:i]['Close']

            # Calculate percentage moves
            max_future_move = (future_prices.max() - current_close) / current_close * 100
            min_future_move = (future_prices.min() - current_close) / current_close * 100
            max_past_move = (past_prices.max() - current_close) / current_close * 100
            min_past_move = (past_prices.min() - current_close) / current_close * 100
            
            # Check for extreme conditions leading to reversals
            if (max_past_move > threshold and min_future_move < -threshold):
                labels[i] = 1  # Extreme high (will reverse down)
            elif (min_past_move < -threshold and max_future_move > threshold):
                labels[i] = 2  # Extreme low (will reverse up)
        
        return labels
    
    def prepare_sequences(self, data, labels=None):
        """
        Prepare sequences for LSTM training/prediction

        Note: Sequences start from index sequence_length. This means:
        - sequence[0] uses data[0:sequence_length] and targets label[sequence_length]
        - The first sequence_length labels (0 to sequence_length-1) are not used
        - This is correct because we need historical data to predict the current state
        """
        # Select feature columns (exclude non-numeric and target columns)
        exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]

        # Handle missing values
        data_clean = data[feature_cols].ffill().fillna(0)

        # Store feature columns for later use
        self.feature_columns = feature_cols

        # Scale features
        if not hasattr(self.scaler, 'scale_'):
            data_scaled = self.scaler.fit_transform(data_clean)
        else:
            data_scaled = self.scaler.transform(data_clean)

        # Validate labels alignment if provided
        if labels is not None and len(labels) != len(data_scaled):
            raise ValueError(f"Labels length ({len(labels)}) must match data length ({len(data_scaled)})")

        # Create sequences
        sequences = []
        targets = []

        for i in range(self.sequence_length, len(data_scaled)):
            sequences.append(data_scaled[i-self.sequence_length:i])
            if labels is not None:
                targets.append(labels[i])
        
        sequences = np.array(sequences)
        if labels is not None:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences
    
    def train_model(self, data_path, epochs=100, batch_size=32, lr=0.001):
        """
        Train the pattern detection model
        """
        # Load and prepare data
        df = pd.read_csv(data_path)
        df = self.parse_csv_data(df)
        
        # Create features
        df_features = self.create_technical_features(df)
        
        # Create labels
        labels = self.label_extremes(df_features)
        
        # Use the detailed training method
        return self.train_model_with_data(df_features, labels, epochs, batch_size, lr)
    
    def train_model_with_data(self, df_features, labels, epochs=100, batch_size=32, lr=0.001):
        """
        Train the pattern detection model with prepared data
        """
        # Prepare sequences
        X, y = self.prepare_sequences(df_features, labels)

        # Check if we have enough samples for stratification
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()

        # Split data - only use stratify if we have enough samples of each class
        if min_samples >= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            logger.warning(f"Insufficient samples for stratification (min class has {min_samples} samples). Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = MarketPatternLSTM(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_acc = 0.0
        train_losses = []
        test_accs = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs, _ = self.model(X_test)
                    test_loss = criterion(test_outputs, y_test)
                    test_acc = (test_outputs.argmax(1) == y_test).float().mean()
                
                scheduler.step(test_loss)
                
                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model('best_market_pattern_model.pth')
                
                train_losses.append(epoch_loss / (len(X_train) // batch_size))
                test_accs.append(test_acc.item())
                
                logger.info(f'Epoch {epoch}: Train Loss: {epoch_loss/(len(X_train)//batch_size):.4f}, '
                           f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Best Acc: {best_acc:.4f}')
                self.model.train()
        
        # Save final model
        self.save_model('market_pattern_model.pth')
        logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
        return train_losses, test_accs

    def parse_csv_data(self, df):
        """
        Parse CSV data with European format handling
        """
        # Handle European decimal format
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
        
        # Parse datetime
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        return df
    
    def predict_pattern(self, data):
        """
        Predict market extreme patterns for new data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Parse and prepare data
        if isinstance(data, str):
            # If CSV string, parse it
            import io
            df = pd.read_csv(io.StringIO(data))
        else:
            df = data.copy()
        
        df = self.parse_csv_data(df)
        df_features = self.create_technical_features(df)
        
        # Prepare sequences
        X = self.prepare_sequences(df_features)
        
        if len(X) == 0:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0,
                'message': f'Need at least {self.sequence_length} data points for prediction'
            }
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(1)
        
        # Get latest prediction
        latest_prediction = predictions[-1].cpu().item()
        latest_probabilities = probabilities[-1].cpu().numpy()
        latest_attention = attention_weights[-1].cpu().numpy()
        
        # Map predictions to readable format
        prediction_map = {0: 'normal', 1: 'extreme_high', 2: 'extreme_low'}
        
        return {
            'prediction': prediction_map[latest_prediction],
            'confidence': float(latest_probabilities.max()),
            'probabilities': {
                'normal': float(latest_probabilities[0]),
                'extreme_high': float(latest_probabilities[1]),
                'extreme_low': float(latest_probabilities[2])
            },
            'attention_weights': latest_attention.tolist(),
            'timestamp': datetime.now().isoformat(),
            'alert': latest_prediction != 0 and latest_probabilities.max() > 0.7
        }
    
    def predict_all_patterns(self, data):
        """
        Predict market extreme patterns for all time points in the data
        Returns numpy array of predictions for visualization
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Parse and prepare data
        if isinstance(data, str):
            import io
            df = pd.read_csv(io.StringIO(data))
        else:
            df = data.copy()
        
        df = self.parse_csv_data(df)
        df_features = self.create_technical_features(df)
        
        # Prepare sequences
        X = self.prepare_sequences(df_features)
        
        if len(X) == 0:
            return np.array([])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            predictions = outputs.argmax(1)
        
        return predictions.cpu().numpy()

    def save_model(self, path):
        """Save model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }, path)
    
    def load_model(self, path):
        """Load model and scaler"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Initialize model with correct input size
        input_size = len(checkpoint['feature_columns'])
        self.model = MarketPatternLSTM(input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        self.sequence_length = checkpoint['sequence_length']

# Cloud Function entry point
def market_pattern_analysis(request):
    """
    Cloud Function for market pattern detection
    """
    try:
        detector = MarketPatternDetector(model_path='market_pattern_model.pth', use_cuda=False)
        
        # Get data from request
        if request.files and 'file' in request.files:
            file = request.files['file']
            content = file.read().decode('utf-8')
            result = detector.predict_pattern(content)
        else:
            request_json = request.get_json(silent=True)
            if not request_json or 'csv_data' not in request_json:
                return json.dumps({
                    'error': 'No CSV data provided',
                    'status': 'error'
                }), 400
            
            result = detector.predict_pattern(request_json['csv_data'])
        
        return json.dumps({
            'results': result,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in market pattern analysis: {str(e)}")
        return json.dumps({
            'error': f'Error processing request: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == "__main__":
    # Example usage for local training
    detector = MarketPatternDetector()
    # detector.train_model('your_market_data.csv')
