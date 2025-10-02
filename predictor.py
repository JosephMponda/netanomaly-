import pandas as pd
import numpy as np
import torch
from model import EnhancedLSTMAutoencoder
from preprocessor import ComprehensiveTrafficPreprocessor
import json

class TrafficAnomalyPredictor:
    """Predictor class for traffic anomaly detection"""
    
    def __init__(self, model_path='models/traffic_model.pth',
                 preprocessor_path='models/preprocessor.pkl',
                 config_path='models/config.json'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.config = None
        
        # Load everything
        self._load_model(model_path, preprocessor_path, config_path)
    
    def _load_model(self, model_path, preprocessor_path, config_path):
        """Load model, preprocessor, and configuration"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load preprocessor
        self.preprocessor = ComprehensiveTrafficPreprocessor.load(preprocessor_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = EnhancedLSTMAutoencoder(
            n_features=checkpoint['n_features'],
            seq_len=checkpoint['seq_len'],
            hidden_dim=checkpoint['hidden_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.threshold = checkpoint['threshold']
        
        print(f"Model loaded successfully!")
        print(f"Features: {checkpoint['n_features']}")
        print(f"Sequence length: {checkpoint['seq_len']}")
        print(f"Threshold: {self.threshold:.6f}")
    
    def predict(self, data):
        """
        Predict anomalies in the given data
        
        Args:
            data: pandas DataFrame or path to CSV file
            
        Returns:
            dict with predictions and analysis
        """
        # Load data if path provided
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Preprocess data
        df_processed = self.preprocessor.preprocess_data(data.copy(), verbose=False)
        
        # Create sequences
        sequences = self.preprocessor.create_sequences(df_processed, verbose=False)
        
        if len(sequences) == 0:
            return {
                'error': 'Not enough data to create sequences',
                'min_required': self.preprocessor.sequence_length + 1
            }
        
        # Convert to tensor
        X_tensor = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            
            # Calculate MSE per sequence
            mse_scores = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2))
            mse_scores = mse_scores.cpu().numpy()
            
            # Identify anomalies
            anomalies = mse_scores > self.threshold
            
            # Calculate anomaly scores (normalized)
            anomaly_scores = (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-8)
        
        # Analyze results
        results = {
            'total_sequences': len(sequences),
            'anomalies_detected': int(anomalies.sum()),
            'anomaly_percentage': float(anomalies.sum() / len(anomalies) * 100),
            'threshold': float(self.threshold),
            'mse_scores': mse_scores.tolist(),
            'anomaly_labels': anomalies.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_indices': np.where(anomalies)[0].tolist()
        }
        
        # Add time analysis if time column exists
        if 'time' in df_processed.columns:
            results['time_analysis'] = self._analyze_anomaly_times(
                df_processed, anomalies, self.config['seq_len']
            )
        
        return results
    
    def _analyze_anomaly_times(self, df_processed, anomalies, seq_len):
        """Analyze when anomalies occur"""
        anomaly_indices = np.where(anomalies)[0]
        anomaly_timestamps = []
        
        for idx in anomaly_indices:
            original_idx = seq_len + idx
            if original_idx < len(df_processed):
                timestamp = df_processed.iloc[original_idx]['time']
                anomaly_timestamps.append(timestamp)
        
        if not anomaly_timestamps:
            return None
        
        time_analysis = pd.DataFrame({
            'timestamp': anomaly_timestamps,
            'hour': [pd.to_datetime(ts).hour for ts in anomaly_timestamps],
            'day_of_week': [pd.to_datetime(ts).dayofweek for ts in anomaly_timestamps],
        })
        
        return {
            'timestamps': [str(ts) for ts in anomaly_timestamps],
            'hourly_distribution': time_analysis['hour'].value_counts().to_dict(),
            'daily_distribution': time_analysis['day_of_week'].value_counts().to_dict()
        }
    
    def predict_single_sequence(self, sequence_data):
        """
        Predict if a single sequence is anomalous
        
        Args:
            sequence_data: numpy array of shape (seq_len, n_features)
            
        Returns:
            dict with prediction results
        """
        if sequence_data.shape != (self.config['seq_len'], self.config['n_features']):
            return {
                'error': f"Invalid shape. Expected ({self.config['seq_len']}, {self.config['n_features']})"
            }
        
        # Convert to tensor
        X_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstruction) ** 2).item()
            
            is_anomaly = mse > self.threshold
            anomaly_score = (mse - 0) / (self.threshold * 2)  # Normalized score
        
        return {
            'is_anomaly': bool(is_anomaly),
            'mse_score': float(mse),
            'anomaly_score': float(min(anomaly_score, 1.0)),
            'threshold': float(self.threshold),
            'confidence': float(abs(mse - self.threshold) / self.threshold)
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'config': self.config,
            'threshold': float(self.threshold),
            'device': str(self.device),
            'feature_columns': self.preprocessor.feature_columns
        }


if __name__ == "__main__":
    # Example usage
    print("=== TRAFFIC ANOMALY PREDICTION ===\n")
    
    # Initialize predictor
    predictor = TrafficAnomalyPredictor()
    
    # Get model info
    info = predictor.get_model_info()
    print(f"Model features: {len(info['feature_columns'])}")
    print(f"Threshold: {info['threshold']:.6f}\n")
    
    # Make predictions on test data
    test_csv = r"C:\Users\USER\OneDrive\Desktop\traffic_stats_combined.csv"
    results = predictor.predict(test_csv)
    
    print(f"Total sequences analyzed: {results['total_sequences']}")
    print(f"Anomalies detected: {results['anomalies_detected']}")
    print(f"Anomaly percentage: {results['anomaly_percentage']:.2f}%")
    
    if 'time_analysis' in results and results['time_analysis']:
        print("\nHourly distribution:")
        for hour, count in sorted(results['time_analysis']['hourly_distribution'].items()):
            print(f"  Hour {hour}: {count} anomalies")
