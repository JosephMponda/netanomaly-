import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import EnhancedLSTMAutoencoder
from preprocessor import ComprehensiveTrafficPreprocessor
import json

class TrafficAnomalyTrainer:
    """Trainer class for traffic anomaly detection model"""
    
    def __init__(self, sequence_length=24, hidden_dim=64, learning_rate=1e-3):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.preprocessor = ComprehensiveTrafficPreprocessor(sequence_length=sequence_length)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = None
        self.training_stats = {}
        
    def prepare_data(self, csv_path, verbose=True):
        """Load and prepare data for training"""
        if verbose:
            print(f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df_processed = self.preprocessor.preprocess_data(df, verbose=verbose)
        sequences = self.preprocessor.create_sequences(df_processed, verbose=verbose)
        
        if verbose:
            print(f"Sequence shape: {sequences.shape}")
        
        return sequences, df_processed
    
    def train(self, sequences, epochs=30, batch_size=32, verbose=True):
        """Train the model"""
        # Prepare data
        X_tensor = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        n_features = sequences.shape[2]
        seq_len = sequences.shape[1]
        
        self.model = EnhancedLSTMAutoencoder(
            n_features=n_features,
            seq_len=seq_len,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if verbose:
            print(f"\n=== TRAINING MODEL ===")
            print(f"Features: {n_features}")
            print(f"Sequence length: {seq_len}")
            print(f"Training samples: {len(sequences)}")
            print(f"Device: {self.device}")
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        # Calculate threshold
        self.threshold = self._calculate_threshold(dataloader, percentile=95)
        
        # Store training stats
        self.training_stats = {
            'n_features': n_features,
            'seq_len': seq_len,
            'hidden_dim': self.hidden_dim,
            'epochs': epochs,
            'final_loss': losses[-1],
            'threshold': float(self.threshold),
            'training_samples': len(sequences)
        }
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Anomaly threshold: {self.threshold:.6f}")
        
        return losses
    
    def _calculate_threshold(self, dataloader, percentile=95):
        """Calculate anomaly detection threshold"""
        self.model.eval()
        mse_scores = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                mse = torch.mean((batch_X - outputs) ** 2, dim=(1, 2))
                mse_scores.extend(mse.cpu().numpy())
        
        threshold = np.percentile(mse_scores, percentile)
        return threshold
    
    def save_model(self, model_path='models/traffic_model.pth', 
                   preprocessor_path='models/preprocessor.pkl',
                   config_path='models/config.json'):
        """Save model, preprocessor, and configuration"""
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_features': self.training_stats['n_features'],
            'seq_len': self.training_stats['seq_len'],
            'hidden_dim': self.hidden_dim,
            'threshold': self.threshold
        }, model_path)
        
        # Save preprocessor
        self.preprocessor.save(preprocessor_path)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")
        print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    # Training script
    print("=== TRAFFIC ANOMALY DETECTION MODEL TRAINING ===\n")
    
    # Initialize trainer
    trainer = TrafficAnomalyTrainer(
        sequence_length=24,
        hidden_dim=64,
        learning_rate=1e-3
    )
    
    # Load and prepare data
    csv_path = r"C:\Users\USER\OneDrive\Desktop\traffic_stats_combined.csv"
    sequences, df_processed = trainer.prepare_data(csv_path, verbose=True)
    
    # Train model
    losses = trainer.train(sequences, epochs=30, batch_size=32, verbose=True)
    
    # Save model
    trainer.save_model()
    
    print("\n=== TRAINING COMPLETE ===")
