import torch
import torch.nn as nn

class EnhancedLSTMAutoencoder(nn.Module):
    """Enhanced LSTM Autoencoder for Traffic Anomaly Detection"""
    
    def __init__(self, n_features, seq_len, hidden_dim=64):
        super(EnhancedLSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder_lstm(x)
        
        # Use the last hidden state and repeat for sequence length
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input)
        
        return decoded
