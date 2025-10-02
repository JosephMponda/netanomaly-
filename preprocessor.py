import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

class ComprehensiveTrafficPreprocessor:
    """Preprocessor for traffic data with comprehensive feature handling"""
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def preprocess_data(self, df, verbose=False):
        """Convert ALL potential numeric columns and prepare data"""
        if verbose:
            print("Comprehensive data cleaning...")
        
        df_clean = df.copy()
        
        # Handle time column
        df_clean['time'] = pd.to_datetime(df_clean['time'], errors='coerce')
        df_clean = df_clean.dropna(subset=['time']).sort_values('time')
        
        if verbose:
            print(f"Records after time cleaning: {len(df_clean)}")
        
        # Convert ALL potential numeric columns (except time and pcap_file)
        columns_to_convert = [col for col in df_clean.columns if col not in ['time', 'pcap_file']]
        converted_count = 0
        
        for col in columns_to_convert:
            try:
                original_non_null = df_clean[col].notna().sum()
                converted = pd.to_numeric(df_clean[col], errors='coerce')
                new_non_null = converted.notna().sum()
                
                if new_non_null >= original_non_null * 0.8:
                    df_clean[col] = converted
                    converted_count += 1
                    if verbose:
                        print(f"CONVERTED '{col}' - {new_non_null}/{len(df_clean)} valid values")
            except Exception as e:
                if verbose:
                    print(f"ERROR converting '{col}': {e}")
        
        # Get final numeric columns
        self.feature_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if verbose:
            print(f"Successfully converted {converted_count} columns")
            print(f"Final numeric features ({len(self.feature_columns)}): {self.feature_columns}")
        
        # Handle missing values
        numeric_data = df_clean[self.feature_columns]
        if numeric_data.isnull().sum().sum() > 0:
            if verbose:
                print("Handling missing values...")
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
                df_clean[self.feature_columns] = self.imputer.fit_transform(numeric_data)
            else:
                df_clean[self.feature_columns] = self.imputer.transform(numeric_data)
        
        # Scale features
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            df_clean[self.feature_columns] = self.scaler.fit_transform(df_clean[self.feature_columns])
        else:
            df_clean[self.feature_columns] = self.scaler.transform(df_clean[self.feature_columns])
        
        if verbose:
            print(f"Final data shape: {df_clean.shape}")
        
        return df_clean
    
    def create_sequences(self, data, seq_length=None, verbose=False):
        """Create sequences for training/prediction"""
        if seq_length is None:
            seq_length = self.sequence_length
        
        if len(data) <= seq_length:
            seq_length = max(5, len(data) // 3)
            if verbose:
                print(f"Adjusted sequence length to {seq_length}")
        
        sequences = []
        data_values = data[self.feature_columns].values
        
        for i in range(len(data_values) - seq_length):
            seq = data_values[i:i+seq_length]
            sequences.append(seq)
        
        if verbose:
            print(f"Created {len(sequences)} sequences of length {seq_length}")
        
        return np.array(sequences)
    
    def save(self, filepath):
        """Save preprocessor state"""
        state = {
            'sequence_length': self.sequence_length,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(sequence_length=state['sequence_length'])
        preprocessor.scaler = state['scaler']
        preprocessor.imputer = state['imputer']
        preprocessor.feature_columns = state['feature_columns']
        
        return preprocessor
