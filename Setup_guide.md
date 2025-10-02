# Traffic Anomaly Detection - Usage Guide

## 📁 Project Structure

Create this folder structure:

```
traffic_anomaly_detection/
├── models/                    # Will store trained models
├── model.py                   # Core model definition
├── preprocessor.py            # Data preprocessing
├── trainer.py                 # Training script
├── predictor.py               # Prediction module
├── api.py                     # Flask API server
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies

Create a `requirements.txt` file:

```txt
pandas>=1.5.0
numpy>=1.23.0
torch>=2.0.0
scikit-learn>=1.2.0
flask>=2.3.0
flask-cors>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Train Your Model

Run the training script:

```bash
python trainer.py
```

This will:
- Load your CSV data
- Preprocess and clean the data
- Train the LSTM autoencoder
- Save the model to `models/traffic_model.pth`
- Save the preprocessor to `models/preprocessor.pkl`
- Save configuration to `models/config.json`

**Expected Output:**
```
=== TRAFFIC ANOMALY DETECTION MODEL TRAINING ===

Comprehensive data cleaning...
Records after time cleaning: 10000
CONVERTED 'bytes_in' - 9950/10000 valid values
...
Final data shape: (10000, 25)

=== TRAINING MODEL ===
Features: 25
Sequence length: 24
Training samples: 9976
Device: cuda

Epoch [5/30], Loss: 0.002345
Epoch [10/30], Loss: 0.001234
...

Training complete!
Final loss: 0.000567
Anomaly threshold: 0.003421

Model saved to: models/traffic_model.pth
```

### Step 3: Test Predictions (Optional)

Test the predictor directly:

```bash
python predictor.py
```

This will load your model and make predictions on test data.

### Step 4: Start the API Server

```bash
python api.py
```

Or for production with gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

**Expected Output:**
```
=== TRAFFIC ANOMALY DETECTION API ===
Initializing model...
Model loaded successfully!
Features: 25
Sequence length: 24
Threshold: 0.003421
 * Running on http://127.0.0.1:5000
```

## 🔌 API Endpoints

### 1. Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Get Model Info

```bash
curl http://localhost:5000/model/info
```

**Response:**
```json
{
  "config": {
    "n_features": 25,
    "seq_len": 24,
    "threshold": 0.003421
  },
  "feature_columns": ["bytes_in", "bytes_out", ...]
}
```

### 3. Predict from CSV File

```bash
curl -X POST http://localhost:5000/predict/csv \
  -F "file=@traffic_data.csv"
```

**Response:**
```json
{
  "success": true,
  "results": {
    "total_sequences": 500,
    "anomalies_detected": 25,
    "anomaly_percentage": 5.0,
    "threshold": 0.003421,
    "anomaly_indices": [45, 123, 267, ...],
    "time_analysis": {
      "hourly_distribution": {
        "0": 3,
        "1": 5,
        ...
      }
    }
  }
}
```

### 4. Predict from JSON Data

```bash
curl -X POST http://localhost:5000/predict/json \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"time": "2024-01-01 00:00:00", "bytes_in": 1000, "bytes_out": 500},
      {"time": "2024-01-01 00:01:00", "bytes_in": 1200, "bytes_out": 600}
    ]
  }'
```

### 5. Real-time Prediction

```bash
curl -X POST http://localhost:5000/predict/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"time": "2024-01-01 00:00:00", "bytes_in": 1000},
      ...
    ],
    "return_details": false
  }'
```

**Response (simplified):**
```json
{
  "success": true,
  "anomaly_detected": true,
  "anomaly_count": 3,
  "anomaly_percentage": 5.2
}
```

## 💻 Frontend Integration Examples

### JavaScript/React Example

```javascript
// Upload CSV file
async function detectAnomalies(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:5000/predict/csv', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('Anomalies:', result.results.anomalies_detected);
  return result;
}

// Send JSON data
async function detectAnomaliesJSON(data) {
  const response = await fetch('http://localhost:5000/predict/json', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ data })
  });
  
  return await response.json();
}

// Real-time monitoring
async function monitorTraffic(trafficData) {
  const response = await fetch('http://localhost:5000/predict/realtime', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      data: trafficData,
      return_details: false
    })
  });
  
  const result = await response.json();
  
  if (result.anomaly_detected) {
    alert(`⚠️ Anomaly detected! ${result.anomaly_count} anomalies found`);
  }
  
  return result;
}
```

### Python Client Example

```python
import requests
import pandas as pd

# API base URL
API_URL = "http://localhost:5000"

# Upload CSV
def predict_csv(csv_path):
    with open(csv_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict/csv", files=files)
    return response.json()

# Send DataFrame
def predict_dataframe(df):
    data = {
        'data': df.to_dict(orient='records')
    }
    response = requests.post(f"{API_URL}/predict/json", json=data)
    return response.json()

# Real-time monitoring
def monitor_traffic(traffic_data):
    data = {
        'data': traffic_data,
        'return_details': False
    }
    response = requests.post(f"{API_URL}/predict/realtime", json=data)
    return response.json()

# Example usage
result = predict_csv('traffic_data.csv')
print(f"Detected {result['results']['anomalies_detected']} anomalies")
```

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

the API logs for detailed error messages
