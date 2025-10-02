from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from predictor import TrafficAnomalyPredictor
import io
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize predictor globally
predictor = None

def init_predictor():
    """Initialize the predictor"""
    global predictor
    try:
        predictor = TrafficAnomalyPredictor(
            model_path='models/traffic_model.pth',
            preprocessor_path='models/preprocessor.pkl',
            config_path='models/config.json'
        )
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/csv', methods=['POST'])
def predict_from_csv():
    """
    Predict anomalies from uploaded CSV file
    
    Expected: CSV file in request.files['file']
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read CSV from uploaded file
        df = pd.read_csv(file)
        
        # Make predictions
        results = predictor.predict(df)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/json', methods=['POST'])
def predict_from_json():
    """
    Predict anomalies from JSON data
    
    Expected JSON format:
    {
        "data": [
            {"time": "2024-01-01 00:00:00", "feature1": 0.5, ...},
            ...
        ]
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Make predictions
        results = predictor.predict(df)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/sequence', methods=['POST'])
def predict_single_sequence():
    """
    Predict if a single sequence is anomalous
    
    Expected JSON format:
    {
        "sequence": [[feature1, feature2, ...], [...], ...]  # Shape: (seq_len, n_features)
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400
        
        sequence = np.array(data['sequence'])
        
        # Make prediction
        result = predictor.predict_single_sequence(sequence)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict anomalies for multiple sequences
    
    Expected JSON format:
    {
        "sequences": [
            [[feature1, feature2, ...], [...], ...],  # Sequence 1
            [[feature1, feature2, ...], [...], ...],  # Sequence 2
            ...
        ]
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({'error': 'No sequences provided'}), 400
        
        results = []
        for i, sequence in enumerate(data['sequences']):
            sequence_array = np.array(sequence)
            result = predictor.predict_single_sequence(sequence_array)
            result['sequence_id'] = i
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_sequences': len(results),
            'anomalies_detected': sum(1 for r in results if r.get('is_anomaly', False))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/realtime', methods=['POST'])
def predict_realtime():
    """
    Real-time prediction endpoint for streaming data
    
    Expected JSON format:
    {
        "data": [
            {"time": "2024-01-01 00:00:00", "feature1": 0.5, ...},
            ...
        ],
        "return_details": true  # Optional, default false
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        df = pd.DataFrame(data['data'])
        return_details = data.get('return_details', False)
        
        # Make predictions
        results = predictor.predict(df)
        
        # Simplified response for real-time
        response = {
            'success': True,
            'anomaly_detected': results['anomalies_detected'] > 0,
            'anomaly_count': results['anomalies_detected'],
            'anomaly_percentage': results['anomaly_percentage']
        }
        
        if return_details:
            response['details'] = results
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=== TRAFFIC ANOMALY DETECTION API ===")
    print("Initializing model...")
