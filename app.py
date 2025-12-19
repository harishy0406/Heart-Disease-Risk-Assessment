from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and metadata
MODEL_PATH = 'models/heart_disease_model.pkl'
METADATA_PATH = 'models/model_metadata.pkl'

model = None
metadata = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
        
        return model is not None and metadata is not None
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
if not load_model():
    print("Warning: Model files not found. Please run the training notebook first.")

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    """Assessment page"""
    return render_template('assessment.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None or metadata is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get form data
        data = request.get_json()
        
        # Get feature columns from metadata
        feature_columns = metadata['feature_columns']
        label_encoders = metadata.get('label_encoders', {})
        
        # Prepare input data
        input_data = {}
        for col in feature_columns:
            value = data.get(col)
            if value is None:
                return jsonify({
                    'error': f'Missing required field: {col}'
                }), 400
            
            # Encode categorical variables if needed
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform([str(value)])[0]
                except ValueError:
                    # If value not in encoder, use the first class
                    input_data[col] = 0
            else:
                # Convert to numeric
                try:
                    input_data[col] = float(value)
                except ValueError:
                    return jsonify({
                        'error': f'Invalid value for {col}: {value}'
                    }), 400
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Decode prediction if target encoder exists
        target_encoder = metadata.get('target_encoder')
        if target_encoder:
            prediction_label = target_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = 'Heart Disease' if prediction == 1 else 'No Heart Disease'
        
        # Get probability
        risk_probability = float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0])
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': str(prediction_label),
            'risk_probability': round(risk_probability * 100, 2),
            'message': f'Risk Assessment: {prediction_label} (Probability: {risk_probability*100:.2f}%)'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 404
    
    return jsonify({
        'accuracy': metadata.get('accuracy', 'N/A'),
        'features': metadata.get('feature_columns', []),
        'feature_importance': metadata.get('feature_importance', [])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

