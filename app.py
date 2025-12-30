from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)



model = None
metadata = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            # Compatibility fix: older pickled sklearn estimators may be missing
            # the `monotonic_cst` attribute when unpickled with newer sklearn.
            # Add the attribute where missing to avoid attribute errors during
            # prediction (see InconsistentVersionWarning / attribute errors).
            try:
                from sklearn.base import BaseEstimator
                from sklearn.tree import DecisionTreeClassifier

                def _fix_monotonic_attr(obj):
                    if isinstance(obj, DecisionTreeClassifier):
                        if not hasattr(obj, 'monotonic_cst'):
                            obj.monotonic_cst = None

                    # Recurse into nested estimators (pipelines, forests, etc.)
                    if isinstance(obj, BaseEstimator):
                        for attr_name in dir(obj):
                            # skip private and callables to reduce noise
                            if attr_name.startswith('__'):
                                continue
                            try:
                                attr = getattr(obj, attr_name)
                            except Exception:
                                continue
                            if isinstance(attr, BaseEstimator):
                                _fix_monotonic_attr(attr)
                            elif isinstance(attr, (list, tuple)):
                                for item in attr:
                                    _fix_monotonic_attr(item)

                _fix_monotonic_attr(model)
            except Exception:
                # If sklearn isn't available or something goes wrong, ignore
                # the compatibility fix and let prediction surface errors.
                pass
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
        
        # Mapping from form field names to dataset column names
        field_mapping = {
            'Age': 'age',
            'Sex': 'sex',
            'ChestPainType': 'cp',
            'RestingBP': 'trestbps',
            'Cholesterol': 'chol',
            'FastingBS': 'fbs',
            'RestingECG': 'restecg',
            'MaxHR': 'thalach',
            'ExerciseAngina': 'exang',
            'Oldpeak': 'oldpeak',
            'ST_Slope': 'slope'
        }
        
        # Default values for columns that might not be in the form
        # These are based on common dataset patterns
        default_values = {
            'ca': 0,  # Number of major vessels colored by flourosopy (default: 0)
            'thal': 2  # Thalassemia (default: 2 = normal)
        }
        
        # Prepare input data
        input_data = {}
        for col in feature_columns:
            value = None
            
            # Method 1: Direct match (exact column name)
            if col in data:
                value = data.get(col)
            # Method 2: Case-insensitive direct match
            elif col.lower() in [k.lower() for k in data.keys()]:
                for key in data.keys():
                    if key.lower() == col.lower():
                        value = data.get(key)
                        break
            # Method 3: Use field mapping (form field to dataset column)
            else:
                for form_field, dataset_col in field_mapping.items():
                    if dataset_col == col:
                        # Try exact match first
                        if form_field in data:
                            value = data.get(form_field)
                            break
                        # Try case-insensitive match
                        for key in data.keys():
                            if key.lower() == form_field.lower():
                                value = data.get(key)
                                break
                        if value is not None:
                            break
            
            # If still no value, use default if available
            if (value is None or value == '') and col in default_values:
                value = default_values[col]
                print(f"Using default value for {col}: {value}")
            
            # If still no value, return error
            if value is None or value == '':
                return jsonify({
                    'error': f'Missing required field: {col}. Received fields: {list(data.keys())}'
                }), 400
            
            # Convert form values to model-expected format
            # Handle specific field conversions
            if col == 'cp':  # ChestPainType
                cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
                if isinstance(value, str) and value in cp_map:
                    value = cp_map[value]
            elif col == 'sex':  # Sex
                sex_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
                if isinstance(value, str) and value in sex_map:
                    value = sex_map[value]
            elif col == 'restecg':  # RestingECG
                ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
                if isinstance(value, str) and value in ecg_map:
                    value = ecg_map[value]
            elif col == 'exang':  # ExerciseAngina
                exang_map = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}
                if isinstance(value, str) and value in exang_map:
                    value = exang_map[value]
            elif col == 'slope':  # ST_Slope
                slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}
                if isinstance(value, str) and value in slope_map:
                    value = slope_map[value]
            
            # Encode categorical variables if needed (for any remaining text values)
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform([str(value)])[0]
                except ValueError:
                    # If value not in encoder, try to convert to numeric first
                    try:
                        input_data[col] = float(value)
                    except ValueError:
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

        # Determine human-readable label (explicit mapping: 0 = Heart Disease)
        decoded_label = None
        target_encoder = metadata.get('target_encoder')
        if target_encoder:
            try:
                decoded_label = target_encoder.inverse_transform([prediction])[0]
            except Exception:
                decoded_label = None

        if decoded_label is None:
            # In this dataset `1` indicates presence of heart disease.
            prediction_label = 'Heart Disease' if int(prediction) == 1 else 'No Heart Disease'
        else:
            prediction_label = str(decoded_label)

        # Normalize predict_proba in case of numeric instability (sum should be 1)
        proba_arr = np.asarray(prediction_proba, dtype=float)
        proba_sum = float(np.sum(proba_arr))
        if proba_sum <= 0:
            normalized = proba_arr
        else:
            normalized = proba_arr / proba_sum

        # Find index corresponding to the heart-disease class (prefer `1` if present)
        cls_idx = 0
        if hasattr(model, 'classes_'):
            try:
                if 1 in list(model.classes_):
                    cls_idx = list(model.classes_).index(1)
                else:
                    cls_idx = list(model.classes_).index(0)
            except ValueError:
                cls_idx = 0

        risk_probability = float(normalized[cls_idx])

        # Return prediction as 1 for Heart Disease, 0 for No Heart Disease (from user perspective)
        user_prediction = 1 if int(prediction) == 1 else 0
        
        return jsonify({
            'prediction': user_prediction,  # 1 = Heart Disease, 0 = No Heart Disease
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

