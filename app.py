import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
# Make sure these files ('diabetes_model.pkl', 'scaler.pkl') exist in the 'model' directory
try:
    model = joblib.load('model/diabetes_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    # Assuming accuracy and other metrics are known post-training
    # For this example, we'll hardcode the accuracy derived from the notebook
    # In a real-world scenario, these might be loaded from a config file or calculated dynamically
    model_accuracy = 0.7532 # From the notebook's evaluation
    
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Please ensure 'model/diabetes_model.pkl' and 'model/scaler.pkl' exist.")
    model = None
    scaler = None
    model_accuracy = None # Set to None if model fails to load

# Define the features that the model expects
# This order must match the order of features used during training
FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# --- Placeholder for future functionalities ---
# You can add more routes, data preprocessing, or other features here.
# For example, a route to display model metrics, or an API for retraining.

@app.route('/metrics')
def get_metrics():
    if model_accuracy is not None:
        return jsonify({'accuracy': model_accuracy, 'model_status': 'loaded'})
    else:
        return jsonify({'error': 'Model metrics not available, model not loaded.'}), 500

# ----------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded. Cannot make predictions.'}), 500

    try:
        data = request.get_json(force=True)
        
        # Convert input data to a Pandas DataFrame
        # Ensure the order of columns matches FEATURE_COLUMNS
        input_df = pd.DataFrame([data])
        
        # Handle missing values if necessary (e.g., fill with median)
        # For simplicity, we assume incoming data handles this, 
        # or the client sends complete data. If not, add imputation here.
        # Example: for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']: 
        #            if col in input_df.columns and input_df[col].iloc[0] == 0: 
        #                input_df[col] = scaler.mean_[FEATURE_COLUMNS.index(col)] # or a specific median

        # Ensure all required features are present and in the correct order
        input_data = input_df[FEATURE_COLUMNS]

        # Scale the input features
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        result = {
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# You can add a simple home route for testing if needed
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # For local development, use app.run(debug=True)
    # For deployment with Gunicorn, Gunicorn will handle running the app
    app.run(host='0.0.0.0', port=5000, debug=True)
