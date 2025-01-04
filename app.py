# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)


# Load models from models directory
model_path = os.path.join('models', 'demand_forecaster_model.joblib')
scaler_path = os.path.join('models', 'feature_scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values
            price = float(request.form['price'])
            promotion = int(request.form['promotion'])
            
            # Create features and scale them
            features = np.array([price, promotion]).reshape(1, -1)
            scaled_features = scaler.transform(features)
            
            # Make prediction with scaled features
            prediction = model.predict(scaled_features)[0]
            
            return render_template('predict.html', prediction=round(prediction, 2))
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
