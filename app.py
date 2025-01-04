from flask import Flask, request, render_template
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Load models from models directory
model = joblib.load('demand_forecaster_model.joblib')
scaler = joblib.load('feature_scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        price = float(request.form['price'])
        promotion = int(request.form['promotion'])
        economic_index = float(request.form['economic_index'])
        competitor_price = float(request.form['competitor_price'])
        
        # Feature engineering
        month = date.month
        is_weekend = 1 if date.weekday() >= 5 else 0
        month_sin = np.sin(2 * np.pi * month/12)
        month_cos = np.cos(2 * np.pi * month/12)
        
        # Create features array
        features = np.array([
            month_sin, month_cos, is_weekend,
            price, promotion, economic_index,
            competitor_price, 1400, 1380, 1420
        ]).reshape(1, -1)
        
        # Make prediction
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        return render_template('predict.html', prediction=round(prediction, 2))
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
