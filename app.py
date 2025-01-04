# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__, template_folder='Templates')

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

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        try:
            promotion = int(request.form['promotion'])
            
            # Price optimization
            price_range = np.arange(10, 100, 0.5)  # Test prices from $10 to $100
            best_price = 0
            max_revenue = 0
            best_demand = 0
            
            for price in price_range:
                features = np.array([price, promotion]).reshape(1, -1)
                scaled_features = scaler.transform(features)
                predicted_demand = model.predict(scaled_features)[0]
                revenue = predicted_demand * price
                
                if revenue > max_revenue:
                    max_revenue = revenue
                    best_price = price
                    best_demand = predicted_demand
            
            results = {
                'optimal_price': round(best_price, 2),
                'expected_demand': round(best_demand, 2),
                'expected_revenue': round(max_revenue, 2)
            }
            
            return render_template('optimize.html', results=results)
        except Exception as e:
            return render_template('optimize.html', error=str(e))
    
    return render_template('optimize.html')

if __name__ == '__main__':
    app.run(debug=True)
