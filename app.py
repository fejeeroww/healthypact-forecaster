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
            date = datetime.strptime(request.form['date'], '%Y-%m-%d')
            price = float(request.form['price'])
            promotion = int(request.form['promotion'])
            economic_index = float(request.form['economic_index'])
            competitor_price = float(request.form['competitor_price'])
            
            # Calculate time features
            month = date.month
            is_weekend = 1 if date.weekday() >= 5 else 0
            month_sin = np.sin(2 * np.pi * month/12)
            month_cos = np.cos(2 * np.pi * month/12)
            
            # Create complete feature array
            features = np.array([
                month_sin,
                month_cos,
                is_weekend,
                price,
                promotion,
                economic_index,
                competitor_price,
                1400,  # default demand_lag_7
                1380,  # default demand_lag_14
                1420   # default rolling_mean_7
            ]).reshape(1, -1)
            
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            return render_template('predict.html', prediction=round(prediction, 2))
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        try:
            # Get input values
            date = datetime.strptime(request.form['date'], '%Y-%m-%d')
            promotion = int(request.form['promotion'])
            economic_index = float(request.form['economic_index'])
            competitor_price = float(request.form['competitor_price'])
            
            # Calculate time features
            month = date.month
            is_weekend = 1 if date.weekday() >= 5 else 0
            month_sin = np.sin(2 * np.pi * month/12)
            month_cos = np.cos(2 * np.pi * month/12)
            
            # Price optimization
            price_range = np.arange(10, 100, 0.5)
            best_price = 0
            max_revenue = 0
            best_demand = 0
            
            for price in price_range:
                features = np.array([
                    month_sin,
                    month_cos,
                    is_weekend,
                    price,
                    promotion,
                    economic_index,
                    competitor_price,
                    1400,  # default demand_lag_7
                    1380,  # default demand_lag_14
                    1420   # default rolling_mean_7
                ]).reshape(1, -1)
                
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
