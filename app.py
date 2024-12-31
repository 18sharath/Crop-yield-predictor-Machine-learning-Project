import numpy as np
import pickle
from flask import Flask, request, render_template, session
import pandas as pd

# Load models
best_model_yield = pickle.load(open('best_model_yield.pkl', 'rb'))
best_model_price = pickle.load(open('best_model_price.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
preprocessor_price = pickle.load(open('preprocessor_price.pkl', 'rb'))

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            # Create DataFrame for features
            features = pd.DataFrame(
                [[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
                columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
            )
            
            # Transform features and predict yield
            transformed_features = preprocessor.transform(features)
            predicted_value = best_model_yield.predict(transformed_features)
            
            # Convert to Python native type for serialization
            predicted_value = float(predicted_value[0])
            session['predicted_value'] = predicted_value  # Store in session
            
            return render_template('index.html', prediction=predicted_value)
        except Exception as e:
            return f"An error occurred: {e}"

@app.route('/price', methods=['GET', 'POST'])
def price():
    if request.method == 'POST':
        try:
            # Retrieve form data
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']
            crop_yield = float(request.form['crop_yield'])

            # Create DataFrame for features
            features = pd.DataFrame(
                [[Area, Item, Year, crop_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
                columns=['Area', 'Item', 'Year', 'crop_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
            )
            
            # Transform features and predict price
            transformed_features = preprocessor_price.transform(features)
            predicted_price = best_model_price.predict(transformed_features)
            
            # Convert to Python native type for serialization
            predicted_price = float(predicted_price[0])
            
            return render_template('price.html', price_prediction=predicted_price, crop_yield=crop_yield)
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        # Automatically fill crop yield from session
        predicted_value = session.get('predicted_value', None)
        return render_template('price.html', predicted_value=predicted_value, crop_yield=predicted_value)

if __name__ == "__main__":
    app.run(debug=True)
