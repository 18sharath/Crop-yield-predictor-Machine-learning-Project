import numpy as np
import pickle
from flask import Flask, request, render_template

# Load models
best_model_yield = pickle.load(open('yield_model.pkl', 'rb'))
best_model_price = pickle.load(open('price_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']
        
        features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]], dtype=object)
        transformed_features = preprocessor.transform(features)
        predicted_value = best_model_yield.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=predicted_value)

@app.route('/price', methods=['GET', 'POST'])
def price():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']
        
        features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]], dtype=object)
        transformed_features = preprocessor.transform(features)
        predicted_price = best_model_price.predict(transformed_features).reshape(1, -1)

        return render_template('price.html', price_prediction=predicted_price)
    return render_template('price.html')

if __name__ == "__main__":
    app.run(debug=True)