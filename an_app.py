from flask import Flask, request, render_template, send_file, session
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns


# Load models
best_model_yield = pickle.load(open('best_model_yield.pkl', 'rb'))
best_model_price = pickle.load(open('best_model_price.pkl', 'rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))
preprocessor_price = pickle.load(open('preprocessor_price.pkl', 'rb'))

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('up_index.html')

@app.route("/predict", methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]], dtype=object)
        features = pd.DataFrame([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]],
                                columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
                                         'avg_temp'])
        transformed_features = preprocessor.transform(features)
        predicted_value = best_model_yield.predict(transformed_features).reshape(1, -1)
        session['predicted_value'] = predicted_value
        return render_template('up_index.html', prediction=predicted_value)

@app.route("/bulk_predict", methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read the CSV file
    try:
        data = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return "Uploaded file is empty. Please upload a valid CSV file.", 400

     # Check if the DataFrame is empty
    if data.empty:
        return "Uploaded file has no data. Please upload a populated CSV file.", 400

    # Ensure required columns are present
    required_columns = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    if not all(col in data.columns for col in required_columns):
        return f"Missing required columns: {set(required_columns) - set(data.columns)}", 400


    features = preprocessor.transform(data)
    predictions = best_model_yield.predict(features)

    # Add predictions to the DataFrame
    data['Predicted_Yield'] = predictions

    # Save to CSV
    output_file = 'predictions.csv'
    data.to_csv(output_file, index=False)

    return send_file(output_file, as_attachment=True)


def validat_csv(f):
    # Check if the number of rows is less than or equal to 10
    if len(f) <= 10:
        return "Insufficient data: Heatmaps require more than 10 instances.", 400

    # Ensure all required columns are present
    required_columns = ['Area', 'Year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes', 'Item',
                        'Predicted_Yield', 'Price']
    for col in required_columns:
        if col not in f.columns:
            f[col] = None  # Add missing columns with default values (e.g., None or 0)

    up_f = f.to_csv('updated_file.csv', index=False)
    return up_f



@app.route("/heatmaps", methods=['POST'])
def generate_heatmaps():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read the CSV file
    f = pd.read_csv(file)

    #check and create validity of the csv
    # Check if the number of rows is less than or equal to 10
    if len(f) <= 10:
        return "Insufficient data: Heatmaps require more than 10 instances.", 400

    # Ensure all required columns are present
    required_columns = [ 'Year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes',
                        'Predicted_Yield', 'Price']
    for col in required_columns:
        if col not in f.columns:
            f[col] = None  # Add missing columns with default values (e.g., None or 0)

    f.to_csv('updated_file.csv', index=False) #returns none


    # Generate first heatmap: Area, Year, Rain, Temp vs Yield
    f = f.rename(columns={
        'average_rain_fall_mm_per_year': 'Rainfall',
        'avg_temp': 'Temp',
        'Predicted_Yield': 'Yield',
        'pesticides_tonnes': 'Pesticides',
        'Price': 'Price'
    })
    heatmap1_data = f[[ 'Year', 'Rainfall', 'Temp', 'Yield']].corr()
    plt.figure(figsize=(8, 6))
    plt.title("Correlation Heatmap: Year, Rain, Temp vs Yield")
    sns.heatmap(heatmap1_data, annot=True, cmap="coolwarm", fmt=".2f")
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    heatmap1_url = base64.b64encode(buf1.getvalue()).decode('utf-8')
    buf1.close()

    # Generate second heatmap: Item, Pesticides, Yield, Price
    '''heatmap2_data = updated_file[['Item', 'Pesticides', 'Yield', 'Price']].corr()
    plt.figure(figsize=(8, 6))
    plt.title("Correlation Heatmap: Item, Pesticides, Yield, Price")
    sns.heatmap(heatmap2_data, annot=True, cmap="coolwarm", fmt=".2f")
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    heatmap2_url = base64.b64encode(buf2.getvalue()).decode('utf-8')
    buf2.close()'''

    return render_template('heatmaps.html', heatmap1_url=heatmap1_url)



if __name__ == "__main__":
    app.run(debug=True)
