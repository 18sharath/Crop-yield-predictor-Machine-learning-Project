# from flask import Flask, request
# import numpy as np
# import pandas as pd
# import pickle

# # creating app flask
# app=Flask(__name__)

#  if __name__=='__main__':
#     app.run(debug=True)






from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#loading models


best_model = pickle.load(open('best_model.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item  = request.form['Item']
        
        features = np.array([[Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]],dtype=object)
        transformed_features = preprocessor.transform(features)
        # prediction = dtr.predict(transformed_features).reshape(1,-1)
        predicted_value= best_model.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = predicted_value)

@app.route('/price',methods=['POST'])
def price()

if __name__=="__main__":
    app.run(debug=True)