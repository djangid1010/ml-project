from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('solar_flare_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Only use the xray_flux for prediction
        xray_flux = float(request.form['xray_flux'])

        # Make prediction using only the one feature
        prediction = model.predict(np.array([[xray_flux]]))

        # Determine flare status based on the prediction
        flare_status = 'Flare' if prediction[0] == 1 else 'No Flare'

        # Render prediction page with result
        return render_template('prediction.html', prediction=flare_status)
    else:
        return render_template('prediction.html')  # Render the page on GET request


if __name__ == "__main__":
    app.run(debug=True)