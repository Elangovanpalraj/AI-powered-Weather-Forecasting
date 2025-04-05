from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(r"D:\STET-details\msc-\Thillainayaki\Thillai\Machine-Learning-Model-for-Weather-Forecasting-main\weather_model.pkl")

# Load model metrics
with open("model_metrics.txt", "r") as f:
    model_metrics = f.read()

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        inputs = [float(request.form[col]) for col in [
            'maxtempC', 'mintempC', 'cloudcover', 'humidity',
            'sunHour', 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph'
        ]]
        prediction = model.predict([inputs])[0]
        return render_template('index.html', prediction=round(prediction, 2), metrics=model_metrics)
    except:
        return render_template('index.html', error="Invalid input or missing values.", metrics=model_metrics)

if __name__ == '__main__':
    app.run(debug=True)
