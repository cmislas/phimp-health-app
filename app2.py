import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import joblib
import requests

app = Flask(__name__)

# Load the trained models
model_heart = joblib.load('heart_disease_risk_model.pkl')
model_diabetes = joblib.load('diabetes_risk_model.pkl')

# Define the file path for the cleaned combined data
cleaned_data_file = '/Users/claudiaislas/Desktop/Fitabase Data 4.12.16-5.12.16/cleaned_combined_data.csv'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Read the cleaned combined data
    data = pd.read_csv(cleaned_data_file)

    # Perform data analysis
    summary = data.describe().to_html()

    # Generate plot
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Predict heart disease and diabetes risk
    features = ['HeartRate', 'TotalSteps', 'Calories', 'MinutesAsleep', 'Weight']
    data['HeartDiseaseRisk'] = model_heart.predict(data[features])
    data['DiabetesRisk'] = model_diabetes.predict(data[features])

    # Identify users at risk
    at_risk_heart = data[data['HeartDiseaseRisk'] == 1]
    at_risk_diabetes = data[data['DiabetesRisk'] == 1]
    risk_message_heart = f"{len(at_risk_heart)} users are at risk of heart disease."
    risk_message_diabetes = f"{len(at_risk_diabetes)} users are at risk of diabetes."

    return render_template('result.html', summary=summary, plot_url=plot_url,
                          risk_message_heart=risk_message_heart,
                          risk_message_diabetes=risk_message_diabetes)

if __name__ == '__main__':
    app.run(debug=True, port=5006)
