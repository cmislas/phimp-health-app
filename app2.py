import os
import requests
import pandas as pd
import joblib
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URLs to your CSV files on GitHub
base_url = "https://raw.githubusercontent.com/cmislas/phimp-health-app/main/"
combined_data_url = base_url + "cleaned_combined_data.csv"

# Function to download files from GitHub
def download_file(url, filename):
    logger.info(f"Downloading file from {url}")
    response = requests.get(url)
    logger.info(f"Response status code: {response.status_code}")
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    with open(filename, 'wb') as f:
        f.write(response.content)
    logger.info(f"File saved as {filename}")

# Download data files
try:
    combined_data_file = 'cleaned_combined_data.csv'
    download_file(combined_data_url, combined_data_file)

    # Load data
    combined_data = pd.read_csv(combined_data_file)

    # Verify column names
    print(combined_data.columns)

    # Load models
    model_heart = joblib.load('heart_disease_risk_model.pkl')
    model_diabetes = joblib.load('diabetes_risk_model.pkl')
except Exception as e:
    logger.error(f"An error occurred: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Perform data analysis
        summary = combined_data.describe().to_html()

        # Generate plot
        fig, ax = plt.subplots()
        combined_data.plot(ax=ax)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Predict heart disease and diabetes risk
        features = ['HeartRate', 'TotalSteps', 'Calories', 'MinutesAsleep', 'Weight']  
        combined_data['HeartDiseaseRisk'] = model_heart.predict(combined_data[features])
        combined_data['DiabetesRisk'] = model_diabetes.predict(combined_data[features])

        # Identify users at risk
        at_risk_heart = combined_data[combined_data['HeartDiseaseRisk'] == 1]
        at_risk_diabetes = combined_data[combined_data['DiabetesRisk'] == 1]
        risk_message_heart = f"{len(at_risk_heart)} users are at risk of heart disease."
        risk_message_diabetes = f"{len(at_risk_diabetes)} users are at risk of diabetes."

        return render_template('result.html', summary=summary, plot_url=plot_url,
                               risk_message_heart=risk_message_heart,
                               risk_message_diabetes=risk_message_diabetes)
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        return "An error occurred during upload."

if __name__ == '__main__':
    app.run(debug=True, port=5005)

