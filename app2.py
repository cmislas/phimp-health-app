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
cleaned_data_url = base_url + "cleaned_combined_data.csv"

# Function to download files from GitHub
def download_file(url, filename):
    logger.info(f"Downloading file from {url}")
    response = requests.get(url)
    logger.info(f"Response status code: {response.status_code}")
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    with open(filename, 'wb') as f:
        f.write(response.content)
    logger.info(f"File saved as {filename}")

# Download and load data files
try:
    cleaned_data_file = 'cleaned_combined_data.csv'
    download_file(cleaned_data_url, cleaned_data_file)

    # Load data
    data = pd.read_csv(cleaned_data_file)

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
        summary = data.describe().to_html()

        # Generate plot
        fig, ax = plt.subplots()
        data.plot(ax=ax)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Correct column names
        features = ['HeartRate', 'TotalSteps', 'Calories', 'MinutesAsleep', 'Weight']
        logger.info(f"Features used for prediction: {features}")
        logger.info(f"Available columns in data dataframe: {data.columns}")

        # Predict heart disease and diabetes risk
        try:
            data['HeartDiseaseRisk'] = model_heart.predict(data[features])
            data['DiabetesRisk'] = model_diabetes.predict(data[features])
            logger.info("Predictions made successfully.")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "An error occurred during prediction."

        # Identify users at risk
        at_risk_heart = data[data['HeartDiseaseRisk'] == 1]
        at_risk_diabetes = data[data['DiabetesRisk'] == 1]
        risk_message_heart = f"{len(at_risk_heart)} users are at risk of heart disease."
        risk_message_diabetes = f"{len(at_risk_diabetes)} users are at risk of diabetes."

        logger.info(risk_message_heart)
        logger.info(risk_message_diabetes)

        return render_template('result.html', summary=summary, plot_url=plot_url,
                               risk_message_heart=risk_message_heart,
                               risk_message_diabetes=risk_message_diabetes)
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        return "An error occurred during upload."

if __name__ == '__main__':
    app.run(debug=True, port=5005)


