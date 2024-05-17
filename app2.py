import os
import requests
import pandas as pd
import joblib
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# URLs to your CSV files on GitHub
base_url = "https://raw.githubusercontent.com/cmislas/phimp-health-app/main/"
daily_activity_url = 'https://raw.githubusercontent.com/cmislas/phimp-health-app/main/dailyActivity_merged.csv'

heartrate_seconds_url = base_url + "heartrate_seconds_sampled.csv"  # Use the sampled file
sleep_data_url = base_url + "sleepDay_merged.csv"
weight_data_url = base_url + "weightLogInfo_merged.csv"

# Function to download files from GitHub
def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Download data files
daily_activity_file = 'dailyActivity_merged.csv'
heartrate_seconds_file = 'heartrate_seconds_sampled.csv'  # Use the sampled file
sleep_data_file = 'sleepDay_merged.csv'
weight_data_file = 'weightLogInfo_merged.csv'

download_file(daily_activity_url, daily_activity_file)
download_file(heartrate_seconds_url, heartrate_seconds_file)
download_file(sleep_data_url, sleep_data_file)
download_file(weight_data_url, weight_data_file)

# Load data
daily_activity = pd.read_csv(daily_activity_file)
heartrate_seconds = pd.read_csv(heartrate_seconds_file)
sleep_data = pd.read_csv(sleep_data_file)
weight_data = pd.read_csv(weight_data_file)

# Load models
model_heart = joblib.load('heart_disease_risk_model.pkl')
model_diabetes = joblib.load('diabetes_risk_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Perform data analysis
    summary = daily_activity.describe().to_html()

    # Generate plot
    fig, ax = plt.subplots()
    daily_activity.plot(ax=ax)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Predict heart disease and diabetes risk
    features = ['HeartRate', 'TotalSteps', 'Calories', 'MinutesAsleep', 'Weight']
    daily_activity['HeartDiseaseRisk'] = model_heart.predict(daily_activity[features])
    daily_activity['DiabetesRisk'] = model_diabetes.predict(daily_activity[features])

    # Identify users at risk
    at_risk_heart = daily_activity[daily_activity['HeartDiseaseRisk'] == 1]
    at_risk_diabetes = daily_activity[daily_activity['DiabetesRisk'] == 1]
    risk_message_heart = f"{len(at_risk_heart)} users are at risk of heart disease."
    risk_message_diabetes = f"{len(at_risk_diabetes)} users are at risk of diabetes."

    return render_template('result.html', summary=summary, plot_url=plot_url,
                           risk_message_heart=risk_message_heart,
                           risk_message_diabetes=risk_message_diabetes)

if __name__ == '__main__':
    app.run(debug=True, port=5005)





