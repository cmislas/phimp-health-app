from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import os

# Load cleaned data
base_url = "https://raw.githubusercontent.com/cmislas/phimp-health-app/main/"
cleaned_data_file = base_url + "cleaned_combined_data.csv"

# Define features and target variable
features = ['HeartRate', 'TotalSteps', 'Calories', 'MinutesAsleep', 'Weight']
data['HeartDiseaseRisk'] = ((data['HeartRate'] > 100) | (data['TotalSteps'] < 5000)) & (data['Weight'] > 80)
data['DiabetesRisk'] = ((data['Calories'] > 2500) | (data['MinutesAsleep'] < 300))

# Check the distribution of the target variables
print(data['HeartDiseaseRisk'].value_counts())
print(data['DiabetesRisk'].value_counts())

X = data[features]
y_heart = data['HeartDiseaseRisk']
y_diabetes = data['DiabetesRisk']


# Ensure that both classes are present
if len(data['HeartDiseaseRisk'].unique()) < 2:
    raise ValueError("HeartDiseaseRisk data contains only one class. Please adjust the conditions to ensure both classes are present.")
if len(data['DiabetesRisk'].unique()) < 2:
    raise ValueError("DiabetesRisk data contains only one class. Please adjust the conditions to ensure both classes are present.")


# Split the data into training and testing sets
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X, y_heart, test_size=0.3, random_state=42)
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X, y_diabetes, test_size=0.3, random_state=42)

# Train the logistic regression model
model_heart = LogisticRegression()
model_heart.fit(X_train_heart, y_train_heart)
joblib.dump(model_heart, 'heart_disease_risk_model.pkl')

model_diabetes = LogisticRegression()
model_diabetes.fit(X_train_diabetes, y_train_diabetes)
joblib.dump(model_diabetes, 'diabetes_risk_model.pkl')


