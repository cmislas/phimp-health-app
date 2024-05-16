import os
import pandas as pd
import numpy as np

# Define the file paths
file_path = '/Users/claudiaislas/Desktop/Fitabase Data 4.12.16-5.12.16'
daily_activity_file = os.path.join(file_path, 'dailyActivity_merged.csv')
heartrate_seconds_file = os.path.join(file_path, 'heartrate_seconds_merged.csv')
sleep_data_file = os.path.join(file_path, 'sleepDay_merged.csv')
weight_data_file = os.path.join(file_path, 'weightLogInfo_merged.csv')

# Load data
daily_activity = pd.read_csv(daily_activity_file)
heartrate_seconds = pd.read_csv(heartrate_seconds_file)
sleep_data = pd.read_csv(sleep_data_file)
weight_data = pd.read_csv(weight_data_file)

# Print columns to verify
print("Daily Activity Columns:", daily_activity.columns)
print("Heart Rate Columns:", heartrate_seconds.columns)
print("Sleep Data Columns:", sleep_data.columns)
print("Weight Data Columns:", weight_data.columns)

# Clean and merge data
heart_rate_avg = heartrate_seconds.groupby('Id')['Value'].mean().reset_index()
minutes_asleep_avg = sleep_data.groupby('Id')['TotalMinutesAsleep'].mean().reset_index()
weight_avg = weight_data.groupby('Id')['WeightKg'].mean().reset_index()

daily_activity = daily_activity.merge(heart_rate_avg, on='Id', how='left')
daily_activity = daily_activity.merge(minutes_asleep_avg, on='Id', how='left')
daily_activity = daily_activity.merge(weight_avg, on='Id', how='left')

daily_activity.rename(columns={
    'Value': 'HeartRate',
    'TotalMinutesAsleep': 'MinutesAsleep',
    'WeightKg': 'Weight'
}, inplace=True)

# Verify the columns after merging and renaming
print("Combined Data Columns:", daily_activity.columns)

# Fill missing values
daily_activity.fillna(daily_activity.mean(numeric_only=True), inplace=True)

# Save the cleaned data
cleaned_data_file = os.path.join(file_path, 'cleaned_combined_data.csv')
daily_activity.to_csv(cleaned_data_file, index=False)
