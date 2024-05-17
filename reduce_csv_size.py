import pandas as pd

# Load the large CSV file
heartrate_seconds_file = 'heartrate_seconds_merged.csv'
heartrate_seconds = pd.read_csv(heartrate_seconds_file)

# Sample 10% of the data
heartrate_seconds_sampled = heartrate_seconds.sample(frac=0.1, random_state=1)

# Save the sampled data to a new CSV file
heartrate_seconds_sampled.to_csv('heartrate_seconds_sampled.csv', index=False)
