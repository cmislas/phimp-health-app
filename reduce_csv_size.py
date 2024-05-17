import pandas as pd

# Load the large CSV file
large_csv_file = 'heartrate_seconds_merged.csv'
df = pd.read_csv(large_csv_file)

# Sample the data (e.g., 10% of the original data)
sample_df = df.sample(frac=0.1, random_state=42)

# Save the sampled data to a new CSV file
sample_csv_file = 'heartrate_seconds_sampled.csv'
sample_df.to_csv(sample_csv_file, index=False)

print(f"Sampled data saved to {sample_csv_file}")
