import pandas as pd

# File paths
input_file = './Project 1/nytaxi2022.csv'
output_file = './Project 1/nytaxi2022_first_million.csv'

# Read the first million rows
chunk_size = 10**6
df = pd.read_csv(input_file, nrows=chunk_size)

df_reduced = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 
         'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
         'payment_type', 'extra', 'total_amount']]

# Save to a new CSV file
df.to_csv(output_file, index=False)
df_reduced.to_csv('./Project 1/nytaxi2022_reduced_first_million.csv', index=False)

print(f"First {chunk_size} rows have been written to {output_file}")