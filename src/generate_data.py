import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the sample data
df = pd.read_csv('../data/ecommerce_data.csv')

# Create lists for new data
categories = df['category'].unique()
product_names = df['product_name'].unique()
reviews = df['review'].unique()

# Function to generate random dates
def random_dates(start, end, n):
    start_u = start.timestamp()
    end_u = end.timestamp()
    return pd.to_datetime([datetime.fromtimestamp(np.random.uniform(start_u, end_u)) for _ in range(n)])

# Generate 12000 records
n_records = 12000
new_data = {
    'user_id': np.random.randint(1001, 5001, n_records),
    'product_id': [f'P{str(i).zfill(3)}' for i in np.random.randint(1, 101, n_records)],
    'product_name': np.random.choice(product_names, n_records),
    'category': np.random.choice(categories, n_records),
    'price': np.random.uniform(10, 500, n_records).round(2),
    'rating': np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0], n_records),
    'purchase_date': random_dates(datetime(2025, 1, 1), datetime(2025, 5, 27), n_records).strftime('%Y-%m-%d'),
    'review': np.random.choice(reviews, n_records)
}

# Create DataFrame and save
new_df = pd.DataFrame(new_data)
new_df.to_csv('../data/ecommerce_data.csv', index=False)
print(f"Generated {n_records} records of e-commerce data")
