import pandas as pd
import numpy as np

# Define the time intervals and their corresponding categories
time_intervals = {
    'early morning': [f'{i}-{i+1}' for i in range(4, 8)],  # 4-5, 5-6, 6-7, 7-8
    'morning': [f'{i}-{i+1}' for i in range(8, 12)],       # 8-9, 9-10, 10-11, 11-12
    'afternoon': [f'{i}-{i+1}' for i in range(12, 16)],    # 12-13, 13-14, 14-15, 15-16
    'evening': [f'{i}-{i+1}' for i in range(16, 20)],      # 16-17, 17-18, 18-19, 19-20
    'night': [f'{i}-{i+1}' for i in range(20, 24)],        # 20-21, 21-22, 22-23, 23-24
    'midnight': [f'{i}-{i+1}' for i in range(0, 4)]        # 0-1, 1-2, 2-3, 3-4
}

# Flatten the dictionary and create a list of time and category pairs
time_category = [(time, category) for category, times in time_intervals.items() for time in times]

# Separate the time and category columns
time, category = zip(*time_category)

# Generate data for 30 days
days = np.arange(1, 31)  # 30 days in a month

# Initialize an empty list to store data
all_data = []

# Probabilities for each category
probabilities = {
    'early morning': 0.1,   # Very less busy
    'morning': 0.5,         # Standard
    'afternoon': 0.9,       # Very busy
    'evening': 0.9,         # Very busy
    'night': 0.5,           # Standard
    'midnight': 0.1         # Very less busy
}

# Loop over each day
for day in days:
    np.random.seed(day)  # For reproducibility based on the day

    # Create a list to store the day's data
    day_data = []

    # Generate data for each time interval
    for t, cat in zip(time, category):
        # Get the probability for the current time category
        prob = probabilities[cat]

        # Generate synthetic data for slots with weighted probability
        slot1 = np.random.choice([0, 1], p=[1 - prob, prob])
        slot2 = np.random.choice([0, 1], p=[1 - prob, prob])
        slot3 = np.random.choice([0, 1], p=[1 - prob, prob])

        # Append the data for the current time slot
        day_data.append([day, t, cat, slot1, slot2, slot3])

    # Convert the day's data to a DataFrame and append to the list
    all_data.append(pd.DataFrame(day_data, columns=['day', 'time', 'category', 'slot1', 'slot2', 'slot3']))

# Concatenate all days into one DataFrame
df = pd.concat(all_data, ignore_index=True)

# Save DataFrame to CSV
df.to_csv('categorized_synthetic_data_month.csv', index=False)

# Display message
print("Dataset saved to 'categorized_synthetic_data_month.csv'")
