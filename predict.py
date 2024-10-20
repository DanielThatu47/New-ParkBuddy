import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv('categorized_synthetic_data_month.csv')

# Clean and process the data
df['day'] = pd.to_numeric(df['day'], errors='coerce')
df['time'] = df['time'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else None)
df['slot1'] = pd.to_numeric(df['slot1'], errors='coerce')
df['slot2'] = pd.to_numeric(df['slot2'], errors='coerce')
df['slot3'] = pd.to_numeric(df['slot3'], errors='coerce')

# Drop rows with NaN values in essential columns
df.dropna(subset=['day', 'time', 'slot1', 'slot2', 'slot3'], inplace=True)

# Ensure 'day' and 'time' are integers
df['day'] = df['day'].astype(int)
df['time'] = df['time'].astype(int)

# Prepare the data for machine learning
X = df[['day', 'time']]
y1 = df['slot1'] < 1  # True if available, False if occupied
y2 = df['slot2'] < 1
y3 = df['slot3'] < 1

# Train a Random Forest Classifier for each slot
models = {}
for slot in [1, 2, 3]:
    X_train, X_test, y_train, y_test = train_test_split(X, locals()[f'y{slot}'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models[slot] = model

def predict_availability(day, time, slot):
    """Predict the availability of a specific slot on a specific day and time."""
    if slot not in models:
        return f"Invalid slot number: {slot}", 0

    prediction = models[slot].predict([[day, time]])[0]
    probability = models[slot].predict_proba([[day, time]])[0][1]  # Probability of being available
    availability_percentage = probability * 100

    if prediction:
        availability_status = "Likely Available"
    else:
        availability_status = "Likely Not Available"

    return f"Predicted availability for Slot {slot} on Day {day} at {time}:00 is: {availability_status}", availability_percentage

# Example usage (you can remove this part when integrating with Flask)
if __name__ == "__main__":
    try:
        day_to_predict = int(input("Enter the day (1-30) to predict availability: "))
        time_to_predict = int(input("Enter the hour (0-23) to predict availability: "))
        slot_to_predict = int(input("Enter the slot to predict availability (1, 2, or 3): "))

        if 1 <= day_to_predict <= 30 and 0 <= time_to_predict <= 23 and slot_to_predict in [1, 2, 3]:
            prediction, percentage = predict_availability(day_to_predict, time_to_predict, slot_to_predict)
            print(prediction)
            print(f"Chance of availability: {percentage:.2f}%")
        else:
            print("Invalid input. Please ensure day is between 1-30, time is between 0-23, and slot is 1, 2, or 3.")
    except ValueError:
        print("Please enter valid integers for day, time, and slot.")
