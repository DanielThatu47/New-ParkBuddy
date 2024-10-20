from flask import Flask, render_template, jsonify, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from predict import predict_availability

app = Flask(__name__, static_url_path='/static')

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    # Calculate total occupancy (average across all slots)
    total_occupancy = df[['slot1', 'slot2', 'slot3']].mean().mean() * 100

    # Calculate occupancy for each hour
    hourly_occupancy = df.groupby('time')[['slot1', 'slot2', 'slot3']].mean().mean(axis=1) * 100

    # Find busiest and quietest times
    busiest_time = hourly_occupancy.idxmax()
    quietest_time = hourly_occupancy.idxmin()

    # Calculate average occupancy by day of week
    df['day_of_week'] = pd.to_datetime(df['day'], format='%d').dt.day_name()
    avg_occupancy = df.groupby('day_of_week')[['slot1', 'slot2', 'slot3']].mean().mean(axis=1) * 100
    avg_occupancy = avg_occupancy.to_dict()

    # Calculate slot utilization
    slot_utilization = df[['slot1', 'slot2', 'slot3']].mean() * 100
    slot_utilization = slot_utilization.to_dict()

    # Create occupancy graph
    plt.figure(figsize=(10, 6))
    hourly_occupancy.plot(kind='line')
    plt.title('Hourly Occupancy')
    plt.xlabel('Hour')
    plt.ylabel('Occupancy (%)')
    occupancy_graph = plot_to_base64(plt)

    # Create slot utilization graph
    plt.figure(figsize=(8, 6))
    plt.bar(slot_utilization.keys(), slot_utilization.values())
    plt.title('Slot Utilization')
    plt.xlabel('Slot')
    plt.ylabel('Utilization (%)')
    slot_utilization_graph = plot_to_base64(plt)

    return jsonify({
        'total_occupancy': f"{total_occupancy:.2f}%",
        'busiest_time': str(busiest_time),
        'quietest_time': str(quietest_time),
        'avg_occupancy': avg_occupancy,
        'slot_utilization': slot_utilization,
        'hourly_occupancy': hourly_occupancy.to_dict(),
        'occupancy_graph': occupancy_graph,
        'slot_utilization_graph': slot_utilization_graph
    })

@app.route('/predict', methods=['POST'])
def predict():
    day = int(request.form['day'])
    time = int(request.form['time'])
    slot = int(request.form['slot'])

    prediction, percentage = predict_availability(day, time, slot)
    return jsonify({'prediction': prediction, 'percentage': percentage})

@app.route('/login')
def login():
    return render_template('login.html')

def generate_occupancy_graph():
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='time', y='total_occupancy', marker='o')
    plt.title('Average Number of People Throughout the Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of People')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_slot_utilization_graph():
    slot_utilization = {
        'Morning': df['slot1'].mean(),
        'Afternoon': df['slot2'].mean(),
        'Evening': df['slot3'].mean(),
    }
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(slot_utilization.keys()), y=list(slot_utilization.values()))
    plt.title('Average Number of People in Each Time Slot')
    plt.xlabel('Time Slot')
    plt.ylabel('Average Number of People')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_to_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
