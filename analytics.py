import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/content/categorized_synthetic_data_month.csv')

# Calculate total occupancy for each time slot
df['total_occupancy'] = df['slot1'] + df['slot2'] + df['slot3']

# Create a function to ensure all time slots are displayed
def analyze_specific_day(day):
    """
    Easy-to-understand analysis for a chosen day of the month with clear graph explanations.

    :param day: int, day of the month (1-30)
    """
    if day < 1 or day > 30:
        print("Please enter a valid day between 1 and 30.")
        return

    # Filter data for the chosen day
    day_data = df[df['day'] == day]

    print(f"\nResults for Day {day}:")
    print("----------------------")

    # Total number of people that day
    total_occupancy = day_data['total_occupancy'].sum()
    print(f"Total number of people throughout the day: {total_occupancy}")

    # When the most and fewest people were around
    busiest_time = day_data.loc[day_data['total_occupancy'].idxmax(), 'time']
    quietest_time = day_data.loc[day_data['total_occupancy'].idxmin(), 'time']
    print(f"Most people were present at: {busiest_time}")
    print(f"Fewest people were present at: {quietest_time}")

    # Average number of people by category
    avg_occupancy = day_data.groupby('category')['total_occupancy'].mean().sort_values(ascending=False)
    print("\nAverage number of people by category:")
    print(avg_occupancy)

    # Show how each time slot was used (slot 1 = morning, slot 2 = afternoon, slot 3 = evening)
    # Create a dictionary to hold utilization values
    slot_utilization = {
        'Morning': day_data['slot1'].mean() if 'slot1' in day_data.columns else 0,
        'Afternoon': day_data['slot2'].mean() if 'slot2' in day_data.columns else 0,
        'Evening': day_data['slot3'].mean() if 'slot3' in day_data.columns else 0,
    }

    # Ensure all time slots are displayed
    time_slots = ['slot1', 'slot2', 'slot3']
    time_slot_labels = ['Morning', 'Afternoon', 'Evening']

    print("\nAverage number of people in each time slot:")
    for label in time_slot_labels:
        print(f"{label}: {slot_utilization[label]:.1f}")

    # Graph showing total number of people throughout the day
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=day_data, x='time', y='total_occupancy', marker='o')
    plt.title(f'Number of People Throughout Day {day}')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of People')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("Graph 1: This line chart shows how the number of people changes throughout the day. "
          "The x-axis represents different times of the day, and the y-axis shows how many people were there at each time.")

    # Bar chart showing slot utilization (morning, afternoon, evening)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=time_slot_labels, y=list(slot_utilization.values()))
    plt.title(f'Average Number of People in Each Time Slot for Day {day}')
    plt.xlabel('Time Slot')
    plt.ylabel('Average Number of People')
    plt.tight_layout()
    plt.show()
    print("Graph 2: This bar chart compares the average number of people present during the morning, afternoon, "
          "and evening. It helps you see which time period was the busiest.")

# Prompt the user to enter a day to analyze
day_to_analyze = int(input("Enter a day (1-30) to analyze: "))
analyze_specific_day(day_to_analyze)
