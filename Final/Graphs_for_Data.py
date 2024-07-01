import pandas as pd
import matplotlib.pyplot as plt

# Define the filename (replace if needed)
filename = "KJS_data.csv"

# Read the CSV file
data = pd.read_csv(filename)

# Specify column names (replace if different)
text_column = "Final Selection(750) "
probability_column = "NaiveBayes_Prob"

# Get the selected text (replace with your actual selection)
selected_text = "Selected"

# Filter data based on selected text
filtered_data = data[data[text_column] == selected_text]

# Sort the filtered data by probability column for line plot
sorted_data = filtered_data.sort_values(by=probability_column)

# Count frequency of each value
value_counts = sorted_data[probability_column].value_counts()

# Create the line plot
plt.plot(value_counts.index, value_counts.values, marker='o', linestyle='-')

plt.xlabel(f"Probabilities ({probability_column})")
plt.ylabel("Frequency")
plt.title(f"Frequency of Probabilities for '{selected_text}'")
plt.grid(True)
plt.show()
