import pandas as pd
import matplotlib.pyplot as plt

# Define the filename (replace if needed)
filename = "KJS_data.csv"

# Read the CSV file
data = pd.read_csv(filename)

# Specify column names (replace if different)
text_column = "Final Selection(750)"
probability_column = "LogisticRegression_Prob"

# Get the selected text (replace with your actual selection)
selected_text = "Not eligible"

# Filter data based on selected text
filtered_data = data[data[text_column] == selected_text]

# Create the histogram plot
plt.hist(filtered_data[probability_column], bins="auto")  # Adjust bins as needed
plt.xlabel(f"Probabilities ({probability_column})")
plt.ylabel("Frequency")
plt.title(f"Frequency of Probabilities for '{selected_text}'")
plt.show()