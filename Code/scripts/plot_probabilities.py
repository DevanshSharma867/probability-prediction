import pandas as pd
import matplotlib.pyplot as plt

def plot_probabilities(csv_path, model_col, selection_col, selection_value, plot_type="hist"):
    df = pd.read_csv(csv_path)
    filtered = df[df[selection_col] == selection_value]
    if plot_type == "hist":
        plt.hist(filtered[model_col], bins="auto")
        plt.xlabel(f"Probabilities ({model_col})")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of Probabilities for '{selection_value}'")
    elif plot_type == "line":
        value_counts = filtered[model_col].value_counts().sort_index()
        plt.plot(value_counts.index, value_counts.values, marker='o', linestyle='-')
        plt.xlabel(f"Probabilities ({model_col})")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of Probabilities for '{selection_value}'")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_probabilities("../results/model_outputs.csv", "NaiveBayes_Prob", "Final Selection(750)", "Selected", plot_type="line")
    plot_probabilities("../results/model_outputs.csv", "LogisticRegression_Prob", "Final Selection(750)", "Not eligible", plot_type="hist")
