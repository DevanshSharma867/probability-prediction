import pandas as pd

def export_selected_with_probabilities(csv_path, output_path, selection_col, selection_value):
    df = pd.read_csv(csv_path)
    selected = df[df[selection_col] == selection_value]
    selected[["Firstname", "Last Name", "NaiveBayes_Prob", "LogisticRegression_Prob"]].to_csv(output_path, index=False)

if __name__ == "__main__":
    export_selected_with_probabilities("../results/model_outputs.csv", "../results/Selected_with_probabilities.csv", "Final Selection(750)", "Selected")
