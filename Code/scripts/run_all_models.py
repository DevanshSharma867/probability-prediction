import pandas as pd
from src.data_preprocessing import load_and_process_data
from src.models.random_forest import train_random_forest
from src.models.logistic_regression import train_logistic_regression
from src.models.naive_bayes import train_naive_bayes

def main():
    df = load_and_process_data("../data/KJS_data.csv")
    # Train models
    rf_model, rf_metrics, rf_probs = train_random_forest(df)
    lr_model, lr_metrics, lr_probs = train_logistic_regression(df)
    nb_model, nb_metrics, nb_probs = train_naive_bayes(df)
    # Save probabilities to DataFrame
    df['RandomForest_Prob'] = rf_probs
    df['LogisticRegression_Prob'] = lr_probs
    df['NaiveBayes_Prob'] = nb_probs
    df.to_csv("../results/model_outputs.csv", index=False)
    print("Random Forest Metrics:", rf_metrics)
    print("Logistic Regression Metrics:", lr_metrics)
    print("Naive Bayes Metrics:", nb_metrics)

if __name__ == "__main__":
    main()
