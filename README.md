# Probability Prediction: Student Scholarship Eligibility

This project uses machine learning to predict student scholarship eligibility based on income, academic performance, and other student details. The workflow includes data cleaning, feature engineering, model training, and evaluation.

---

## Project Folder Structure

```
probability-prediction/
├── Analysis.py
├── Code/
│   ├── config.yaml
│   ├── results/
│   ├── scripts/
│   │   ├── export_selected.py
│   │   ├── plot_probabilities.py
│   │   └── run_all_models.py
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── utils.py
│   │   └── models/
│   │       ├── logistic_regression.py
│   │       ├── naive_bayes.py
│   │       └── random_forest.py
│   └── tests/
│       └── test_data_preprocessing.py
├── Graphs/
│   ├── boards_percentage_distribution.png
│   ├── disability_status_proportion.png
│   ├── income_distribution.png
│   ├── parents_status_proportion.png
│   └── roc_curve.png
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
```

---

## Features

- Cleans and preprocesses student data ([`src/data_preprocessing.py`](Code/src/data_preprocessing.py))
- Defines a binary target variable for scholarship eligibility
- Trains and evaluates Naive Bayes, Logistic Regression, and Random Forest models ([`src/models/`](Code/src/models/))
- Calculates and saves eligibility probabilities for all students ([`scripts/run_all_models.py`](Code/scripts/run_all_models.py))
- Exports selected students with probabilities ([`scripts/export_selected.py`](Code/scripts/export_selected.py))
- Visualizes data distributions and model performance ([`scripts/plot_probabilities.py`](Code/scripts/plot_probabilities.py))
- Unit test for data preprocessing ([`tests/test_data_preprocessing.py`](Code/tests/test_data_preprocessing.py))

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- pyyaml

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file (e.g., `KJS_data.csv`) in a `data/` directory inside `Code/` (not included due to privacy).
2. Configure paths in [`Code/config.yaml`](Code/config.yaml) if needed.
3. Run all models and generate probabilities:
   ```bash
   python Code/scripts/run_all_models.py
   ```
   This will output results to `Code/results/model_outputs.csv`.

4. Plot probability distributions:
   ```bash
   python Code/scripts/plot_probabilities.py
   ```

5. Export selected students with their probabilities:
   ```bash
   python Code/scripts/export_selected.py
   ```

6. Run unit tests:
   ```bash
   python -m unittest Code/tests/test_data_preprocessing.py
   ```

7. View generated plots in the `Graphs/` folder.

## Notes

- **Data files are not included** due to privacy restrictions.
- The code is modular and can be adapted for similar prediction tasks.
- Use the `Code/src/` and `Code/scripts/` folders for maintainable development.
