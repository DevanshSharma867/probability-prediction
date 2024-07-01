import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Read the original CSV file into a DataFrame
try:
    df_original = pd.read_csv("KJS_data.csv", encoding='utf-8')

    # Strip leading/trailing spaces from column names
    df_original.columns = df_original.columns.str.strip()

    # Verify column names
    # print("Column names:", df_original.columns.tolist())

    # Process data
    df_processed = df_original[['Actual Income', 'Class 10 Percentage', 'Physicallychallenged', 'Otherpersonaldetails', 'Stream Mentor', "Duplicate"]].copy()

    df_processed['Actual Income'] = pd.to_numeric(df_processed['Actual Income'], errors='coerce')
    df_processed['Class 10 Percentage'] = pd.to_numeric(df_processed['Class 10 Percentage'], errors='coerce')
    df_processed['Physicallychallenged'] = df_processed['Physicallychallenged'].apply(lambda x: 0 if str(x).strip().lower() == "no" else 1)
    df_processed['Otherpersonaldetails'] = df_processed['Otherpersonaldetails'].map({"Single parent alive": 1, "Orphan": 2}).fillna(0).astype(int)

    # Determine scholarship eligibility
    df_processed['Scholarship'] = ((df_processed['Actual Income'] < 320000) & (df_processed['Class 10 Percentage'] > 85)).astype(int)

    # Drop rows with missing values
    df_processed.dropna(inplace=True)

    # Features and target variable
    X = df_processed[['Actual Income', 'Class 10 Percentage', 'Physicallychallenged', 'Otherpersonaldetails']]
    y = df_processed['Scholarship']

    # Handle missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Create and fit Naive Bayes model with hyperparameter tuning
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_probs = nb_model.predict_proba(X_test)[:, 1]

    # Hyperparameter tuning for Logistic Regression
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    lr_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=StratifiedKFold(5))
    lr_model.fit(X_train, y_train)
    lr_best_model = lr_model.best_estimator_
    lr_probs = lr_best_model.predict_proba(X_test)[:, 1]

    # Calculate probabilities for the entire dataset for Naive Bayes
    nb_probs_full = nb_model.predict_proba(X_scaled)[:, 1]

    # Calculate probabilities for the entire dataset for Logistic Regression
    lr_probs_full = lr_best_model.predict_proba(X_scaled)[:, 1]

    # Create a copy of the processed DataFrame to align indices
    df_probs = df_processed.copy()
    df_probs['NaiveBayes_Prob'] = nb_probs_full
    df_probs['LogisticRegression_Prob'] = lr_probs_full

    # Set probabilities to 0 if "Stream Mentor" is not "Commerce", "Arts", or "Science"
    invalid_stream_mask = ~df_processed['Stream Mentor'].isin(["Commerce", "Arts ", "Science "])
    df_probs.loc[invalid_stream_mask, ['NaiveBayes_Prob', 'LogisticRegression_Prob']] = 0
    
    invalid_stream_mask = ~df_processed["Duplicate"].isin([1, 0])
    df_probs.loc[invalid_stream_mask, ['NaiveBayes_Prob', 'LogisticRegression_Prob']] = 0


    # Align the probabilities with the original DataFrame
    df_original['NaiveBayes_Prob'] = pd.NA
    df_original['LogisticRegression_Prob'] = pd.NA
    df_original.loc[df_probs.index, 'NaiveBayes_Prob'] = df_probs['NaiveBayes_Prob']
    df_original.loc[df_probs.index, 'LogisticRegression_Prob'] = df_probs['LogisticRegression_Prob']

    # Evaluate Naive Bayes model
    nb_roc_auc = roc_auc_score(y_test, nb_probs)
    nb_accuracy = accuracy_score(y_test, (nb_probs >= 0.5).astype(int))
    nb_precision = precision_score(y_test, (nb_probs >= 0.5).astype(int))
    nb_recall = recall_score(y_test, (nb_probs >= 0.5).astype(int))

    # Evaluate Logistic Regression model
    lr_roc_auc = roc_auc_score(y_test, lr_probs)
    lr_accuracy = accuracy_score(y_test, (lr_probs >= 0.5).astype(int))
    lr_precision = precision_score(y_test, (lr_probs >= 0.5).astype(int))
    lr_recall = recall_score(y_test, (lr_probs >= 0.5).astype(int))

    print(f"Naive Bayes - ROC-AUC: {nb_roc_auc}, Accuracy: {nb_accuracy}, Precision: {nb_precision}, Recall: {nb_recall}")
    print(f"Logistic Regression - ROC-AUC: {lr_roc_auc}, Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}")

    print("Length of Naive Bayes probabilities:", len(nb_probs_full))
    print("Length of Logistic Regression probabilities:", len(lr_probs_full))

    # Save the modified DataFrame back to the original CSV file
    df_original.to_csv("KJS_data.csv", index=False)

except PermissionError:
    print("Permission denied: Unable to read/write the file.")
except FileNotFoundError:
    print("File not found: Please check the file path.")
except UnicodeDecodeError:
    print("Error decoding file: Check the file encoding and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
