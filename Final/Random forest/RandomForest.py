import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Read the original CSV file into a DataFrame
try:
    df_original = pd.read_csv("KJS_data.csv", encoding='utf-8')

    # Lists to store the processed data
    NetFamilyIncome = []
    BoardsPercentage = []
    DisabilityStatus = []
    ParentsStatus = []
    Scholarship = []
    indices = []

    for idx, row in df_original.iterrows():
        try:
            net_income = int(row[120])
            boards_percentage = float(row[39])
            disability_status = 0 if row[15] == "No" else 1
            if row[16] == "Single parent alive":
                parents_status = 1
            elif row[16] == "Orphan":
                parents_status = 2
            else:
                parents_status = 0

            # Append values to respective lists
            NetFamilyIncome.append(net_income)
            BoardsPercentage.append(boards_percentage)
            DisabilityStatus.append(disability_status)
            ParentsStatus.append(parents_status)
            indices.append(idx)  # Track original indices

            # Determine scholarship eligibility
            if (net_income < 320000) and (boards_percentage > 85):
                Scholarship.append(1)
            else:
                Scholarship.append(0)

        except ValueError:
            # Skip rows with invalid or missing data
            continue

    # Creating DataFrame
    data = {
        'NetFamilyIncome': NetFamilyIncome,
        'BoardsPercentage': BoardsPercentage,
        'DisabilityStatus': DisabilityStatus,
        'ParentsStatus': ParentsStatus,
        'Scholarship': Scholarship
    }

    df = pd.DataFrame(data)

    # Features and target variable
    X = df[['NetFamilyIncome', 'BoardsPercentage', 'DisabilityStatus', 'ParentsStatus']]
    y = df['Scholarship']

    # Handle missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create and fit Random Forest model with tuned hyperparameters
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    # Calculate probabilities for the entire dataset for Random Forest
    rf_probs_full = rf_model.predict_proba(X_scaled)[:, 1]

    # Append probabilities to the original DataFrame
    df_original.loc[indices, 'RandomForest_Prob'] = rf_probs_full

    # Evaluate Random Forest model
    rf_roc_auc = roc_auc_score(y_test, rf_probs)
    rf_accuracy = accuracy_score(y_test, (rf_probs >= 0.5).astype(int))
    rf_precision = precision_score(y_test, (rf_probs >= 0.5).astype(int))
    rf_recall = recall_score(y_test, (rf_probs >= 0.5).astype(int))

    print(f"Random Forest - ROC-AUC: {rf_roc_auc}, Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}")

    print("Length of Random Forest probabilities:", len(rf_probs_full))

    for row in rf_probs_full:
        print(row)
    # Save the modified DataFrame back to the original CSV file
    # df_original.to_csv("KJS_data.csv", index=False)

except PermissionError:
    print("Permission denied: Unable to read/write the file.")
except FileNotFoundError:
    print("File not found: Please check the file path.")
except UnicodeDecodeError:
    print("Error decoding file: Check the file encoding and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
