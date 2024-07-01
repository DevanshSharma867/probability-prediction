import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import csv

# Lists to store the data
NetFamilyIncome = []
BoardsPercentage = []
DisabilityStatus = []
ParentsStatus = []
Scholarship = []
indices = []

# Read the CSV file and process the data
try:
    with open("KJS_data.csv", "r", encoding='utf-8') as file:
        content = csv.reader(file)
        header = next(content)  # Skip the header row
        for idx, row in enumerate(content):
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

            except (ValueError, IndexError):
                # Skip rows with invalid or missing data
                continue

except PermissionError:
    print("Permission denied: Unable to read the file.")
except FileNotFoundError:
    print("File not found: Please check the file path.")
except UnicodeDecodeError:
    print("Error decoding file: Check the file encoding and try again.")

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

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
# Setting test_size to 0.2 and using the original number of samples for the test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Create and fit Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

# Evaluate Logistic Regression model
lr_roc_auc = roc_auc_score(y_test, lr_probs)
lr_accuracy = accuracy_score(y_test, (lr_probs >= 0.5).astype(int))
lr_precision = precision_score(y_test, (lr_probs >= 0.5).astype(int))
lr_recall = recall_score(y_test, (lr_probs >= 0.5).astype(int))

print(f"Logistic Regression - ROC-AUC: {lr_roc_auc}, Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}")

# Predict probabilities for all instances
all_probs = lr_model.predict_proba(X_scaled)[:, 1]

# Print the probabilities from Logistic Regression model for all instances
print("\nLogistic Regression Probabilities for all instances:")
print(len(all_probs))