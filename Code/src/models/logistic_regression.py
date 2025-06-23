import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def train_logistic_regression(df):
    """
    Trains a Logistic Regression model and returns metrics and probabilities.
    """
    features = ['NetFamilyIncome', 'BoardsPercentage', 'DisabilityStatus', 'ParentsStatus']
    X = df[features]
    y = df['Scholarship']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    all_probs = lr_model.predict_proba(X_scaled)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_test, lr_probs),
        'accuracy': accuracy_score(y_test, (lr_probs >= 0.5).astype(int)),
        'precision': precision_score(y_test, (lr_probs >= 0.5).astype(int)),
        'recall': recall_score(y_test, (lr_probs >= 0.5).astype(int)),
    }
    return lr_model, metrics, all_probs
