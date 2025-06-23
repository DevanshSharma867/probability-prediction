import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def train_random_forest(df):
    """
    Trains a Random Forest model and returns metrics and probabilities.
    """
    features = ['NetFamilyIncome', 'BoardsPercentage', 'DisabilityStatus', 'ParentsStatus']
    X = df[features]
    y = df['Scholarship']
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_probs_full = rf_model.predict_proba(X_scaled)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_test, rf_probs),
        'accuracy': accuracy_score(y_test, (rf_probs >= 0.5).astype(int)),
        'precision': precision_score(y_test, (rf_probs >= 0.5).astype(int)),
        'recall': recall_score(y_test, (rf_probs >= 0.5).astype(int)),
    }
    return rf_model, metrics, rf_probs_full
