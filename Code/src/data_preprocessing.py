# Data Preprocessing Module
import pandas as pd

def load_and_process_data(filepath):
    """
    Loads and processes the KJS_data.csv file, returning a cleaned DataFrame.
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    # Use column names, not indices
    df['NetFamilyIncome'] = pd.to_numeric(df['Actual Income'], errors='coerce')
    df['BoardsPercentage'] = pd.to_numeric(df['Class 10 Percentage'], errors='coerce')
    df['DisabilityStatus'] = df['Physicallychallenged'].apply(lambda x: 0 if str(x).strip().lower() == "no" else 1)
    df['ParentsStatus'] = df['Otherpersonaldetails'].map({"Single parent alive": 1, "Orphan": 2}).fillna(0).astype(int)
    df['Scholarship'] = ((df['NetFamilyIncome'] < 320000) & (df['BoardsPercentage'] > 85)).astype(int)
    return df
