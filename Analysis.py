import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

# Load the CSV file
file_path = 'All_Data.csv'
students_data = pd.read_csv(file_path)

# Clean column names
students_data.columns = students_data.columns.str.strip()

# Extract relevant columns
relevant_columns = [
    'Familyincome',
    'Class 10 Percentage',
    'Disability Percent',
    'Is Student Single Parents',
    'Final Selection(750)'
]

students_data_filtered = students_data[relevant_columns].copy()

# Convert target variable to binary (1 if selected, 0 otherwise)
students_data_filtered['Final Selection(750)'] = students_data_filtered['Final Selection(750)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Check the distribution of the target variable
print(students_data_filtered['Final Selection(750)'].value_counts())

# If only one class is present, display an appropriate message
if students_data_filtered['Final Selection(750)'].nunique() == 1:
    print("The dataset contains only one class in the target variable. Model training requires both classes (0 and 1).")
else:
    # Convert non-numeric values in numeric columns to NaN
    students_data_filtered['Familyincome'] = pd.to_numeric(students_data_filtered['Familyincome'], errors='coerce')
    students_data_filtered['Class 10 Percentage'] = pd.to_numeric(students_data_filtered['Class 10 Percentage'], errors='coerce')
    students_data_filtered['Disability Percent'] = pd.to_numeric(students_data_filtered['Disability Percent'], errors='coerce')

    # Define features and target variable
    X = students_data_filtered.drop(columns='Final Selection(750)')
    y = students_data_filtered['Final Selection(750)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the distribution of the target variable in the training set
    print(y_train.value_counts())

    # If imbalance is detected, resample the training set
    if y_train.nunique() == 1:
        # Combine X_train and y_train for resampling
        training_data = pd.concat([X_train, y_train], axis=1)

        # Separate majority and minority classes
        majority_class = training_data[training_data['Final Selection(750)'] == 0]
        minority_class = training_data[training_data['Final Selection(750)'] == 1]

        # Upsample minority class
        minority_upsampled = resample(minority_class,
                                      replace=True,  # sample with replacement
                                      n_samples=len(majority_class),  # to match majority class
                                      random_state=42)  # reproducible results

        # Combine majority class with upsampled minority class
        training_data_upsampled = pd.concat([majority_class, minority_upsampled])

        # Separate X_train and y_train after resampling
        X_train = training_data_upsampled.drop(columns='Final Selection(750)')
        y_train = training_data_upsampled['Final Selection(750)']

    # Define preprocessing for numerical and categorical data
    numerical_features = ['Familyincome', 'Class 10 Percentage', 'Disability Percent']
    categorical_features = ['Is Student Single Parents']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='No')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    students_data_filtered['Probability'] = model.predict_proba(X)[:, 1]

    # Display the first few rows with the new 'Probability' column
    print(students_data_filtered.head())
