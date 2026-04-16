# Real Estate Linear Regression Assignment

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('Real estate.csv')

# 1. Data Understanding
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Drop unnecessary column if exists
if 'No' in df.columns:
    df.drop('No', axis=1, inplace=True)

# 2. Preprocessing
print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

# Features and target
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 6. Save Predictions
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

results.to_csv('predictions.csv', index=False)

print("\nPredictions saved successfully!")
