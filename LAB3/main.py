# GitHub REPO (LAB3) https://github.com/oringejooz/AI-MLLab.git

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target  # Add target variable

print("Dataset loaded. Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Handling missing values (if any)
df.fillna(df.median(), inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Feature Engineering
df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['AveOccup']

# Exploratory Data Analysis
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Pair plots
sns.pairplot(df.sample(500))  # Sample to avoid performance issues
plt.show()

# Splitting data
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

# Train and evaluate model
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Cross-Validation RMSE: {cv_rmse.mean():.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# Save model
joblib.dump(pipeline, 'best_linear_regression_model.pkl')
print("\nBest model saved as 'best_linear_regression_model.pkl'")

# Make Predictions
sample_house = X_test.iloc[0:1]
predicted_price = pipeline.predict(sample_house)
print(f"\nPredicted House Price: {predicted_price[0]:.4f}")
print(f"Actual House Price: {y_test.iloc[0]:.4f}")
