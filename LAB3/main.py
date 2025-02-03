import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target  # Add target variable

# Display first 5 rows
print(df.head())

# Data Preprocessing
print(df.isnull().sum())  # Check for missing values
print(df.describe())      # Summary statistics

df = df.dropna()  # Drop missing values

# Exploratory Data Analysis
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# Splitting data into training and testing Sets
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Machine Learning Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2:.4f}")

# Make Predictions
sample_house = X_test.iloc[0:1]  # Select a sample house
predicted_price = model.predict(sample_house)
print(f"Predicted House Price: {predicted_price[0]}")
