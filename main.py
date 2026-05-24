import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("data/energy.csv")

# Display First 5 Rows
print("\nDataset Preview:\n")
print(df.head())

# Convert Date Column
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
df['Day'] = df['date'].dt.day
df['Month'] = df['date'].dt.month
df['Year'] = df['date'].dt.year

# Features and Target
X = df[['Day', 'Month', 'Year']]
y = df['energy']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Linear Regression Model
# =========================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# =========================
# Random Forest Model
# =========================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
import joblib

# Save Random Forest Model
joblib.dump(rf_model, "models/energy_model.pkl")

rf_predictions = rf_model.predict(X_test)

# =========================
# Evaluation
# =========================
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

lr_r2 = r2_score(y_test, lr_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n===== MODEL PERFORMANCE =====")

print("\nLinear Regression")
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

print("\nRandom Forest")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# =========================
# Comparison DataFrame
# =========================
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Linear Regression": lr_predictions,
    "Random Forest": rf_predictions
})

print("\nPrediction Comparison:\n")
print(comparison.head())

# Save Predictions
comparison.to_csv("outputs/predictions.csv", index=False)

# =========================
# Visualization
# =========================
plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values,
    label="Actual",
    linewidth=2
)

plt.plot(
    lr_predictions,
    label="Linear Regression Predictions"
)

plt.plot(
    rf_predictions,
    label="Random Forest Predictions"
)

plt.xlabel("Samples")
plt.ylabel("Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")

plt.legend()

plt.savefig("images/prediction_graph.png")

plt.show()

# =========================
# Model Comparison Graph
# =========================

models = ['Linear Regression', 'Random Forest']
rmse_values = [lr_rmse, rf_rmse]

plt.figure(figsize=(7,5))

plt.bar(models, rmse_values)

plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")

plt.savefig("images/model_comparison.png")

plt.show()

# =========================
# Correlation Heatmap
# =========================

import seaborn as sns

plt.figure(figsize=(8,6))

sns.heatmap(
    df[['energy', 'Day', 'Month', 'Year']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.savefig("images/heatmap.png")

plt.show()