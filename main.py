import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocess import preprocess
from src.features import create_features
from src.model import train_model
from src.evaluate import evaluate
from src.visualize import plot_results

# Load data
df = load_data("data/energy.csv")

# Process
df = preprocess(df)
df = create_features(df)

# Split
X = df[['day','month','day_of_week','lag1']]
y = df['energy']

split = int(len(df)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train
model = train_model(X_train, y_train)

# Predict
predictions = model.predict(X_test)
output = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

output.to_csv("outputs/predictions.csv", index=False)

# Evaluate
rmse, r2 = evaluate(y_test, predictions)

