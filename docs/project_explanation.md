# 📄 Project Explanation: AI-Powered Energy Consumption Forecasting

## 📌 Objective
The goal of this project is to predict future energy consumption using Machine Learning based on historical time-series data.

---

## ⚙️ Working Process

### 1. Data Collection
- Dataset used: `energy.csv`
- Contains date and energy consumption values

---

### 2. Data Preprocessing
- Converted date column to datetime format
- Handled missing values (if any)
- Sorted data chronologically

---

### 3. Feature Engineering
- Extracted time-based features:
  - Day
  - Month
  - Year

---

### 4. Model Used
- Linear Regression
- Trained on historical data to learn patterns

---

### 5. Prediction
- Model predicts future energy consumption values

---

### 6. Evaluation
- RMSE (Root Mean Square Error)
- R² Score

---

### 7. Visualization
- Actual vs Predicted graph generated
- Helps understand model performance

---

## 📊 Output Files

- `outputs/predictions.csv` → predicted values
- `images/prediction_graph.png` → visualization
- `images/model_output.png` → model results

---

## 🎯 Conclusion
The model successfully captures trends in energy consumption and provides a basic forecasting solution.

---

## 🚀 Future Scope
- Use advanced models (Random Forest, LSTM)
- Add external factors (weather, holidays)
- Deploy as web application
