import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =========================
# Load Model
# =========================
model = joblib.load("models/energy_model.pkl")

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Input Parameters")

st.sidebar.markdown("Enter values to predict future energy consumption.")

# User Inputs

day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.slider("Year", 2024, 2035, 2026)

# =========================
# Main Title
# =========================
st.title("⚡ AI-Powered Energy Consumption Forecasting Dashboard")

st.markdown("### Predict future energy consumption using Machine Learning")

# =========================
# Input Data
# =========================
input_data = pd.DataFrame({
    'Day': [day],
    'Month': [month],
    'Year': [year]
})

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Energy Consumption"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Energy Consumption: {prediction[0]:.2f} units"
    )

    # =========================
    # Metrics
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Day", day)
    col2.metric("Month", month)
    col3.metric("Year", year)

    # =========================
    # Prediction Chart
    # =========================
    chart_data = pd.DataFrame({
        'Category': ['Predicted Energy'],
        'Value': [prediction[0]]
    })

    st.subheader("📊 Prediction Visualization")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(chart_data['Category'], chart_data['Value'])
    ax.set_ylabel("Energy Consumption")

    st.pyplot(fig)

# =========================
# Information Section
# =========================
st.markdown("---")

st.subheader("📌 About Project")

st.write(
    "This dashboard uses Machine Learning to forecast future energy consumption based on historical patterns."
)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built by using Streamlit & Machine Learning 🚀")