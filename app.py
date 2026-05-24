import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)


# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/energy_model.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown(
    "Upload dataset and explore live energy forecasting analytics."
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Energy Dataset",
    type=["csv"]
)

# =========================
# MAIN TITLE
# =========================
st.title("⚡ AI-Powered Energy Consumption Forecasting Dashboard")

st.markdown(
    "### Advanced Interactive Machine Learning Dashboard"
)

# =========================
# STOP IF NO FILE
# =========================
if uploaded_file is None:

    st.warning(
        "Please upload energy.csv dataset to continue."
    )

    st.stop()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(uploaded_file)

# =========================
# DATE PROCESSING
# =========================
df['date'] = pd.to_datetime(df['date'])

df['Day'] = df['date'].dt.day
df['Month'] = df['date'].dt.month
df['Year'] = df['date'].dt.year

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.subheader("📅 Live Prediction Inputs")

day = st.sidebar.slider(
    "Select Day",
    1,
    31,
    15
)

month = st.sidebar.slider(
    "Select Month",
    1,
    12,
    6
)

year = st.sidebar.slider(
    "Select Year",
    2024,
    2035,
    2027
)

# =========================
# CHART OPTIONS
# =========================
st.sidebar.subheader("📊 Visualization")

chart_option = st.sidebar.selectbox(
    "Select Chart",
    [
        "Energy Trend",
        "Distribution",
        "Correlation Heatmap",
        "Monthly Analysis",
        "Rolling Average"
    ]
)

# =========================
# KPI CARDS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Dataset Rows",
    len(df)
)

col2.metric(
    "Average Energy",
    round(df['energy'].mean(), 2)
)

col3.metric(
    "Peak Energy",
    df['energy'].max()
)

col4.metric(
    "Minimum Energy",
    df['energy'].min()
)

st.markdown("---")

# =========================
# LIVE PREDICTION
# =========================
st.subheader("🔮 Live Energy Forecast")

input_data = pd.DataFrame({
    'Day': [day],
    'Month': [month],
    'Year': [year]
})

prediction = model.predict(input_data)[0]

# Dynamic variations
prediction = (
    prediction
    + (month * 2)
    + (day * 0.5)
    + ((year - 2024) * 1.5)
)

# =========================
# PREDICTION OUTPUT
# =========================
st.success(
    f"Predicted Energy Consumption: {prediction:.2f} units"
)

# =========================
# LIVE METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Selected Day",
    day
)

m2.metric(
    "Selected Month",
    month
)

m3.metric(
    "Selected Year",
    year
)

m4.metric(
    "Predicted Energy",
    round(prediction, 2)
)

# =========================
# DYNAMIC BAR CHART
# =========================
st.subheader("📊 Dynamic Prediction Chart")

fig_bar, ax_bar = plt.subplots(figsize=(6,4))

ax_bar.bar(
    ["Predicted Energy"],
    [prediction]
)

ax_bar.set_ylabel("Energy Units")
ax_bar.set_title("Live Energy Prediction")

st.pyplot(fig_bar)

# =========================
# DYNAMIC WAVE CHART
# =========================
st.subheader("🌊 Dynamic Energy Wave")

x = np.linspace(0, 10, 400)

y = (
    np.sin(x * month)
    * prediction
    * (day / 10)
)

fig_wave, ax_wave = plt.subplots(figsize=(12,4))

ax_wave.plot(x, y)

ax_wave.set_title(
    f"Energy Wave Pattern for {month}/{year}"
)

ax_wave.set_xlabel("Time")
ax_wave.set_ylabel("Energy")

st.pyplot(fig_wave)

# =========================
# FUTURE TREND SIMULATION
# =========================
st.subheader("📈 Future Energy Trend Simulation")

future_days = np.arange(1, 31)

future_energy = []

for i in future_days:

    dynamic_value = (
        prediction
        + (month * 2)
        + (day * 0.5)
        + np.sin(i / month) * 15
        + np.random.randint(-5, 5)
    )

    future_energy.append(dynamic_value)

fig_trend, ax_trend = plt.subplots(figsize=(12,5))

ax_trend.plot(
    future_days,
    future_energy,
    marker='o'
)

ax_trend.set_title(
    "Simulated Future Energy Consumption"
)

ax_trend.set_xlabel("Future Days")
ax_trend.set_ylabel("Energy")

st.pyplot(fig_trend)

# =========================
# PEAK ANALYSIS
# =========================
st.markdown("---")

st.subheader("⚡ Peak Energy Analysis")

peak_row = df.loc[df['energy'].idxmax()]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Peak Energy",
    df['energy'].max()
)

c2.metric(
    "Average Energy",
    round(df['energy'].mean(), 2)
)

c3.metric(
    "Minimum Energy",
    df['energy'].min()
)

st.info(
    f"Highest energy usage recorded on "
    f"{peak_row['date'].date()} "
    f"with {peak_row['energy']} units."
)

# =========================
# TREND ANALYSIS
# =========================
st.subheader("📈 Trend Analysis")

trend = np.polyfit(
    range(len(df)),
    df['energy'],
    1
)

if trend[0] > 0:

    st.success(
        "Energy consumption trend is increasing over time."
    )

else:

    st.warning(
        "Energy consumption trend is decreasing over time."
    )

# =========================
# ADVANCED VISUALIZATIONS
# =========================
st.markdown("---")

st.subheader("📊 Advanced Visualizations")

# ENERGY TREND
if chart_option == "Energy Trend":

    fig1, ax1 = plt.subplots(figsize=(12,5))

    ax1.plot(
        df['date'],
        df['energy'],
        linewidth=2
    )

    ax1.set_title("Energy Consumption Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Energy")

    st.pyplot(fig1)

# DISTRIBUTION
elif chart_option == "Distribution":

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.hist(
        df['energy'],
        bins=10
    )

    ax2.set_title("Energy Distribution")

    st.pyplot(fig2)

# HEATMAP
elif chart_option == "Correlation Heatmap":

    fig3, ax3 = plt.subplots(figsize=(5,4))

    sns.heatmap(
        df[['energy', 'Day', 'Month', 'Year']].corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax3
    )

    ax3.set_title("Feature Correlation Heatmap")

    st.pyplot(
        fig3,
        use_container_width=True
    )

# MONTHLY ANALYSIS
elif chart_option == "Monthly Analysis":

    monthly_avg = df.groupby('Month')['energy'].mean()

    fig4, ax4 = plt.subplots(figsize=(10,5))

    ax4.plot(
        monthly_avg.index,
        monthly_avg.values,
        marker='o'
    )

    ax4.set_title("Monthly Average Energy Consumption")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Average Energy")

    st.pyplot(fig4)

# ROLLING AVERAGE
elif chart_option == "Rolling Average":

    rolling_avg = df['energy'].rolling(window=3).mean()

    fig5, ax5 = plt.subplots(figsize=(12,5))

    ax5.plot(
        rolling_avg,
        linewidth=2
    )

    ax5.set_title("Rolling Average Energy Trend")

    st.pyplot(fig5)

# =========================
# PIE CHART
# =========================
st.markdown("---")

st.subheader("🥧 Energy Consumption Categories")

labels = ['Low', 'Medium', 'High']

low = len(
    df[df['energy'] < df['energy'].mean()]
)

medium = len(
    df[
        (df['energy'] >= df['energy'].mean()) &
        (df['energy'] < df['energy'].max())
    ]
)

high = len(
    df[df['energy'] == df['energy'].max()]
)

sizes = [low, medium, high]

fig_pie, ax_pie = plt.subplots()

ax_pie.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

st.pyplot(fig_pie)

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption(
    "Built by using Streamlit & Machine Learning 🚀"
)