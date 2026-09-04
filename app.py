import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Page Configuration
st.set_page_config(
    page_title="OptimaSpend: Polynomial Regression Sales Predictor",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_and_train_model():
  # Generating synthetic Advertising Dataset for demo robustness if file is missing,
  # or loading from standard URL.
  np.random.seed(42)
  n_samples = 200
  tv = np.random.uniform(10, 300, n_samples)
  radio = np.random.uniform(5, 50, n_samples)
  newspaper = np.random.uniform(1, 100, n_samples)

  # Polynomial relationship with diminishing returns & noise
  sales = (
      3.0
      + 0.05 * tv
      + 0.1 * radio
      + 0.01 * newspaper
      - 0.0001 * (tv**2)
      + np.random.normal(0, 1.5, n_samples)
  )
  sales = np.maximum(sales, 2)  # ensure positive sales

  df = pd.DataFrame(
      {"TV": tv, "Radio": radio, "Newspaper": newspaper, "Sales": sales}
  )

  X = df[["TV", "Radio", "Newspaper"]]
  y = df["Sales"]

  # Train Polynomial Regression (Degree 2)
  poly = PolynomialFeatures(degree=2, include_bias=False)
  X_poly = poly.fit_transform(X)

  model = LinearRegression()
  model.fit(X_poly, y)

  return model, poly, df


model, poly, df = load_and_train_model()

# App Header
st.markdown(
    '<p class="main-header">OptimaSpend: Ad-Spend & Sales Predictor</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Leverage Polynomial Regression to forecast sales'
    ' and avoid diminishing returns.</p>',
    unsafe_allow_html=True,
)
st.divider()

# Sidebar Inputs
st.sidebar.header("🎛️ Budget Allocation Controls")
tv_budget = st.sidebar.slider("TV Advertising Budget ($)", 0.0, 300.0, 150.0, 5.0)
radio_budget = st.sidebar.slider(
    "Radio Advertising Budget ($)", 0.0, 50.0, 25.0, 1.0
)
newspaper_budget = st.sidebar.slider(
    "Newspaper Advertising Budget ($)", 0.0, 100.0, 40.0, 1.0
)

# Prediction Calculation
input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
input_poly = poly.transform(input_data)
predicted_sales = model.predict(input_poly)[0]

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("📊 Instant Sales Prediction")
  st.metric(
      label="Estimated Product Sales (Units)",
      value=f"{max(0, predicted_sales):.2f}k",
  )

  total_budget = tv_budget + radio_budget + newspaper_budget
  st.metric(label="Total Ad Spend", value=f"${total_budget:,.2f}")

  if predicted_sales > 20:
    st.success(
        "🚀 High efficiency zone! Your current allocation yields strong"
        " predicted returns."
    )
  else:
    st.warning(
        "⚠️ Diminishing returns warning: Higher spending levels in certain"
        " channels may be underperforming."
    )

with col2:
  st.subheader("📈 Dataset Overview & Relationship")
  st.line_chart(df[["Sales"]].head(50))

# Extra feature: Budget breakdown chart
st.divider()
st.subheader("💡 Multi-Channel Budget Distribution")
budget_df = pd.DataFrame({
    "Channel": ["TV", "Radio", "Newspaper"],
    "Budget ($)": [tv_budget, radio_budget, newspaper_budget],
})
st.bar_chart(budget_df.set_index("Channel"))