from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OptimaSpend | AI Sales Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BEAUTIFUL UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 5%, rgba(124,58,237,.22), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(6,182,212,.16), transparent 26%),
        radial-gradient(circle at 60% 100%, rgba(59,130,246,.10), transparent 30%),
        linear-gradient(135deg, #050816 0%, #0b1026 50%, #080b18 100%);
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #060b20 0%, #0a1230 55%, #080b1b 100%);
    border-right: 1px solid rgba(139,92,246,.28);
}

.sidebar-logo {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-sub {
    color: #94a3b8;
    font-size: 12px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 8px;
}

.metric-card {
    min-height: 142px;
    padding: 21px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 12px 38px rgba(0,0,0,.28);
    transition: .25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 48px rgba(99,102,241,.24);
}

.purple {
    background: linear-gradient(135deg, rgba(124,58,237,.40), rgba(49,46,129,.22));
}

.blue {
    background: linear-gradient(135deg, rgba(37,99,235,.40), rgba(30,64,175,.20));
}

.green {
    background: linear-gradient(135deg, rgba(13,148,136,.40), rgba(6,78,59,.20));
}

.orange {
    background: linear-gradient(135deg, rgba(245,158,11,.34), rgba(120,53,15,.20));
}

.pink {
    background: linear-gradient(135deg, rgba(219,39,119,.34), rgba(126,34,206,.18));
}

.metric-label {
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 9px;
}

.metric-note {
    color: #34d399;
    font-size: 12px;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin: 24px 0 13px 0;
}

.panel {
    background: linear-gradient(145deg, rgba(15,23,42,.90), rgba(15,23,42,.58));
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,.22);
}

.warning {
    padding: 17px 20px;
    border-radius: 15px;
    border: 1px solid rgba(245,158,11,.55);
    background: linear-gradient(135deg, rgba(245,158,11,.16), rgba(120,53,15,.16));
    color: #fde68a;
}

.insight {
    padding: 15px 17px;
    border-radius: 13px;
    margin-bottom: 10px;
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(255,255,255,.08);
}

.insight-green { border-left: 4px solid #10b981; }
.insight-blue { border-left: 4px solid #38bdf8; }
.insight-purple { border-left: 4px solid #a78bfa; }
.insight-orange { border-left: 4px solid #f59e0b; }

.stButton > button,
.stDownloadButton > button {
    border: 0;
    border-radius: 12px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    font-weight: 700;
    min-height: 42px;
    box-shadow: 0 8px 25px rgba(99,102,241,.20);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 28px rgba(139,92,246,.45);
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
}

div[data-testid="stMetric"] {
    background: rgba(15,23,42,.70);
    border: 1px solid rgba(148,163,184,.13);
    padding: 12px;
    border-radius: 14px;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD THE REAL DATASET
# Supports:
#   advertising.csv
#   archive (3).zip containing advertising.csv
#   data (1)(1).csv
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        Path("advertising.csv"),
        Path("data (1)(1).csv"),
        Path("data (1).csv"),
        Path("archive (3).csv"),
    ]

    for file_path in possible_files:
        if file_path.exists():
            return pd.read_csv(file_path), str(file_path)


    return None, None


df, source_name = load_data()

if df is None:
    st.error(
        "Dataset not found. Put advertising.csv in the same folder as app.py."
    )
    st.stop()


# ============================================================
# CLEAN / VALIDATE DATA
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

required_columns = ["tv", "radio", "newspaper", "sales"]

if not all(col in df.columns for col in required_columns):
    st.error(
        "The dataset must contain these columns: "
        "TV, Radio, Newspaper, Sales"
    )
    st.write("Detected columns:", list(df.columns))
    st.stop()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns).reset_index(drop=True)


# ============================================================
# TRAIN POLYNOMIAL REGRESSION
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[["tv", "radio", "newspaper"]]
    y = data["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = Pipeline([
        ("poly", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),
        ("regression", LinearRegression())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
    }

    return model, X_test, y_test, predictions, metrics


model, X_test, y_test, test_predictions, metrics = train_model(df)

r2 = metrics["r2"]
mae = metrics["mae"]
rmse = metrics["rmse"]


# ============================================================
# FUNCTIONS
# ============================================================

def predict_sales(tv, radio, newspaper):
    values = pd.DataFrame({
        "tv": [tv],
        "radio": [radio],
        "newspaper": [newspaper]
    })
    return float(model.predict(values)[0])


def card(label, value, note, css_class):
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_chart_layout(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=30, r=25, t=65, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">🔮 OptimaSpend</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-sub">AI Ad-Spend & Sales Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🎯 Prediction",
            "💰 Budget Allocation",
            "📈 Analytics",
            "🤖 AI Insights",
            "📋 Dataset"
        ]
    )

    st.markdown("---")

    st.markdown("### ⚡ System Status")
    st.success("Model Online")
    st.success("Dataset Loaded")
    st.info("Polynomial Regression • Degree 2")

    st.markdown("---")

    st.caption(f"Data source: {source_name}")
    st.caption(f"Records: {len(df)}")
    st.caption("OptimaSpend v3.0")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-title">
        OptimaSpend: Ad-Spend & Sales Predictor
    </div>

    <div class="hero-sub">
        Advanced Polynomial Regression dashboard for sales forecasting,
        budget optimization and diminishing-return analysis.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    avg_sales = df["sales"].mean()
    total_spend = (
        df["tv"].sum()
        + df["radio"].sum()
        + df["newspaper"].sum()
    )

    avg_spend = (
        df["tv"].mean()
        + df["radio"].mean()
        + df["newspaper"].mean()
    )

    efficiency = avg_sales / avg_spend if avg_spend else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "📦 Average Sales",
            f"{avg_sales:.2f}k",
            "Dataset average",
            "purple"
        )

    with c2:
        card(
            "💰 Total Ad Spend",
            f"${total_spend:,.2f}",
            "All 3 channels",
            "blue"
        )

    with c3:
        card(
            "📈 Sales / Spend",
            f"{efficiency:.3f}x",
            "Average efficiency",
            "green"
        )

    with c4:
        card(
            "🎯 Model Accuracy",
            f"{max(0, r2)*100:.1f}%",
            "R² validation score",
            "orange"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning">
            ⚠️ <b>Diminishing Returns Monitor</b><br>
            The model includes a polynomial term so that nonlinear behavior
            can be detected. Increasing a channel indefinitely may not produce
            proportional sales growth.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📊 Dataset Overview & Relationships</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        trend = df.copy()
        trend["Data Point"] = np.arange(1, len(trend) + 1)

        fig = px.line(
            trend,
            x="Data Point",
            y="sales",
            markers=True,
            title="Sales Trend"
        )

        fig.update_traces(
            line=dict(color="#8b5cf6", width=3),
            marker=dict(color="#38bdf8", size=5)
        )

        st.plotly_chart(
            make_chart_layout(fig),
            use_container_width=True
        )

    with right:

        budget_data = pd.DataFrame({
            "Channel": ["TV", "Radio", "Newspaper"],
            "Spend": [
                df["tv"].sum(),
                df["radio"].sum(),
                df["newspaper"].sum()
            ]
        })

        fig = px.pie(
            budget_data,
            names="Channel",
            values="Spend",
            hole=0.58,
            title="Advertising Spend Distribution",
            color="Channel",
            color_discrete_map={
                "TV": "#8b5cf6",
                "Radio": "#06b6d4",
                "Newspaper": "#10b981"
            }
        )

        st.plotly_chart(
            make_chart_layout(fig),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">💡 Multi-Channel Budget Distribution</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        budget_data,
        x="Channel",
        y="Spend",
        text="Spend",
        color="Channel",
        color_discrete_map={
            "TV": "#8b5cf6",
            "Radio": "#06b6d4",
            "Newspaper": "#10b981"
        }
    )

    fig.update_traces(
        texttemplate="$%{text:.1f}",
        textposition="outside"
    )

    st.plotly_chart(
        make_chart_layout(fig),
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🎯 Prediction":

    st.markdown(
        '<div class="section-title">🎯 Instant Sales Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Change the advertising budgets below to generate a live sales forecast."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        tv_budget = st.slider(
            "📺 TV Advertising Budget ($)",
            float(df["tv"].min()),
            float(df["tv"].max()),
            float(df["tv"].median()),
            1.0
        )

    with c2:
        radio_budget = st.slider(
            "📻 Radio Advertising Budget ($)",
            float(df["radio"].min()),
            float(df["radio"].max()),
            float(df["radio"].median()),
            1.0
        )

    with c3:
        newspaper_budget = st.slider(
            "📰 Newspaper Advertising Budget ($)",
            float(df["newspaper"].min()),
            float(df["newspaper"].max()),
            float(df["newspaper"].median()),
            1.0
        )

    predicted = predict_sales(
        tv_budget,
        radio_budget,
        newspaper_budget
    )

    total = tv_budget + radio_budget + newspaper_budget
    efficiency = predicted / total if total else 0

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "📦 Estimated Product Sales",
            f"{max(0, predicted):.2f}k",
            "Live polynomial forecast",
            "purple"
        )

    with c2:
        card(
            "💰 Total Ad Spend",
            f"${total:,.2f}",
            "Current scenario",
            "blue"
        )

    with c3:
        card(
            "📈 Predicted Efficiency",
            f"{efficiency:.3f}x",
            "Sales per advertising dollar",
            "green"
        )

    if predicted > df["sales"].median():
        st.success(
            "🚀 This scenario predicts sales above the dataset median."
        )
    else:
        st.warning(
            "⚠️ This scenario predicts sales below the dataset median."
        )

    # Prediction breakdown
    scenario = pd.DataFrame({
        "Channel": ["TV", "Radio", "Newspaper"],
        "Budget": [
            tv_budget,
            radio_budget,
            newspaper_budget
        ]
    })

    fig = px.bar(
        scenario,
        x="Channel",
        y="Budget",
        text="Budget",
        color="Channel",
        color_discrete_map={
            "TV": "#8b5cf6",
            "Radio": "#06b6d4",
            "Newspaper": "#10b981"
        },
        title="Current Prediction Budget"
    )

    fig.update_traces(
        texttemplate="$%{text:.1f}",
        textposition="outside"
    )

    st.plotly_chart(
        make_chart_layout(fig),
        use_container_width=True
    )


# ============================================================
# BUDGET ALLOCATION / SCENARIO PLANNER
# ============================================================

elif page == "💰 Budget Allocation":

    st.markdown(
        '<div class="section-title">🚀 Scenario Planner</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Compare two advertising strategies before changing your real budget."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### 🟣 Scenario A")

        a_tv = st.slider(
            "TV — A",
            float(df["tv"].min()),
            float(df["tv"].max()),
            float(df["tv"].median()),
            1.0,
            key="a_tv"
        )

        a_radio = st.slider(
            "Radio — A",
            float(df["radio"].min()),
            float(df["radio"].max()),
            float(df["radio"].median()),
            1.0,
            key="a_radio"
        )

        a_news = st.slider(
            "Newspaper — A",
            float(df["newspaper"].min()),
            float(df["newspaper"].max()),
            float(df["newspaper"].median()),
            1.0,
            key="a_news"
        )

    with right:

        st.markdown("### 🔵 Scenario B")

        b_tv = st.slider(
            "TV — B",
            float(df["tv"].min()),
            float(df["tv"].max()),
            min(float(df["tv"].median() * 1.2), float(df["tv"].max())),
            1.0,
            key="b_tv"
        )

        b_radio = st.slider(
            "Radio — B",
            float(df["radio"].min()),
            float(df["radio"].max()),
            float(df["radio"].median()),
            1.0,
            key="b_radio"
        )

        b_news = st.slider(
            "Newspaper — B",
            float(df["newspaper"].min()),
            float(df["newspaper"].max()),
            float(df["newspaper"].median()),
            1.0,
            key="b_news"
        )

    a_sales = predict_sales(a_tv, a_radio, a_news)
    b_sales = predict_sales(b_tv, b_radio, b_news)

    a_spend = a_tv + a_radio + a_news
    b_spend = b_tv + b_radio + b_news

    a_eff = a_sales / a_spend if a_spend else 0
    b_eff = b_sales / b_spend if b_spend else 0

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Scenario A Sales",
        f"{a_sales:.2f}"
    )

    c2.metric(
        "Scenario B Sales",
        f"{b_sales:.2f}",
        f"{b_sales-a_sales:+.2f}"
    )

    c3.metric(
        "Scenario A Efficiency",
        f"{a_eff:.3f}x"
    )

    c4.metric(
        "Scenario B Efficiency",
        f"{b_eff:.3f}x",
        f"{b_eff-a_eff:+.3f}"
    )

    comparison = pd.DataFrame({
        "Scenario": ["Scenario A", "Scenario B"],
        "Predicted Sales": [a_sales, b_sales],
        "Total Budget": [a_spend, b_spend]
    })

    fig = px.bar(
        comparison,
        x="Scenario",
        y="Predicted Sales",
        text="Predicted Sales",
        color="Scenario",
        color_discrete_sequence=["#8b5cf6", "#06b6d4"],
        title="Scenario Sales Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        make_chart_layout(fig),
        use_container_width=True
    )

    if b_eff > a_eff:
        st.success(
            "💡 Scenario B currently gives the better predicted efficiency."
        )
    elif a_eff > b_eff:
        st.success(
            "💡 Scenario A currently gives the better predicted efficiency."
        )
    else:
        st.info("Both scenarios have the same predicted efficiency.")


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="section-title">📈 Advanced Model Analytics</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": test_predictions
    })

    fig = px.scatter(
        comparison,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Sales"
    )

    fig.add_shape(
        type="line",
        x0=comparison["Actual"].min(),
        y0=comparison["Actual"].min(),
        x1=comparison["Actual"].max(),
        y1=comparison["Actual"].max(),
        line=dict(
            color="#f59e0b",
            dash="dash"
        )
    )

    st.plotly_chart(
        make_chart_layout(fig),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">🔥 Correlation Heatmap</div>',
        unsafe_allow_html=True
    )

    corr = df[
        ["tv", "radio", "newspaper", "sales"]
    ].corr()

    corr_display = corr.copy()
    corr_display.columns = ["TV", "Radio", "Newspaper", "Sales"]
    corr_display.index = ["TV", "Radio", "Newspaper", "Sales"]

    fig = px.imshow(
        corr_display,
        text_auto=".2f",
        aspect="auto",
        title="Advertising Channels vs Sales"
    )

    st.plotly_chart(
        make_chart_layout(fig, 480),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📊 Channel Relationships</div>',
        unsafe_allow_html=True
    )

    channel_corr = pd.DataFrame({
        "Channel": ["TV", "Radio", "Newspaper"],
        "Correlation": [
            corr["sales"]["tv"],
            corr["sales"]["radio"],
            corr["sales"]["newspaper"]
        ]
    })

    fig = px.bar(
        channel_corr,
        x="Channel",
        y="Correlation",
        text="Correlation",
        color="Correlation",
        color_continuous_scale=["#f59e0b", "#06b6d4", "#8b5cf6"],
        title="Correlation With Sales"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        make_chart_layout(fig),
        use_container_width=True
    )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "🤖 AI Insights":

    st.markdown(
        '<div class="section-title">🤖 AI-Powered Marketing Insights</div>',
        unsafe_allow_html=True
    )

    corr = df[
        ["tv", "radio", "newspaper", "sales"]
    ].corr()["sales"].drop("sales")

    strongest = corr.abs().idxmax()
    weakest = corr.abs().idxmin()

    names = {
        "tv": "TV",
        "radio": "Radio",
        "newspaper": "Newspaper"
    }

    strongest_name = names[strongest]
    weakest_name = names[weakest]

    st.markdown(
        f"""
        <div class="insight insight-green">
            🚀 <b>Strongest Sales Driver</b><br>
            {strongest_name} has the strongest relationship with Sales
            with a correlation of {corr[strongest]:.3f}.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight insight-blue">
            📊 <b>Optimization Opportunity</b><br>
            {weakest_name} has the weakest absolute correlation with Sales
            ({corr[weakest]:.3f}). Test whether reducing or reallocating this
            budget improves overall efficiency.
        </div>
        """,
        unsafe_allow_html=True
    )

    if r2 >= 0.80:
        quality = "Excellent"
    elif r2 >= 0.60:
        quality = "Good"
    elif r2 >= 0.40:
        quality = "Moderate"
    else:
        quality = "Needs Improvement"

    st.markdown(
        f"""
        <div class="insight insight-purple">
            🎯 <b>Model Quality: {quality}</b><br>
            Validation R² = {r2:.2%}. MAE = {mae:.3f}.
            RMSE = {rmse:.3f}.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Model gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, r2 * 100)),
            title={"text": "Model R² Performance"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, 40], "color": "#2b163f"},
                    {"range": [40, 70], "color": "#172554"},
                    {"range": [70, 100], "color": "#064e3b"}
                ]
            }
        )
    )

    st.plotly_chart(
        make_chart_layout(fig, 350),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">🏆 Channel Ranking</div>',
        unsafe_allow_html=True
    )

    ranking = (
        corr.abs()
        .sort_values(ascending=False)
        .reset_index()
    )

    ranking.columns = ["Channel", "Absolute Correlation"]

    ranking["Channel"] = ranking["Channel"].map(names)

    fig = px.bar(
        ranking,
        x="Absolute Correlation",
        y="Channel",
        orientation="h",
        text="Absolute Correlation",
        color="Absolute Correlation",
        color_continuous_scale=["#06b6d4", "#8b5cf6"],
        title="Sales Relationship Strength"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}"
    )

    st.plotly_chart(
        make_chart_layout(fig, 380),
        use_container_width=True
    )

    st.markdown(
        """
        <div class="panel">
            💡 <b>Recommendation Engine</b><br><br>
            Use the Scenario Planner to test multiple allocations rather
            than increasing one channel blindly. The model is a decision
            support tool; real campaign results should be used to validate
            major budget changes.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📋 Dataset":

    st.markdown(
        '<div class="section-title">📋 Advertising Dataset</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Average Sales", f"{df['sales'].mean():.2f}")
    c4.metric("Maximum Sales", f"{df['sales'].max():.2f}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">📊 Descriptive Statistics</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    # Download predictions
    full_predictions = model.predict(
        df[["tv", "radio", "newspaper"]]
    )

    export_df = df.copy()
    export_df["Predicted_Sales"] = full_predictions
    export_df["Prediction_Error"] = (
        export_df["sales"] - export_df["Predicted_Sales"]
    )

    st.download_button(
        "⬇️ Download Dataset + Predictions",
        export_df.to_csv(index=False),
        "optimaspend_predictions.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:18px;">
        🔮 <b>OptimaSpend</b>
        &nbsp;•&nbsp; AI Sales Intelligence
        &nbsp;•&nbsp; Polynomial Regression
        &nbsp;•&nbsp; Scenario Planning
    </div>
    """,
    unsafe_allow_html=True
)
