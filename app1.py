# ====================================================================
# AI-Powered HR Retention & Decision Support System
# VNIT Nagpur | Nisarg Rathod
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import issparse
from time import sleep
import warnings

warnings.filterwarnings("ignore")

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="Enterprise HR Retention AI",
    page_icon="🏢",
    layout="wide"
)

# ====================================================================
# MODEL LOADING & CALIBRATION
# ====================================================================
@st.cache_data
def load_model():
    df = pd.read_csv("HR_comma_sep.csv").drop_duplicates().reset_index(drop=True)

    X = df.drop("left", axis=1)
    y = df["left"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = lgb.LGBMClassifier(
        n_estimators=1800,
        learning_rate=0.02,
        num_leaves=24,
        max_depth=11,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipe.fit(X, y)

    calibrated = CalibratedClassifierCV(
        estimator=pipe,
        method="isotonic",
        cv=5
    )
    calibrated.fit(X, y)

    return calibrated, df, pipe, X

pipeline, df, base_pipe, X_train = load_model()

# ====================================================================
# BUSINESS INTELLIGENCE LOGIC
# ====================================================================
def risk_band(p):
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    return "High"

def estimate_retention_cost(salary_level, risk):
    base_cost = {"low": 3000, "medium": 8000, "high": 15000}
    multiplier = {"Low": 0.5, "Medium": 1.0, "High": 1.5}
    return int(base_cost[salary_level] * multiplier[risk])

def estimate_time_to_leave(prob, years, satisfaction):
    """
    Survival-style estimation (months)
    """
    base = 36
    risk_factor = prob * 24
    tenure_factor = max(0, (years - 3) * 3)
    satisfaction_factor = (0.6 - satisfaction) * 20 if satisfaction < 0.6 else 0
    months = max(3, int(base - risk_factor - tenure_factor - satisfaction_factor))
    return months

def hr_alerts(emp):
    alerts = []
    if emp["satisfaction_level"] < 0.4:
        alerts.append("Low job satisfaction")
    if emp["last_evaluation"] < 0.5:
        alerts.append("Low recent performance")
    if emp["number_project"] >= 6:
        alerts.append("Burnout risk (high workload)")
    if emp["time_spend_company"] >= 4 and emp["promotion_last_5years"] == 0:
        alerts.append("Stagnation risk (no promotion)")
    return alerts

# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
st.sidebar.title("🏢 Enterprise HR AI")
st.sidebar.markdown("**AI-Driven Retention & Workforce Planning**")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Dashboard",
        "🎯 Employee Risk Assessment",
        "📈 Retention Cost & Survival",
        "🔍 Explainable AI (SHAP)"
    ]
)

# ====================================================================
# PAGE 1 – EXECUTIVE DASHBOARD
# ====================================================================
if page == "📊 Executive Dashboard":
    st.title("📊 Executive HR Dashboard")

    attrition_rate = df["left"].mean() * 100
    avg_satisfaction = df["satisfaction_level"].mean()
    high_risk_count = int(attrition_rate / 100 * len(df))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", len(df))
    c2.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
    c3.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")

    fig = px.bar(
        df,
        x="Department",
        color="left",
        barmode="group",
        title="Attrition by Department",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This dashboard enables HR leadership to proactively identify "
        "high-risk departments and prioritize retention investments."
    )

# ====================================================================
# PAGE 2 – INDIVIDUAL RISK ASSESSMENT
# ====================================================================
if page == "🎯 Employee Risk Assessment":
    st.title("🎯 Individual Employee Attrition Risk")

    with st.form("risk_form"):
        c1, c2 = st.columns(2)

        with c1:
            satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
            evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
            projects = st.slider("Number of Projects", 2, 7, 4)
            hours = st.slider("Monthly Hours", 90, 310, 200)
            years = st.slider("Years at Company", 1, 10, 3)

        with c2:
            accident = st.selectbox("Work Accident", ["No", "Yes"])
            promo = st.selectbox("Promotion in 5 Years", ["No", "Yes"])
            dept = st.selectbox("Department", df["Department"].unique())
            salary = st.selectbox("Salary Level", df["salary"].unique())

        submit = st.form_submit_button("Analyze Risk")

    if submit:
        emp = pd.DataFrame([{
            "satisfaction_level": satisfaction,
            "last_evaluation": evaluation,
            "number_project": projects,
            "average_montly_hours": hours,
            "time_spend_company": years,
            "Work_accident": 1 if accident == "Yes" else 0,
            "promotion_last_5years": 1 if promo == "Yes" else 0,
            "Department": dept,
            "salary": salary
        }])

        prob = pipeline.predict_proba(emp)[0][1]
        risk = risk_band(prob)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Attrition Risk (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "red" if risk == "High" else "orange"}}
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.subheader(f"Risk Level: {risk}")

        alerts = hr_alerts(emp.iloc[0])
        if alerts:
            st.warning("⚠️ HR Risk Indicators")
            for a in alerts:
                st.write("•", a)

# ====================================================================
# PAGE 3 – RETENTION COST & SURVIVAL ANALYSIS
# ====================================================================
if page == "📈 Retention Cost & Survival":
    st.title("📈 Retention Cost & Time-to-Exit Estimation")

    with st.expander("Why this matters"):
        st.write(
            "This module estimates **financial loss** and **expected time to exit**, "
            "allowing HR to prioritize interventions based on urgency and cost impact."
        )

    satisfaction = st.slider("Employee Satisfaction", 0.0, 1.0, 0.4)
    years = st.slider("Years at Company", 1, 10, 4)
    prob = st.slider("Predicted Attrition Probability", 0.0, 1.0, 0.6)
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    risk = risk_band(prob)
    cost = estimate_retention_cost(salary, risk)
    months = estimate_time_to_leave(prob, years, satisfaction)

    c1, c2 = st.columns(2)
    c1.metric("Estimated Retention Cost ($)", cost)
    c2.metric("Estimated Time to Exit (Months)", months)

    st.success(
        "HR Recommendation: Initiate retention actions **before "
        f"{months} months** to minimize replacement cost."
    )

# ====================================================================
# PAGE 4 – EXPLAINABLE AI (SHAP)
# ====================================================================
if page == "🔍 Explainable AI (SHAP)":
    st.title("🔍 Explainable AI – Model Transparency")

    model = base_pipe.named_steps["classifier"]
    preprocessor = base_pipe.named_steps["preprocessor"]

    X_proc = preprocessor.transform(X_train)
    if issparse(X_proc):
        X_proc = X_proc.toarray()

    X_proc_df = pd.DataFrame(X_proc, columns=preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc_df)

    st.subheader("Global Feature Importance")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_proc_df, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.subheader("Feature Impact Distribution")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_proc_df, show=False)
    st.pyplot(fig2)

    st.info(
        "Explainable AI ensures transparency, fairness, and trust — "
        "critical for HR decision-making."
    )
