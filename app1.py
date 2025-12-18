# ====================================================================
# Professional HR Attrition Decision Support System
# Developed by: Nisarg Rathod (VNIT Nagpur)
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    page_title="AI-Powered HR Retention System",
    page_icon="🏢",
    layout="wide"
)

# ====================================================================
# LOAD & TRAIN MODEL (CALIBRATED)
# ====================================================================
@st.cache_data
def load_model():
    df = pd.read_csv("HR_comma_sep.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop("left", axis=1)
    y = df["left"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = lgb.LGBMClassifier(
        n_estimators=1888,
        learning_rate=0.019,
        num_leaves=22,
        max_depth=11,
        random_state=42
    )

    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    base_pipeline.fit(X, y)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="isotonic",
        cv=5
    )
    calibrated_model.fit(X, y)

    return calibrated_model, df, base_pipeline, X

pipeline, df, base_pipeline, X_train = load_model()

# ====================================================================
# BUSINESS LOGIC
# ====================================================================
def risk_band(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    return "High Risk"

def hr_alerts(emp):
    alerts = []
    if emp["satisfaction_level"] < 0.4:
        alerts.append("Low job satisfaction detected")
    if emp["last_evaluation"] < 0.5:
        alerts.append("Recent performance evaluation is low")
    if emp["number_project"] >= 6:
        alerts.append("High workload – burnout risk")
    if emp["time_spend_company"] >= 4 and emp["promotion_last_5years"] == 0:
        alerts.append("Long tenure without promotion")
    return alerts

# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
st.sidebar.title("🏢 HR Analytics System")
st.sidebar.markdown("**AI-powered employee retention platform**")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Dashboard", "🎯 Individual Risk Assessment", "🔍 Explainable AI Insights"]
)

# ====================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ====================================================================
if page == "📊 Executive Dashboard":
    st.title("📊 Executive HR Dashboard")

    total_employees = len(df)
    attrition_rate = df["left"].mean() * 100
    avg_satisfaction = df["satisfaction_level"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", total_employees)
    c2.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
    c3.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")

    st.subheader("Attrition Distribution by Department")
    fig = px.bar(df, x="Department", color="left", barmode="group", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insight")
    st.info(
        "Departments with lower satisfaction and high workload show higher attrition risk. "
        "This dashboard enables proactive workforce planning."
    )

# ====================================================================
# PAGE 2: INDIVIDUAL RISK ASSESSMENT
# ====================================================================
if page == "🎯 Individual Risk Assessment":
    st.title("🎯 Individual Employee Attrition Risk Assessment")

    with st.form("employee_form"):
        c1, c2 = st.columns(2)

        with c1:
            satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
            evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7)
            projects = st.slider("Number of Projects", 2, 7, 4)
            hours = st.slider("Avg Monthly Hours", 90, 310, 200)
            years = st.slider("Years at Company", 2, 10, 3)

        with c2:
            accident = st.selectbox("Work Accident", ["No", "Yes"])
            promo = st.selectbox("Promotion in Last 5 Years", ["No", "Yes"])
            dept = st.selectbox("Department", df["Department"].unique())
            salary = st.selectbox("Salary", df["salary"].unique())

        submit = st.form_submit_button("Analyze Employee Risk")

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

        with st.spinner("Running AI risk analysis..."):
            sleep(1)
            prob = pipeline.predict_proba(emp)[0][1]

        decision = "LEAVE" if prob >= 0.4 else "STAY"

        st.subheader(f"Prediction: {decision}")
        st.subheader(f"Attrition Risk Level: {risk_band(prob)}")

        alerts = hr_alerts(emp.iloc[0])
        if alerts:
            st.warning("⚠️ HR Risk Signals")
            for a in alerts:
                st.write("•", a)

        st.success(
            "This assessment combines machine learning predictions with human-centered HR rules "
            "to ensure reliable and actionable decisions."
        )

# ====================================================================
# PAGE 3: EXPLAINABLE AI INSIGHTS
# ====================================================================
if page == "🔍 Explainable AI Insights":
    st.title("🔍 Explainable AI – Model Transparency")

    st.write(
        "This section explains **why** the AI model predicts attrition, "
        "ensuring transparency and trust for HR decision-makers."
    )

    model = base_pipeline.named_steps["classifier"]
    preprocessor = base_pipeline.named_steps["preprocessor"]

    X_proc = preprocessor.transform(X_train)
    if issparse(X_proc):
        X_proc = X_proc.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

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
        "Explainable AI ensures the system does not behave like a black box. "
        "HR managers can clearly see how satisfaction, workload, and tenure "
        "drive attrition risk."
    )
