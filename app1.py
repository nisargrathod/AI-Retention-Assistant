# ====================================================================
# ENTERPRISE HR ATTRITION ANALYTICS PLATFORM
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
from sklearn.metrics import confusion_matrix
from scipy.sparse import issparse
from lifelines import KaplanMeierFitter

import warnings
warnings.filterwarnings("ignore")

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="Enterprise HR Attrition Analytics",
    page_icon="🏢",
    layout="wide"
)

# ====================================================================
# CORPORATE UI STYLING
# ====================================================================
st.markdown("""
<style>
body { background-color: #ECF0F1; }
h1, h2, h3 { color: #0B1C2D; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# LOAD & TRAIN CALIBRATED MODEL
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

    return calibrated, pipe, df, X, y

pipeline, base_pipe, df, X_train, y_train = load_model()

# ====================================================================
# BUSINESS LOGIC
# ====================================================================
def risk_band(p):
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    return "High"

def hr_recommendations(emp):
    recs = []
    if emp["satisfaction_level"] < 0.4:
        recs.append("Conduct engagement discussion with employee")
    if emp["number_project"] >= 6:
        recs.append("Rebalance workload to reduce burnout")
    if emp["time_spend_company"] >= 4 and emp["promotion_last_5years"] == 0:
        recs.append("Discuss career growth and promotion roadmap")
    if not recs:
        recs.append("Continue regular engagement monitoring")
    return recs

# ====================================================================
# SIDEBAR MENU (ENTERPRISE NAVIGATION)
# ====================================================================
st.sidebar.title("🏢 HR Analytics Platform")
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 BI Dashboard",
        "🎯 Prediction",
        "💡 Recommendation",
        "🔍 Explainability",
        "⚖️ Fairness & Bias Audit",
        "📈 Survival Analysis"
    ]
)

# ====================================================================
# HOME
# ====================================================================
if menu == "🏠 Home":
    st.title("Enterprise HR Attrition Analytics")
    st.write("""
    This platform enables **data-driven HR decision-making** by combining:
    - Machine Learning prediction
    - Explainable AI
    - Fairness auditing
    - Survival analysis
    """)

# ====================================================================
# BI DASHBOARD (TABLEAU / POWER BI STYLE)
# ====================================================================
if menu == "📊 BI Dashboard":
    st.title("📊 Executive HR Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", len(df))
    c2.metric("Attrition Rate (%)", f"{df['left'].mean()*100:.2f}")
    c3.metric("Avg Satisfaction", f"{df['satisfaction_level'].mean():.2f}")

    fig = px.bar(
        df,
        x="Department",
        color="left",
        barmode="group",
        template="plotly_white",
        title="Attrition by Department"
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# PREDICTION
# ====================================================================
if menu == "🎯 Prediction":
    st.title("🎯 Employee Attrition Prediction")

    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
            evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
            projects = st.slider("Number of Projects", 2, 7, 4)
            years = st.slider("Years at Company", 1, 10, 3)
        with c2:
            hours = st.slider("Monthly Hours", 90, 310, 200)
            promo = st.selectbox("Promotion in 5 Years", ["No", "Yes"])
            dept = st.selectbox("Department", df["Department"].unique())
            salary = st.selectbox("Salary", df["salary"].unique())
        submit = st.form_submit_button("Predict Risk")

    if submit:
        emp = pd.DataFrame([{
            "satisfaction_level": satisfaction,
            "last_evaluation": evaluation,
            "number_project": projects,
            "average_montly_hours": hours,
            "time_spend_company": years,
            "Work_accident": 0,
            "promotion_last_5years": 1 if promo == "Yes" else 0,
            "Department": dept,
            "salary": salary
        }])

        prob = pipeline.predict_proba(emp)[0][1]
        risk = risk_band(prob)

        st.metric("Attrition Probability", f"{prob*100:.1f}%")
        st.metric("Risk Category", risk)

# ====================================================================
# RECOMMENDATION
# ====================================================================
if menu == "💡 Recommendation":
    st.title("💡 HR Recommendations")

    sample_emp = df.sample(1).iloc[0]
    recs = hr_recommendations(sample_emp)

    for r in recs:
        st.success(r)

# ====================================================================
# EXPLAINABILITY (SHAP)
# ====================================================================
if menu == "🔍 Explainability":
    st.title("🔍 Explainable AI Insights")

    model = base_pipe.named_steps["classifier"]
    preprocessor = base_pipe.named_steps["preprocessor"]

    Xp = preprocessor.transform(X_train)
    if issparse(Xp):
        Xp = Xp.toarray()
    Xp_df = pd.DataFrame(Xp, columns=preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xp_df)

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, Xp_df, plot_type="bar", show=False)
    st.pyplot(fig1)

# ====================================================================
# FAIRNESS & BIAS AUDIT
# ====================================================================
if menu == "⚖️ Fairness & Bias Audit":
    st.title("⚖️ Fairness & Bias Audit")

    preds = pipeline.predict(X_train)
    cm = confusion_matrix(y_train, preds)

    st.write("Confusion Matrix (Overall)")
    st.write(cm)

    st.info(
        "Future extension: demographic parity, equal opportunity, "
        "and subgroup fairness metrics."
    )

# ====================================================================
# SURVIVAL ANALYSIS (REAL CURVES)
# ====================================================================
if menu == "📈 Survival Analysis":
    st.title("📈 Employee Survival Analysis")

    kmf = KaplanMeierFitter()

    durations = df["time_spend_company"]
    event = df["left"]

    kmf.fit(durations, event)
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    st.pyplot(fig)

    st.info(
        "Survival curves estimate the probability that an employee "
        "remains with the organization over time."
    )
