# ====================================================================
# All Necessary Imports
# ====================================================================
import streamlit as st
from streamlit_option_menu import option_menu
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
import warnings
from time import sleep
from scipy.sparse import issparse

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={"font": {"size": title_size, "family": "tahoma"}},
        hoverlabel={"font_size": hover_font_size}
    )

def box_plot(the_df, column):
    fig = px.box(
        data_frame=the_df,
        x=column,
        template="plotly_dark",
        height=600
    )
    custome_layout(fig)
    return fig

def bar_plot(the_df, column):
    fig = px.bar(
        the_df[column].value_counts(),
        template="plotly_dark",
        height=600
    )
    custome_layout(fig)
    return fig

def pie_chart(the_df, column):
    fig = px.pie(
        the_df,
        names=column,
        template="plotly_dark",
        height=600
    )
    custome_layout(fig, showlegend=True)
    return fig

def create_heat_map(the_df):
    corr = the_df.select_dtypes(include=np.number).corr()
    fig = px.imshow(
        corr,
        text_auto="0.2f",
        aspect=1,
        color_continuous_scale="greens",
        template="plotly_dark",
        height=650
    )
    return fig

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(
        page_title="Employee Retention AI",
        page_icon="🤖",
        layout="wide"
    )
    warnings.filterwarnings("ignore")

    # ================================================================
    # Load Data + Train + CALIBRATE MODEL (FIXED)
    # ================================================================
    @st.cache_data
    def load_data_and_train_model():
        df = pd.read_csv("HR_comma_sep.csv")
        df_original = df.copy()

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

        # ✅ Correct for sklearn >=1.3
        calibrated_pipeline = CalibratedClassifierCV(
            estimator=base_pipeline,
            method="isotonic",
            cv=5
        )

        calibrated_pipeline.fit(X, y)

        return calibrated_pipeline, df_original

    pipeline, df = load_data_and_train_model()

    # ================================================================
    # Risk Band Logic
    # ================================================================
    def get_risk_band(prob):
        if prob < 0.30:
            return "Low Risk"
        elif prob < 0.60:
            return "Medium Risk"
        return "High Risk"

    # ================================================================
    # HR Warning Signals (Override Layer)
    # ================================================================
    def hr_warnings(emp):
        alerts = []
        if emp["satisfaction_level"] < 0.4:
            alerts.append("Low satisfaction detected")
        if emp["last_evaluation"] < 0.5:
            alerts.append("Low performance evaluation")
        if emp["number_project"] >= 6:
            alerts.append("High workload – burnout risk")
        if emp["time_spend_company"] >= 4 and emp["promotion_last_5years"] == 0:
            alerts.append("Long tenure without promotion")
        return alerts

    # ================================================================
    # Sidebar
    # ================================================================
    with st.sidebar:
        st.title("🤖 AI Retention Assistant")
        st.write("**Developed by Nisarg Rathod**")
        page = option_menu(
            None,
            ["Home", "Vizualizations", "Prediction"],
            icons=["house", "bar-chart", "activity"],
            default_index=0
        )

    # ================================================================
    # Pages
    # ================================================================
    if page == "Home":
        st.header("Employee Retention Dataset")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("Dataset Summary")
        st.table(df.describe().T)

    if page == "Vizualizations":
        st.header("Data Visualizations")
        st.subheader("Correlation Heatmap")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    if page == "Prediction":
        st.header("🎯 Predict Attrition & HR Decision Support")

        with st.form("predict_form"):
            sat_map = {
                "Very Dissatisfied": 0.1,
                "Dissatisfied": 0.3,
                "Neutral": 0.5,
                "Satisfied": 0.7,
                "Very Satisfied": 0.9
            }
            eval_map = {
                "Needs Improvement": 0.4,
                "Meets Expectations": 0.7,
                "Exceeds Expectations": 0.9
            }

            c1, c2 = st.columns(2)

            with c1:
                satisfaction = sat_map[st.select_slider("Satisfaction Level", sat_map.keys())]
                evaluation = eval_map[st.select_slider("Last Evaluation", eval_map.keys())]
                projects = st.slider("Number of Projects", 2, 7, 4)
                hours = st.slider("Avg Monthly Hours", 90, 310, 200)
                years = st.slider("Years at Company", 2, 10, 3)

            with c2:
                accident = 1 if st.selectbox("Work Accident", ["No", "Yes"]) == "Yes" else 0
                promo = 1 if st.selectbox("Promotion in 5 Years", ["No", "Yes"]) == "Yes" else 0
                dept = st.selectbox("Department", df["Department"].unique())
                salary = st.selectbox("Salary", df["salary"].unique())

            submit = st.form_submit_button("Get Prediction")

        if submit:
            input_df = pd.DataFrame([{
                "satisfaction_level": satisfaction,
                "last_evaluation": evaluation,
                "number_project": projects,
                "average_montly_hours": hours,
                "time_spend_company": years,
                "Work_accident": accident,
                "promotion_last_5years": promo,
                "Department": dept,
                "salary": salary
            }])

            with st.spinner("Analyzing employee profile..."):
                sleep(1)
                probs = pipeline.predict_proba(input_df)[0]
                leave_prob = probs[1]

                # HR-sensitive threshold
                decision = "LEAVE" if leave_prob >= 0.4 else "STAY"

            st.subheader(f"Prediction: {decision}")
            st.subheader(f"Attrition Risk Level: {get_risk_band(leave_prob)}")

            alerts = hr_warnings(input_df.iloc[0])
            if alerts:
                st.warning("⚠️ HR Risk Signals Detected")
                for a in alerts:
                    st.write("•", a)

            if decision == "LEAVE":
                st.info("💡 HR Action Recommended: Immediate retention strategy advised.")

# ====================================================================
if __name__ == "__main__":
    main()
