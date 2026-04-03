# ====================================================================
# All Necessary Imports
# ====================================================================
import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import warnings
from time import sleep
from scipy.sparse import issparse
from scipy.special import expit, logit
import io
import base64

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel
from scipy.optimize import milp, LinearConstraint, Bounds

# --- Imports for Evaluation 2 (Intelligent Interface: Groq + Evidently) ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Imports for AI Research Lab (Counterfactuals) ---
import dice_ml
from dice_ml import Dice

# ====================================================================
# PROJECT METADATA (For M.Tech Documentation)
# ====================================================================
PROJECT_INFO = {
    "title": "RetainAI: Enterprise Workforce Intelligence System",
    "subtitle": "An AI-Powered Employee Attrition Prediction & Strategic Retention Framework",
    "author": "Nisarg Rathod",
    "version": "2.0",
    "tech_stack": [
        "Machine Learning (LightGBM, Random Forest, Logistic Regression)",
        "Explainable AI (SHAP, DiCE Counterfactuals)",
        "Causal Inference (DoWhy)",
        "LLM Integration (Groq/Llama 3.3)",
        "Data Drift Detection (Evidently AI)",
        "Mathematical Optimization (MILP for Budget Allocation)",
        "Streamlit (Web Framework)"
    ],
    "objective": "To develop an intelligent HR decision support system that not only predicts employee attrition but explains WHY employees leave, generates actionable retention strategies, and provides financial justification for HR interventions.",
    "key_contributions": [
        "Multi-model benchmarking with statistical validation",
        "Causal analysis to identify true attrition drivers vs correlations",
        "Counterfactual explanations for personalized retention strategies",
        "MILP-based budget optimization for maximum ROI",
        "LLM-powered communication drafting",
        "Data drift monitoring for production readiness"
    ],
    "limitations": [
        "Model trained on historical data - may not capture sudden market changes",
        "Counterfactual generation assumes linear feature relationships",
        "Budget optimization uses simplified cost formulas",
        "LLM-generated content requires human review",
        "Causal inference relies on assumed graph structure"
    ],
    "future_scope": [
        "Real-time data integration with HRIS systems",
        "Sentiment analysis of employee communications",
        "Predictive career path recommendations",
        "Automated intervention triggering based on risk thresholds",
        "Multi-location and multi-currency support"
    ]
}

# ====================================================================
# 1. ADVANCED UI STYLING (CSS)
# ====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding-left: 20px;
        padding-right: 20px;
    }

    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #0E1117;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 400;
        color: #9ca3af;
    }

    .custom-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        color: #c9d1d9;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #17B794 0%, #11998e 100%);
        border: none;
        padding: 12px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(23, 183, 148, 0.4);
    }

    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #21262d;
        color: #ffffff;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader {
        background-color: #21262d;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .llm-response {
        background-color: #21262d;
        border-left: 4px solid #17B794;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        color: #e6edf3;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .action-item {
        background-color: #161b22;
        padding: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #17B794;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .action-item-high-effort {
        border-left: 3px solid #EEB76B;
    }
    
    .prediction-card {
        display: flex;
        background-color: #1c2128;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #30363d;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    .prediction-card:hover {
        box-shadow: 0 12px 32px rgba(0,0,0,0.5);
        transform: translateY(-2px);
    }
    
    .card-section {
        flex: 1;
        padding: 30px 25px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
    }
    
    .card-section:not(:last-child)::after {
        content: '';
        position: absolute;
        right: 0;
        top: 20%;
        height: 60%;
        width: 1px;
        background: linear-gradient(to bottom, transparent, #30363d, transparent);
    }
    
    .card-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    
    .card-result {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    
    .result-leave {
        color: #EEB76B;
        text-shadow: 0 0 20px rgba(238, 183, 107, 0.3);
    }
    
    .result-stay {
        color: #17B794;
        text-shadow: 0 0 20px rgba(23, 183, 148, 0.3);
    }
    
    .card-percentage {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .percentage-stay {
        color: #17B794;
    }
    
    .percentage-leave {
        color: #EEB76B;
    }
    
    .card-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 12px;
    }
    
    .indicator-stay {
        background-color: #17B794;
        box-shadow: 0 0 10px rgba(23, 183, 148, 0.5);
    }
    
    .indicator-leave {
        background-color: #EEB76B;
        box-shadow: 0 0 10px rgba(238, 183, 107, 0.5);
    }
    
    .card-section-first {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    }
    
    .highlight-leave .card-section-first {
        background: linear-gradient(135deg, #2d2515 0%, #1c2128 100%);
    }
    
    .highlight-stay .card-section-first {
        background: linear-gradient(135deg, #0d2818 0%, #1c2128 100%);
    }
    
    @media (max-width: 768px) {
        .prediction-card {
            flex-direction: column;
        }
        .card-section:not(:last-child)::after {
            right: 20%;
            top: auto;
            bottom: 0;
            height: 1px;
            width: 60%;
            background: linear-gradient(to right, transparent, #30363d, transparent);
        }
        .card-result {
            font-size: 1.8rem;
        }
        .card-percentage {
            font-size: 1.5rem;
        }
    }
    
    /* NEW: Documentation Page Styles */
    .doc-section {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
    }
    .doc-section h3 {
        color: #17B794;
        margin-top: 0;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    .tech-badge {
        display: inline-block;
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 5px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .contribution-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .contribution-item::before {
        content: "✓";
        color: #17B794;
        font-weight: bold;
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .limitation-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .limitation-item::before {
        content: "⚠";
        color: #EEB76B;
        margin-right: 10px;
        font-size: 1.1rem;
    }
    
    /* NEW: Metric Cards Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #17B794;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# TEMPERATURE SCALING FUNCTION (Improved with configurable temp)
# ====================================================================
def calibrate_probability(prob, temperature=0.55):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    scaled_logit = logit(prob) * temperature
    return float(expit(scaled_logit))

def calibrate_probability_array(probs, temperature=0.55):
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    scaled_logit = logit(probs) * temperature
    return expit(scaled_logit)

# ====================================================================
# DOWNLOAD HELPER FUNCTION (NEW)
# ====================================================================
def get_download_link(df, filename="export.csv", text="⬇️ Download CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display:inline-flex;align-items:center;gap:6px;padding:10px 20px;background:linear-gradient(90deg,#17B794,#11998e);color:white;border-radius:8px;text-decoration:none;font-weight:600;">{text}</a>'
    return href

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={"font": {"size": title_size, "family": "tahoma"}},
        hoverlabel={"bgcolor": "#000", "font_size": hover_font_size, "font_family": "arial"},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9"
    )

def box_plot(the_df, column):
    fig = px.box(
        data_frame=the_df, x=column, title=f'{column.title().replace("_", " ")} Distribution & 5-Summary',
        template="plotly_dark", labels={column: column.title().replace("_", " ")}, height=600,
        color_discrete_sequence=['#17B794']
    )
    custome_layout(fig, showlegend=False)
    return fig

def bar_plot(the_df, column, orientation="v", top_10=False):
    dep = the_df[column].value_counts()
    if top_10:
        dep = the_df[column].value_counts().nlargest(10)
    fig = px.bar(data_frame=dep,
                 x=dep.index,
                 y=dep.values,
                 orientation=orientation,
                 color=dep.index.astype(str),
                 title=f'Observations Distribution Via {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794"],
                 labels={"x": column.title().replace("_", " "),
                         "y": "Count of Employees"},
                 template="plotly_dark",
                 text_auto=True,
                 height=650)
    custome_layout(fig, title_size=28)
    return fig

def pie_chart(the_df, column):
    counts = the_df[column].value_counts()
    fig = px.pie(data_frame=counts,
                 names=counts.index,
                 values=counts.values,
                 title=f'Popularity of {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"],
                 template="plotly_dark",
                 height=650
                 )
    custome_layout(fig, showlegend=True, title_size=28)
    pulls = np.zeros(len(counts))
    if len(pulls) > 1:
        pulls[-1] = 0.1
    fig.update_traces(
        textfont={"size": 16, "family": "arial", "color": "#fff"},
        hovertemplate="Label:%{label}<br>Frequency: %{value:0.4s}<br>Percentage: %{percent}",
        marker=dict(line=dict(color='#000000', width=0.5)),
        pull=pulls,
    )
    return fig

def create_heat_map(the_df):
    numeric_df = the_df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()
    fig = px.imshow(
        correlation,
        template="plotly_dark",
        text_auto="0.2f",
        aspect=1,
        color_continuous_scale="greens",
        title="Correlation Heatmap of Data",
        height=650,
    )
    custome_layout(fig)
    return fig

def create_vizualization(the_df, viz_type="box", data_type="number"):
    figs = []
    num_columns = list(the_df.select_dtypes(include=data_type).columns)
    cols_index = []

    if viz_type == "box":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() > 10:
                figs.append(box_plot(the_df, num_columns[i]))
                cols_index.append(i)

    if viz_type == "bar":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() < 15:
                figs.append(bar_plot(the_df, num_columns[i]))
                cols_index.append(i)

    if viz_type == "pie":
        num_columns = list(the_df.columns)
        for i in range(len(num_columns)):
            if 1 < the_df[num_columns[i]].nunique() <= 4:
                figs.append(pie_chart(the_df, num_columns[i]))
                cols_index.append(i)

    if len(cols_index) > 0:
        tabs = st.tabs([num_columns[i].title().replace("_", " ") for i in cols_index])
        for i in range(len(cols_index)):
            tabs[i].plotly_chart(figs[i], use_container_width=True)

# ====================================================================
# NEW: CONFUSION MATRIX PLOT
# ====================================================================
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=['Stay', 'Leave'],
        y=['Stay', 'Leave'],
        template="plotly_dark",
        text_auto=True,
        color_continuous_scale=['#1c2128', '#17B794'],
        title=title,
        height=400
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    custome_layout(fig, title_size=20)
    return fig

# ====================================================================
# NEW: ROC CURVE PLOT
# ====================================================================
def plot_roc_curve(y_true, y_proba_dict, title="ROC Curve Comparison"):
    fig = go.Figure()
    colors = ['#17B794', '#EEB76B', '#9C3D54']
    
    for idx, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    custome_layout(fig, title_size=20)
    return fig

# ====================================================================
# NEW: PRECISION-RECALL CURVE
# ====================================================================
def plot_precision_recall_curve(y_true, y_proba_dict, title="Precision-Recall Curve"):
    fig = go.Figure()
    colors = ['#17B794', '#EEB76B', '#9C3D54']
    
    for idx, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=model_name,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    custome_layout(fig, title_size=20)
    return fig

# ====================================================================
# NEW: DATA DRIFT DETECTION (EVIDENTLY AI - NOW ACTUALLY USED)
# ====================================================================
def run_data_drift_analysis(reference_df, current_df, target_col='left'):
    """
    Runs Evidently AI data drift detection between reference and current datasets.
    Returns the drift report and summary metrics.
    """
    try:
        # Identify numerical and categorical columns
        numerical_cols = reference_df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        categorical_cols = reference_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        column_mapping = ColumnMapping(
            target=target_col,
            prediction='prediction',
            numerical_features=numerical_cols,
            categorical_features=categorical_cols
        )
        
        # Add dummy predictions for the preset
        ref_df = reference_df.copy()
        cur_df = current_df.copy()
        ref_df['prediction'] = 0
        cur_df['prediction'] = 0
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
        
        return report, True
    except Exception as e:
        st.warning(f"Data drift analysis could not run: {e}")
        return None, False

# ====================================================================
# Logic Engine Functions
# ====================================================================

def analyze_why_people_leave(df):
    st.markdown("### 🔍 Why do people leave?")
    st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Our AI has analyzed the data to find the root causes of attrition.</p>", unsafe_allow_html=True)
    
    required_cols = ['salary', 'satisfaction_level', 'average_montly_hours', 'number_project']
    if all(col in df.columns for col in required_cols):
        df_causal = df.copy()
        salary_map = {'low': 1, 'medium': 2, 'high': 3}
        df_causal['salary_num'] = df_causal['salary'].map(salary_map)
        causal_graph = """digraph { salary_num -> satisfaction_level; satisfaction_level -> left; average_montly_hours -> left; number_project -> average_montly_hours; }"""
        df_model = df_causal[['salary_num', 'satisfaction_level', 'average_montly_hours', 'number_project', 'left']]
        
        st.markdown("""
        <div class="custom-card">
            <h4 style='margin-top:0; color: #17B794;'>📊 AI Causal Logic Diagram</h4>
            <p style='font-size: 0.9em; color: #8b949e;'>Internal hypothesis: Salary impacts Satisfaction, which leads to Attrition.</p>
        </div>
        """, unsafe_allow_html=True)
        st.graphviz_chart(causal_graph)
        st.markdown("<br>", unsafe_allow_html=True)

        effects = {}
        model_sal = CausalModel(data=df_model, treatment='salary_num', outcome='left', graph=causal_graph.replace('\n', ' '))
        est_sal = model_sal.estimate_effect(model_sal.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
        effects['Salary'] = abs(est_sal.value)
        model_sat = CausalModel(data=df_model, treatment='satisfaction_level', outcome='left', graph=causal_graph.replace('\n', ' '))
        est_sat = model_sat.estimate_effect(model_sat.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
        effects['Satisfaction'] = abs(est_sat.value)
        model_hr = CausalModel(data=df_model, treatment='average_montly_hours', outcome='left', graph=causal_graph.replace('\n', ' '))
        est_hr = model_hr.estimate_effect(model_hr.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
        effects['Overwork'] = abs(est_hr.value) * 10

        sorted_effects = sorted(effects.items(), key=lambda item: item[1], reverse=True)
        def get_display_info(rank, factor, value):
            if rank == 1: color = "#FF4B4B"; status = "CRITICAL DRIVER"; advice = "This is the #1 reason people leave."
            elif rank == 2: color = "#FFA500"; status = "MAJOR FACTOR"; advice = "Important to address."
            else: color = "#FFD700"; status = "MODERATE FACTOR"; advice = "Monitor this factor."
            return color, status, advice

        c1, c2, c3 = st.columns(3)
        for idx, (col, factor_value) in enumerate(sorted_effects):
            color, status, advice = get_display_info(idx + 1, col, factor_value)
            card_html = f"<div style='background-color: {color}20; border: 1px solid {color}; border-radius: 12px; padding: 20px; text-align: center; height: 100%;'><h2 style='color: {color}; margin: 0; font-size: 2rem;'>#{idx+1} {col}</h2><h4 style='color: white; margin: 10px 0; font-weight: 600;'>{status}</h4><p style='color: #ccc; font-size: 0.9rem;'>{advice}</p></div>"
            with [c1, c2, c3][idx]: st.markdown(card_html, unsafe_allow_html=True)

        with st.expander("🔧 Technical Validation"):
            st.write("### 1. Random Common Cause Test")
            try:
                refute_rcc = model_sal.refute_estimate(model_sal.identify_effect(), est_sal, method_name="random_common_cause")
                st.table(refute_rcc.refutation_result)
            except Exception as e: st.error(f"Error: {e}")
            st.write("---")
            try:
                refactor_placebo = model_sal.refute_estimate(model_sal.identify_effect(), est_sal, method_name="placebo_treatment_refuter")
                st.write("### 2. Placebo Treatment Refuter")
                st.table(refactor_placebo.refutation_result)
            except Exception: pass
            
            # NEW: Causal Effect Summary Table
            st.write("---")
            st.write("### 3. Causal Effect Summary")
            effect_df = pd.DataFrame([
                {"Factor": k, "Estimated Causal Effect": v, "Interpretation": "Positive effect on attrition" if v > 0 else "Protective factor"}
                for k, v in effects.items()
            ])
            st.dataframe(effect_df, use_container_width=True)
            st.markdown(get_download_link(effect_df, "causal_effects.csv", "📥 Export Causal Analysis"), unsafe_allow_html=True)
    else:
        st.info("📊 *Advanced Causal Graph requires specific columns (satisfaction_level, salary, etc.) which are not in this uploaded dataset. Using dynamic SHAP analysis instead.*")


# ====================================================================
# Evaluation 2: Intelligent Interface Functions
# ====================================================================

def run_groq_consultant(employee_name, department, situation, solution, budget):
    st.subheader("✍️ AI Communication Assistant")
    st.write("Describe the situation, and our AI will draft a professional response for you.")
    
    if "overwork" in situation.lower():
        root_cause = "High Workload & Potential Burnout"
    elif "salary" in situation.lower():
        root_cause = "Compensation & Salary Competitiveness"
    elif "morale" in situation.lower():
        root_cause = "Low Job Satisfaction & Morale"
    else:
        root_cause = "Attrition Risk Factors"
        
    action_description = solution
    cost_str = budget

    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            st.warning("🔑 System Error: API Key missing.")
            return

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            timeout=30
        )
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return

    template = """
    You are an expert HR Consultant.
    
    **Employee:** {employee_name} ({department})
    **Situation (HR View):** {situation}
    **Technical Root Cause (Internal):** {root_cause}
    **Proposed Solution:** {action_description}
    **Estimated Cost:** {cost_str}
    
    **Task:**
    Write a polite, professional, and empathetic email draft from the HR Manager to the employee.
    Acknowledge their value. Gently address the situation. Propose the solution clearly.
    
    **Tone:** Professional, Supportive.
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    with st.spinner("Drafting your message..."):
        try:
            response = chain.invoke({
                "employee_name": employee_name,
                "department": department,
                "situation": situation,
                "root_cause": root_cause,
                "action_description": action_description,
                "cost_str": cost_str
            })
            
            st.markdown("#### 📧 Generated Email Draft")
            st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
            
            # NEW: Copy to clipboard button simulation
            st.code(response, language=None)
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                st.warning("⏳ **AI is busy right now.** The servers are overloaded. Please wait 30 seconds and try again.")
            else:
                st.error(f"Error generating draft: {e}")


# ====================================================================
# NEW: PROJECT DOCUMENTATION PAGE
# ====================================================================
def render_documentation_page():
    st.header("📋 Project Documentation")
    st.markdown(f"<p style='color: #8b949e; font-size: 1.1rem;'>{PROJECT_INFO['subtitle']}</p>", unsafe_allow_html=True)
    
    # Objective Section
    st.markdown("---")
    with st.expander("🎯 Research Objective", expanded=True):
        st.markdown(f"""
        <div class="doc-section">
            <p style="font-size: 1rem; line-height: 1.8; color: #c9d1d9;">
            {PROJECT_INFO['objective']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tech Stack
    with st.expander("⚙️ Technology Stack"):
        tech_html = "".join([f"<span class='tech-badge'>{tech}</span>" for tech in PROJECT_INFO['tech_stack']])
        st.markdown(f"<div style='margin-top: 15px;'>{tech_html}</div>", unsafe_allow_html=True)
    
    # Key Contributions
    with st.expander("🏆 Key Research Contributions"):
        contrib_html = "".join([f"<div class='contribution-item'>{contrib}</div>" for contrib in PROJECT_INFO['key_contributions']])
        st.markdown(contrib_html, unsafe_allow_html=True)
    
    # Methodology
    with st.expander("🔬 Methodology", expanded=True):
        st.markdown("""
        <div class="doc-section">
            <h3>Phase 1: Data Engineering</h3>
            <ul>
                <li>Data collection and preprocessing (handling missing values, encoding)</li>
                <li>Train-test split (80-20 ratio with stratification)</li>
                <li>Feature engineering using ColumnTransformer</li>
            </ul>
            
            <h3>Phase 2: Model Development</h3>
            <ul>
                <li>Multi-model training: LightGBM, Random Forest, Logistic Regression</li>
                <li>Hyperparameter tuning with cross-validation</li>
                <li>Class imbalance handling via scale_pos_weight</li>
                <li>Probability calibration using temperature scaling</li>
            </ul>
            
            <h3>Phase 3: Explainability</h3>
            <ul>
                <li>Global explanations via SHAP summary plots</li>
                <li>Local explanations via SHAP force plots</li>
                <li>Counterfactual explanations using DiCE</li>
            </ul>
            
            <h3>Phase 4: Causal Analysis</h3>
            <ul>
                <li>Causal graph construction based on domain knowledge</li>
                <li>DoWhy backdoor adjustment for effect estimation</li>
                <li>Robustness testing (placebo, random common cause refuters)</li>
            </ul>
            
            <h3>Phase 5: Decision Support</h3>
            <ul>
                <li>MILP-based budget optimization</li>
                <li>LLM-powered communication drafting</li>
                <li>Strategic roadmap generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Limitations
    with st.expander("⚠️ Limitations"):
        limit_html = "".join([f"<div class='limitation-item'>{limit}</div>" for limit in PROJECT_INFO['limitations']])
        st.markdown(limit_html, unsafe_allow_html=True)
    
    # Future Scope
    with st.expander("🚀 Future Scope"):
        future_html = "".join([f"<div class='contribution-item'>{scope}</div>" for scope in PROJECT_INFO['future_scope']])
        st.markdown(future_html, unsafe_allow_html=True)
    
    # Author Info
    st.markdown("---")
    st.markdown("""
    <div class="doc-section" style="text-align: center;">
        <h3 style="color: #17B794;">Researcher</h3>
        <p style="font-size: 1.5rem; font-weight: 600; color: #fff; margin: 10px 0;">Nisarg Rathod</p>
        <p style="color: #8b949e;">M.Tech Project | Enterprise Workforce Intelligence</p>
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
# NEW: RESULTS SUMMARY PAGE (For Viva/Defense)
# ====================================================================
def render_results_summary(pipeline, df, X_train_ref, X_test_cur, y_train, y_test):
    st.header("📊 Results Summary")
    st.markdown("<p style='color: #8b949e;'>Quick overview of model performance for thesis defense.</p>", unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("### 📈 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Training Set", f"{len(X_train_ref):,}")
    c3.metric("Test Set", f"{len(X_test_cur):,}")
    c4.metric("Attrition Rate", f"{df['left'].mean()*100:.1f}%")
    
    # Model Performance Summary
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Summary")
    
    with st.spinner("Generating performance metrics..."):
        y_pred = pipeline.predict(X_test_cur)
        y_proba = pipeline.predict_proba(X_test_cur)[:, 1]
        
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "Value": [
                f"{accuracy_score(y_test, y_pred):.4f}",
                f"{precision_score(y_test, y_pred):.4f}",
                f"{recall_score(y_test, y_pred):.4f}",
                f"{f1_score(y_test, y_pred):.4f}",
                f"{roc_auc_score(y_test, y_proba):.4f}"
            ],
            "Interpretation": [
                "Overall correctness of predictions",
                "Of predicted 'Leave', how many actually left",
                "Of actual 'Leave', how many we caught",
                "Balance between Precision and Recall",
                "Ability to distinguish between classes"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown(get_download_link(metrics_df, "model_metrics.csv", "📥 Export Metrics"), unsafe_allow_html=True)
    
    # Classification Report
    st.markdown("---")
    st.markdown("### 📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Stay', 'Leave'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Visual Summary
    st.markdown("---")
    st.markdown("### 📊 Visual Summary")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred, "Confusion Matrix (LightGBM)"), use_container_width=True)
    with col_v2:
        # Feature Importance Bar
        shap_vals, X_proc_df = get_shap_explanations(pipeline, df)
        if shap_vals is not None:
            if isinstance(shap_vals, list):
                importance_vals = np.abs(shap_vals[1]).mean(0)
            else:
                importance_vals = np.abs(shap_vals).mean(0)
            
            feat_imp_df = pd.DataFrame({
                'Feature': X_proc_df.columns,
                'Importance': importance_vals
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig = px.bar(
                feat_imp_df, x='Importance', y='Feature',
                orientation='h',
                title="Top 10 Feature Importance (SHAP)",
                template="plotly_dark",
                color_discrete_sequence=['#17B794']
            )
            custome_layout(fig, title_size=18)
            st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ====================================================================
    # CROSS-PAGE NAVIGATION HANDLER (FIXED FLOW)
    # ====================================================================
    menu_options = [
        '⚙️ Global Setup', 
        'Home', 
        'Employee Insights', 
        'Predict Attrition', 
        'Why They Leave', 
        'Budget Planner', 
        'AI Assistant', 
        'AI Research Lab', 
        'Strategic Roadmap',
        '📊 Results Summary',  # NEW
        '📋 Documentation'     # NEW
    ]
    
    default_idx = 1  # Default to 'Home'
    if 'nav_to' in st.session_state:
        target_page = st.session_state.pop('nav_to')
        if target_page in menu_options:
            default_idx = menu_options.index(target_page)

    # ====================================================================
    # THE GLOBAL ROUTER
    # ====================================================================
    if 'is_global' in st.session_state and st.session_state['is_global']:
        pipeline = st.session_state['global_pipeline']
        df = st.session_state['global_df']
        X_train_ref = st.session_state['global_X_train']
        X_test_cur = st.session_state['global_X_test']
        y_train = st.session_state.get('global_y_train', pd.Series([0]))
        y_test = st.session_state.get('global_y_test', pd.Series([0]))
        preprocessor = pipeline.named_steps['preprocessor']
        st.toast("✅ Using Custom Uploaded Company Data", icon="📊")
    else:
        @st.cache_data
        def load_data_and_train_model(_model_version="v4_calibrated"):
            # FIX: Better error handling with redirect option
            if not os.path.exists('HR_comma_sep.csv'):
                st.error("❌ Default dataset 'HR_comma_sep.csv' not found.")
                st.markdown("""
                <div class="custom-card" style="border-left: 4px solid #EEB76B;">
                    <h4 style="color: #EEB76B; margin-top: 0;">Options:</h4>
                    <ol style="color: #c9d1d9;">
                        <li>Place 'HR_comma_sep.csv' in the same folder as this script</li>
                        <li>Go to <strong>⚙️ Global Setup</strong> to upload your own dataset</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Global Setup"):
                    st.session_state['nav_to'] = "⚙️ Global Setup"
                    st.rerun()
                st.stop()
                
            st.write("📂 Step 1/3: Loading Dataset from CSV...")
            df = pd.read_csv('HR_comma_sep.csv')
            
            st.write("🧹 Step 2/3: Preprocessing & Splitting Data...")
            df_original = df.copy()
            df_train = df.drop_duplicates().reset_index(drop=True)
            X = df_train.drop('left', axis=1)
            y = df_train['left']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            categorical_features = X.select_dtypes(include=['object']).columns
            numerical_features = X.select_dtypes(include=np.number).columns
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
            
            st.write("🤖 Step 3/3: Training AI Model (LightGBM)...")
            
            best_params = {
                'n_estimators': 150, 
                'learning_rate': 0.05, 
                'num_leaves': 12, 
                'max_depth': 4, 
                'min_child_samples': 40,
                'reg_alpha': 0.5, 
                'reg_lambda': 0.5, 
                'subsample': 0.8, 
                'colsample_bytree': 0.8,
                'random_state': 42, 
                'verbose': -1,
                'scale_pos_weight': 1.5
            }
            
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', lgb.LGBMClassifier(**best_params))])
            
            final_pipeline.fit(X_train, y_train)
            
            return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model(_model_version="v4_calibrated")
        st.empty()

    @st.cache_data
    def get_shap_explanations(_pipeline, _df):
        try:
            model = _pipeline.named_steps['classifier']
            preprocessor = _pipeline.named_steps['preprocessor']
            X = _df.drop('left', axis=1).drop_duplicates()
            X_processed = preprocessor.transform(X)
            if issparse(X_processed): X_processed = X_processed.toarray()
            
            clean_names = []
            for name in preprocessor.get_feature_names_out():
                if '__' in name:
                    name = name.split('__')[-1]
                clean_names.append(name.replace('_', ' '))
            
            X_processed_df = pd.DataFrame(X_processed, columns=clean_names)
            
            # More robust booster extraction
            booster = None
            if hasattr(model, "booster_"):
                booster = model.booster_
            elif hasattr(model, "_Booster"):
                booster = model._Booster
            elif hasattr(model, "booster"):
                booster = model.booster
            else:
                booster = model  # Fallback: try directly
            
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_processed_df)
            return shap_values, X_processed_df
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            return None, None

    def get_retention_strategies(employee_data):
        strategies = []
        if isinstance(employee_data, pd.DataFrame): employee_data = employee_data.iloc[0]
        if 'satisfaction_level' in employee_data.index and employee_data['satisfaction_level'] <= 0.45: strategies.append("🗣️ Conduct 1-on-1 meeting.")
        if 'number_project' in employee_data.index:
            if employee_data['number_project'] <= 2: strategies.append("📈 Discuss career aspirations.")
            if employee_data['number_project'] >= 6: strategies.append("⚠️ Assess workload/burnout.")
        if 'time_spend_company' in employee_data.index and 'promotion_last_5years' in employee_data.index:
            if employee_data['time_spend_company'] >= 4 and employee_data['promotion_last_5years'] == 0: strategies.append("📊 Develop career path.")
        if 'last_evaluation' in employee_data.index and 'satisfaction_level' in employee_data.index:
            if employee_data['last_evaluation'] >= 0.8 and employee_data['satisfaction_level'] < 0.6: strategies.append("🏆 Acknowledge high performance.")
        if not strategies: strategies.append("✅ Monitor engagement.")
        return strategies

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div style='padding: 20px; text-align: center;'>
            <h1 style='font-size: 1.8rem; color: #17B794; margin-bottom: 0;'>RetainAI</h1>
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px; letter-spacing: 1px;'>ENTERPRISE WORKFORCE INTELLIGENCE</p>
            <p style='color: #555d6b; font-size: 0.75rem; margin-top: 5px;'>Predict • Prevent • Optimize Attrition</p>
        </div>
        <hr style='border-color: #30363d; margin: 20px 0;'>
        """, unsafe_allow_html=True)
        
        page = option_menu(
            menu_title=None,
            options=menu_options,  
            icons=['gear', 'house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill', 'clipboard-data', 'book'], 
            menu_icon="cast", default_index=default_idx, 
            styles={
                "container": {"padding": "0!important", "background-color": 'transparent'},
                "icon": {"color": "#17B794", "font-size": "18px"},
                "nav-link": {"color": "#c9d1d9", "font-size": "14px", "text-align": "left", "margin": "0px", "margin-bottom": "8px"},
                "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"},
            }
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding:20px; border-top:1px solid #2d333b;'><div style='font-size:0.85rem; color:#8b949e;'>Built by</div><div style='font-size:1.6rem; font-weight:600; color:#00E5A8; margin-bottom:10px;'>Nisarg Rathod</div><div style='display:flex; justify-content:center; gap:15px;'><a href='https://www.linkedin.com/in/nisarg-rathod/' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#0A66C2; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='16' height='16' style='filter:invert(1);'/>LinkedIn</a><a href='https://github.com/nisargrathod' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#24292e; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='16' height='16' style='filter:invert(1);'/>GitHub</a></div></div>", unsafe_allow_html=True)

    # ====================================================================
    # PAGE: GLOBAL SETUP
    # ====================================================================
    if page == "⚙️ Global Setup":
        st.header("⚙️ Global Setup: Upload Your Company Data")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Turn this AI into your company's dedicated assistant. Upload your HR dataset, and the system will automatically retrain itself.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your HR Dataset (CSV format)", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!"); st.dataframe(new_df.head())
                st.markdown("---"); st.markdown("### Step 1: Identify the Target (Attrition Column)")
                target_col = st.selectbox("Which column indicates if the employee left?", new_df.columns)
                unique_vals = new_df[target_col].unique(); left_value = st.selectbox(f"In '{target_col}', which value means 'Left'?", unique_vals)
                st.markdown("---"); st.markdown("### Step 2: Data Preprocessing Settings")
                feature_cols = [c for c in new_df.columns if c != target_col]
                categorical_auto = new_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
                numerical_auto = new_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
                c_type, c_num = st.columns(2)
                with c_type: st.write("**Text Columns**"); st.write(categorical_auto if categorical_auto else "None")
                with c_num: st.write("**Number Columns**"); st.write(numerical_auto if numerical_auto else "None")
                
                # NEW: Data validation before training
                st.markdown("---"); st.markdown("### Step 3: Data Validation")
                validation_issues = []
                if new_df.isnull().sum().sum() > 0:
                    validation_issues.append(f"⚠️ Found {new_df.isnull().sum().sum()} missing values (will be dropped)")
                if len(new_df) < 100:
                    validation_issues.append("⚠️ Dataset is very small (<100 rows). Results may not be reliable.")
                if new_df[target_col].nunique() != 2:
                    validation_issues.append("⚠️ Target column should have exactly 2 unique values.")
                
                for issue in validation_issues:
                    st.warning(issue)
                if not validation_issues:
                    st.success("✅ All validation checks passed!")
                
                if st.button("🚀 Train Custom AI Model", type="primary"):
                    with st.spinner("🤖 AI is learning your data..."):
                        y = new_df[target_col].apply(lambda x: 1 if x == left_value else 0); X = new_df[feature_cols]
                        valid_idx = X.dropna().index; X_clean = X.loc[valid_idx]; y_clean = y.loc[valid_idx]
                        
                        if len(categorical_auto) == 0: preprocessor_global = ColumnTransformer(transformers=[('num', 'passthrough', numerical_auto)])
                        else: preprocessor_global = ColumnTransformer(transformers=[('num', 'passthrough', numerical_auto), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_auto)])
                        
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
                        
                        spw = min((y_train_g == 0).sum() / (y_train_g == 1).sum(), 2.0) if (y_train_g == 1).sum() > 0 else 1.0
                        
                        global_pipeline = Pipeline(steps=[('preprocessor', preprocessor_global), ('classifier', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=12, max_depth=4, min_child_samples=40, reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, scale_pos_weight=spw))])
                        global_pipeline.fit(X_train_g, y_train_g)
                        y_pred_g = global_pipeline.predict(X_test_g); acc = accuracy_score(y_test_g, y_pred_g)
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g; st.session_state['is_global'] = True
                        
                        # NEW: Show training summary
                        st.success(f"🎉 Training Complete!")
                        c_m1, c_m2, c_m3 = st.columns(3)
                        c_m1.metric("Accuracy", f"{acc:.1%}")
                        c_m2.metric("Training Samples", f"{len(X_train_g):,}")
                        c_m3.metric("Test Samples", f"{len(X_test_g):,}")
                        
                        st.session_state['nav_to'] = "Home"
                        st.rerun()
            except Exception as e: st.error(f"Error: {e}")
        if st.button("🔄 Reset to Default Demo Data"):
            if 'is_global' in st.session_state: del st.session_state['is_global']
            st.rerun()

    # ====================================================================
    # PAGE: HOME (ENHANCED WITH ALERTS & ACTIONS)
    # ====================================================================
    if page == "Home":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        total_employees = len(df); attrition_rate = (df['left'].sum() / len(df)) * 100
        
        # PROACTIVE ALERT SYSTEM
        alerts = []
        if attrition_rate > 20: alerts.append(("🚨 CRITICAL", f"Overall Attrition rate is {attrition_rate:.1f}% (Exceeds 20% threshold)"))
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.5: alerts.append(("⚠️ WARNING", "Average Employee Satisfaction is critically low (<50%)"))
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: alerts.append(("⚠️ WARNING", "High Average Working Hours detected (Burnout risk)"))
        
        for level, msg in alerts:
            if "CRITICAL" in level: st.error(msg)
            else: st.warning(msg)
        
        if 'satisfaction_level' in df.columns:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        
        # QUICK ACTIONS
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🎯 Predict Employee", use_container_width=True):
                st.session_state['nav_to'] = "Predict Attrition"
                st.rerun()
        with c2:
            if st.button("💰 Budget Planner", use_container_width=True):
                st.session_state['nav_to'] = "Budget Planner"
                st.rerun()
        with c3:
            if st.button("📧 Draft Email", use_container_width=True):
                st.session_state['nav_to'] = "AI Assistant"
                st.rerun()
        with c4:
            if st.button("🚀 6-Month Plan", use_container_width=True):
                st.session_state['nav_to'] = "Strategic Roadmap"
                st.rerun()

        # AT-RISK EMPLOYEE LIST
        st.markdown("---")
        st.markdown("### 🔴 Top 10 At-Risk Employees (Currently Working)")
        st.caption("Predicted by AI as most likely to leave. Click 'Predict Employee' to see detailed retention strategies.")
        
        current_df = df[df['left'] == 0].copy()
        if len(current_df) > 0:
            feature_columns_home = [c for c in df.columns if c != 'left']
            risks = pipeline.predict_proba(current_df[feature_columns_home])[:, 1]
            current_df['Risk_Score'] = calibrate_probability_array(risks)
            top_risk = current_df.nlargest(10, 'Risk_Score')
            
            for idx, row in top_risk.iterrows():
                risk_pct = row['Risk_Score'] * 100
                dept = row.get('Department', 'N/A')
                salary = row.get('salary', 'N/A')
                color = "#FF4B4B" if risk_pct > 70 else "#EEB76B" if risk_pct > 50 else "#17B794"
                st.markdown(f"<div style='background:{color}15; border-left:4px solid {color}; padding:10px 15px; margin:5px 0; border-radius:0 8px 8px 0; display:flex; justify-content:space-between;'><span><strong>{dept}</strong> | {salary.title()} Salary</span><span style='color:{color}; font-weight:700;'>Risk: {risk_pct:.1f}%</span></div>", unsafe_allow_html=True)
        else:
            st.info("No current employees found in dataset.")
        
        st.markdown("---")
        st.markdown("### 📄 Employee Data Snapshot")
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown(get_download_link(df.head(100), "employee_snapshot.csv", "📥 Export Snapshot"), unsafe_allow_html=True)

    # ====================================================================
    # PAGE: EMPLOYEE INSIGHTS
    # ====================================================================
    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis")
        st.write("Explore the workforce demographics to identify patterns.")
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    # ====================================================================
    # PAGE: PREDICT ATTRITION (ENHANCED WITH BATCH)
    # ====================================================================
    if page == "Predict Attrition":
        tab_ind, tab_batch = st.tabs(["🎯 Individual Prediction", "📦 Batch Upload"])
        
        with tab_ind:
            st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9ca3af;'>Enter employee details to see if they will Stay or Leave.</p>", unsafe_allow_html=True)
            
            with st.expander("🧪 Model Diagnostics (Verification)", expanded=False):
                st.write("Not sure if the AI is working? Test it against real historical data:")
                c_test1, c_test2 = st.columns(2)
                with c_test1:
                    if st.button("Test with Employee who Left"):
                        sample = df[df['left'] == 1].iloc[0]
                        test_df = sample.drop('left').to_frame().T
                        raw_prob = pipeline.predict_proba(test_df)[0][1]
                        calibrated_prob = calibrate_probability(raw_prob)
                        pred = 1 if calibrated_prob >= 0.5 else 0
                        if pred == 1: st.success(f"✅ **Correct!** Prediction: Leave ({calibrated_prob*100:.1f}%)")
                        else: st.error(f"❌ **Incorrect.** Prediction: Stay ({calibrated_prob*100:.1f}%)")
                        st.json(sample.to_dict(), expanded=False)
                with c_test2:
                    if st.button("Test with Employee who Stayed"):
                        sample = df[df['left'] == 0].iloc[0]
                        test_df = sample.drop('left').to_frame().T
                        raw_prob = pipeline.predict_proba(test_df)[0][1]
                        calibrated_prob = calibrate_probability(raw_prob)
                        pred = 1 if calibrated_prob >= 0.5 else 0
                        if pred == 0: st.success(f"✅ **Correct!** Prediction: Stay ({(1-calibrated_prob)*100:.1f}%)")
                        else: st.error(f"❌ **Incorrect.** Prediction: Leave ({calibrated_prob*100:.1f}%)")
                        st.json(sample.to_dict(), expanded=False)

            st.markdown("---")
            feature_columns = [c for c in df.columns if c != 'left']
            is_default_data = 'satisfaction_level' in feature_columns 

            with st.form("Predict_value_form"):
                st.markdown("##### 👤 Employee Profile")
                input_data = {}
                
                if is_default_data:
                    satisfaction_map = {'Very Dissatisfied': 0.1, 'Dissatisfied': 0.3, 'Neutral': 0.5, 'Satisfied': 0.7, 'Very Satisfied': 0.9}
                    evaluation_map = {'Needs Improvement': 0.4, 'Meets Expectations': 0.7, 'Exceeds Expectations': 0.9}
                    c1, c2 = st.columns(2)
                    with c1:
                        satisfaction_text = st.select_slider('Satisfaction Level', options=satisfaction_map.keys())
                        evaluation_text = st.select_slider('Last Evaluation Score', options=evaluation_map.keys())
                        number_project = st.slider('Number of Projects', 2, 7, 4)
                        average_montly_hours = st.slider('Avg. Monthly Hours', 90, 310, 200)
                        time_spend_company = st.slider('Years at Company', 2, 10, 3)
                    with c2:
                        work_accident_text = st.selectbox('Work Accident', ('No', 'Yes'))
                        promotion_text = st.selectbox('Promotion in Last 5 Years', ('No', 'Yes'))
                        Department = st.selectbox('Department', df['Department'].unique())
                        salary = st.selectbox('Salary', df['salary'].unique())
                    
                    input_data = {'satisfaction_level': satisfaction_map[satisfaction_text], 'last_evaluation': evaluation_map[evaluation_text], 'number_project': number_project, 'average_montly_hours': average_montly_hours, 'time_spend_company': time_spend_company, 'Work_accident': 1 if work_accident_text == 'Yes' else 0, 'promotion_last_5years': 1 if promotion_text == 'Yes' else 0, 'Department': Department, 'salary': salary}
                else:
                    cols = st.columns(2)
                    for i, col in enumerate(feature_columns):
                        with cols[i%2]:
                            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                                input_data[col] = st.selectbox(col.replace('_', ' ').title(), df[col].unique())
                            else:
                                min_val = float(df[col].min()); max_val = float(df[col].max())
                                if df[col].nunique() > 50: input_data[col] = st.number_input(col.replace('_', ' ').title(), value=float(df[col].mean()), min_value=min_val, max_value=max_val)
                                else: input_data[col] = st.slider(col.replace('_', ' ').title(), min_value=min_val, max_value=max_val, value=float(df[col].mean()))

                predict_button = st.form_submit_button(label='🔮 Analyze Employee', type='primary')

            if 'prediction_result' not in st.session_state:
                st.session_state.prediction_result = None
                st.session_state.input_df = None
                st.session_state.prediction_probas = None

            if predict_button:
                input_df = pd.DataFrame([input_data])
                with st.spinner('AI is analyzing...'):
                    sleep(1)
                    input_df = input_df[feature_columns] 
                    raw_probas = pipeline.predict_proba(input_df)[0]
                    calibrated_stay = calibrate_probability(raw_probas[0], temperature=0.55)
                    calibrated_leave = 1 - calibrated_stay
                    prediction = 1 if calibrated_leave >= 0.5 else 0
                    
                    st.session_state.prediction_result = prediction
                    st.session_state.input_df = input_df
                    st.session_state.prediction_probas = [calibrated_stay, calibrated_leave]

            if st.session_state.prediction_result is not None:
                st.markdown("---")
                
                stay_prob = st.session_state.prediction_probas[0]
                leave_prob = st.session_state.prediction_probas[1]
                
                stay_percent = int(round(stay_prob * 100))
                leave_percent = int(round(leave_prob * 100))
                
                if st.session_state.prediction_result == 1:
                    result_text = "LEAVE"
                    result_class = "result-leave"
                    highlight_class = "highlight-leave"
                    stay_indicator = "indicator-leave"
                    leave_indicator = "indicator-leave"
                else:
                    result_text = "STAY"
                    result_class = "result-stay"
                    highlight_class = "highlight-stay"
                    stay_indicator = "indicator-stay"
                    leave_indicator = "indicator-stay"
                
                st.markdown(f"""
                <div class="prediction-card {highlight_class}">
                    <div class="card-section card-section-first">
                        <div class="card-label">Employee is Likely to</div>
                        <div class="card-result {result_class}">{result_text}</div>
                    </div>
                    <div class="card-section">
                        <div class="card-label">Probability to Stay</div>
                        <div class="card-percentage percentage-stay">{stay_percent}%</div>
                        <div class="card-indicator {stay_indicator}"></div>
                    </div>
                    <div class="card-section">
                        <div class="card-label">Probability to Leave</div>
                        <div class="card-percentage percentage-leave">{leave_percent}%</div>
                        <div class="card-indicator {leave_indicator}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.prediction_result == 1:
                    st.markdown("---"); st.markdown("### 💡 Recommended Actions")
                    for rec in get_retention_strategies(st.session_state.input_df): st.info(rec)
                    st.markdown("---"); st.markdown("### 🔮 AI Retention Strategies (What-If Simulator)")
                    st.write("<p style='color: #9ca3af; margin-bottom: 15px;'>Here are 3 different ways to prevent this employee from leaving, ranked by feasibility.</p>", unsafe_allow_html=True)
                    if st.button("💡 Show Me How to Keep Them", type="primary", key="gen_cf"):
                        with st.spinner("Simulating retention strategies..."):
                            try:
                                query_instance = st.session_state.input_df
                                continuous_features = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                                if not continuous_features: st.error("No numerical columns found for simulation.")
                                else:
                                    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left')
                                    m = dice_ml.Model(model=pipeline, backend='sklearn')
                                    exp = Dice(d, m, method='random')
                                    cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
                                    cf_df = cf.cf_examples_list[0].final_cfs_df
                                    original = query_instance.iloc[0]
                                    scenarios_html = []
                                    for i in range(len(cf_df)):
                                        changes = []; cf_row = cf_df.iloc[i]; has_high_effort = False
                                        for col in original.index:
                                            orig_val = original[col]; new_val = cf_row[col]
                                            if isinstance(orig_val, float):
                                                if abs(orig_val - new_val) > 0.05:
                                                    col_lower = col.lower(); action_text = ""
                                                    if 'satisfaction' in col_lower: action_text = f"🤝 <strong>Boost Engagement</strong>: Improve satisfaction score from <strong>{orig_val:.2f}</strong> to <strong>{new_val:.2f}</strong>."
                                                    elif 'hours' in col_lower:
                                                        diff = orig_val - new_val
                                                        if diff > 0: action_text = f"⏰ <strong>Reduce Workload</strong>: Cut monthly hours by ~<strong>{abs(diff):.0f}</strong>."
                                                        else: action_text = f"⏰ <strong>Increase Engagement</strong>: Adjust hours to ~<strong>{new_val:.0f}</strong>."
                                                    elif 'project' in col_lower: action_text = f"📂 <strong>Rebalance Projects</strong>: Adjust project count to <strong>{int(new_val)}</strong>."
                                                    elif 'evaluation' in col_lower: action_text = f"📊 <strong>Performance Coaching</strong>: Guide evaluation score to <strong>{new_val:.2f}</strong>."
                                                    else: action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val:.2f} to {new_val:.2f}."
                                                    if action_text: changes.append(action_text)
                                            else:
                                                if orig_val != new_val:
                                                    if 'department' in col.lower(): has_high_effort = True; action_text = f"🏢 <strong>Department Transfer</strong>: Move from <strong>{orig_val}</strong> to <strong>{new_val}</strong>. <span style='color:#EEB76B;'>(High Effort)</span>"
                                                    else: action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val} to {new_val}."
                                                    changes.append(action_text)
                                        if not changes: changes.append("• (AI suggests maintaining current status with minor supervision)")
                                        changes_str = "".join([f"<div class='action-item {'action-item-high-effort' if has_high_effort else ''}'>{c}</div>" for c in changes])
                                        scenarios_html.append(f"<div class='custom-card' style='border-color: #17B794;'><h4 style='color: #17B794; margin-top:0;'>Strategy {i+1}</h4><p style='color: #c9d1d9; font-size: 0.9rem; line-height: 1.6;'>{changes_str}</p><div style='margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;'><small style='color: #17B794;'><strong>Result:</strong> If implemented, the AI predicts the employee will <strong>STAY</strong>.</small></div></div>")
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    cols_list = [col_s1, col_s2, col_s3]
                                    for i, html in enumerate(scenarios_html):
                                        with cols_list[i]: st.markdown(html, unsafe_allow_html=True)
                            except Exception as e: st.error(f"Error generating strategies: {e}")
        
        # BATCH PREDICTION TAB (Enhanced)
        with tab_batch:
            st.markdown("### 📦 Batch Prediction")
            st.write("Upload a CSV of multiple employees to get predictions instantly.")
            batch_file = st.file_uploader("Upload Employee CSV", type=["csv"], key="batch_uploader")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                req_cols = [c for c in feature_columns if c in batch_df.columns]
                if len(req_cols) == len(feature_columns):
                    probs = pipeline.predict_proba(batch_df[feature_columns])[:, 1]
                    batch_df['Risk Score'] = calibrate_probability_array(probs)
                    batch_df['Prediction'] = batch_df['Risk Score'].apply(lambda x: "LEAVE" if x > 0.5 else "STAY")
                    batch_df['Risk %'] = (batch_df['Risk Score'] * 100).round(1).astype(str) + "%"
                    batch_df = batch_df.sort_values('Risk Score', ascending=False)
                    
                    # NEW: Summary stats
                    c_b1, c_b2, c_b3 = st.columns(3)
                    c_b1.metric("Total Processed", len(batch_df))
                    c_b2.metric("High Risk (Leave)", (batch_df['Prediction'] == 'LEAVE').sum())
                    c_b3.metric("Low Risk (Stay)", (batch_df['Prediction'] == 'STAY').sum())
                    
                    st.dataframe(batch_df, use_container_width=True)
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results", csv, "batch_predictions.csv", "text/csv")
                else:
                    missing = set(feature_columns) - set(batch_df.columns)
                    st.error(f"Missing columns in uploaded file: {missing}")

    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers")
        st.write("Understand the specific factors driving your team's attrition risk, explained simply.")
        st.write("---")
        analyze_why_people_leave(df)
        with st.spinner("Analyzing model insights..."):
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
            if shap_values is not None:
                if isinstance(shap_values, list): vals = np.abs(shap_values[1]).mean(0)
                else: vals = np.abs(shap_values).mean(0)
                feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance'])
                feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
                top_3 = feature_importance.head(3)
                def get_feature_advice(feature_name):
                    if 'satisfaction' in feature_name.lower(): return "Employee Morale", "Conduct regular engagement surveys."
                    elif 'project' in feature_name.lower(): return "Workload Balance", "Review project allocations."
                    elif 'time' in feature_name.lower() or 'tenure' in feature_name.lower(): return "Tenure", "Watch for turnover at 3-5 years."
                    elif 'salary' in feature_name.lower(): return "Compensation", "Review market rates annually."
                    else: return "Performance", "Track evaluation scores."
                c1, c2, c3 = st.columns(3)
                cols = [c1, c2, c3]
                for idx, row in enumerate(top_3.iterrows()):
                    feature = row[1]['Feature']; advice_title, advice_text = get_feature_advice(feature)
                    card_html = f"<div class='custom-card' style='text-align: center; height: 100%;'><h3 style='color: #17B794; margin-top: 0;'>{advice_title}</h3><p style='color: #c9d1d9; font-size: 0.9rem;'>{advice_text}</p><small style='color: #8b949e;'>(Source: {feature})</small></div>"
                    with cols[idx]: st.markdown(card_html, unsafe_allow_html=True)
                
                # NEW: Export feature importance
                st.markdown(get_download_link(feature_importance, "feature_importance.csv", "📥 Export Feature Importance"), unsafe_allow_html=True)
        with st.expander("🔧 Technical Deep Dive (SHAP)"):
            st.write("Below are the raw SHAP plots for data scientists.")
            if shap_values is not None:
                fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False)
                st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)
                fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False, plot_type='dot')
                st.pyplot(fig1, bbox_inches='tight'); plt.close(fig1)

    # ====================================================================
    # REBUILT PAGE: BUDGET PLANNER
    # ====================================================================
    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-bottom: 30px;'>A two-step financial playbook to secure executive budget approval and optimize your spend.</p>", unsafe_allow_html=True)
        
        # --- PHASE 1: TRUE COST OF TURNOVER ---
        st.markdown("""
        <div class="custom-card" style="border-left: 5px solid #FF4B4B; background: linear-gradient(to right, #2d1515 0%, #1c2128 100%);">
            <h3 style="color: #FF4B4B; margin-top: 0;">💸 Phase 1: The Burn Rate</h3>
            <p style="color: #e6edf3; font-size: 1rem;"><strong>The Question:</strong> "How much money are we actually losing, and where?"</p>
            <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;'>CFOs speak in money. Translating "23% attrition" into hard numbers immediately gets budget approval.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'Department' in df.columns:
            salary_map = {'low': 400000, 'medium': 600000, 'high': 900000}
            df_cost = df.copy()
            df_cost['annual_salary'] = df_cost['salary'].map(salary_map) if 'salary' in df_cost.columns else 500000
            df_left = df_cost[df_cost['left'] == 1].copy()
            
            if len(df_left) > 0:
                if 'time_spend_company' in df_left.columns:
                    tenure_multiplier = (1 + (df_left['time_spend_company'] * 0.10)).clip(upper=2.0)
                    df_left['total_cost'] = df_left['annual_salary'] * 0.5 * tenure_multiplier
                else:
                    df_left['total_cost'] = df_left['annual_salary'] * 0.75
                
                dept_costs = df_left.groupby('Department').agg(
                    employees_left=('left', 'count'),
                    total_cost=('total_cost', 'sum'),
                    avg_salary=('annual_salary', 'mean'),
                    avg_tenure=('time_spend_company', 'mean') if 'time_spend_company' in df_left.columns else pd.Series(3.0)
                ).reset_index().sort_values('total_cost', ascending=False)
                
                grand_total = dept_costs['total_cost'].sum(); total_left = dept_costs['employees_left'].sum()
                avg_exit_cost = grand_total / total_left
                
                c_shock1, c_shock2, c_shock3 = st.columns(3)
                c_shock1.metric("Total Money Lost to Attrition", f"₹{grand_total/10000000:.2f} Cr", delta=f"{total_left} Employees Left", delta_color="inverse")
                c_shock2.metric("Avg. Cost Per Exit", f"₹{avg_exit_cost:,.0f}", delta="Industry Standard Math")
                c_shock3.metric("Most Expensive Dept.", dept_costs.iloc[0]['Department'], delta=f"₹{dept_costs.iloc[0]['total_cost']/10000000:.2f} Cr Lost", delta_color="inverse")
                
                with st.expander("📐 How we calculate this (Click to verify)"):
                    st.markdown("""
                    We use the **Standard HR Industry Formula** for turnover cost:\n
                    > `Cost = 50% of Annual Salary × (1 + Years of Tenure × 10%)`
                    
                    *   **Base Replacement Cost:** 50% of salary (covers recruiting, onboarding, admin).
                    *   **Experience Penalty:** +10% for every year of experience the employee had (captures institutional knowledge loss).
                    *   **Cap:** The multiplier is capped at 2.0x (200% of salary) for very senior employees.
                    
                    *Example:* A 5-year employee earning ₹6L costs `0.5 × 6L × 1.5 = ₹4.5L` to replace.
                    """)
                
                fig_cost = px.bar(
                    dept_costs, 
                    x='Department', 
                    y='total_cost', 
                    title="True Cost of Turnover by Department",
                    template="plotly_dark",
                    color='total_cost',
                    color_continuous_scale=['#17B794', '#FF4B4B'],
                    height=450
                )
                fig_cost.update_layout(
                    yaxis_title="Total Loss (₹)",
                    yaxis=dict(tickformat=',.0f'),
                    xaxis={'categoryorder': 'total descending'},
                    xaxis_title="",
                    showlegend=False
                )
                custome_layout(fig_cost, title_size=24, showlegend=False); 
                st.plotly_chart(fig_cost, use_container_width=True)
                
                # NEW: Export cost analysis
                st.markdown(get_download_link(dept_costs, "turnover_cost_analysis.csv", "📥 Export Cost Analysis"), unsafe_allow_html=True)
                
                top_dept = dept_costs.iloc[0]
                st.markdown(f"""<div class="custom-card" style="border-left: 4px solid #FF4B4B;"><h4 style="color: #FF4B4B; margin-top: 0;">Executive Talking Point</h4><p style="color: #e6edf3; font-size: 1.1rem;"><strong>{top_dept['Department']} attrition cost us ₹{top_dept['total_cost']/10000000:.2f} Crores last year.</strong> That's the price of losing {top_dept['employees_left']} employees and their accumulated experience.</p></div>""", unsafe_allow_html=True)
            else:
                st.info("No historical attrition data found in this dataset.")
        else:
            st.warning("Department column missing. Cannot calculate departmental costs.")

        st.markdown("<br><hr style='border-color: #30363d;'><br>", unsafe_allow_html=True)
        
        # --- PHASE 2: THE ROI OPTIMIZER ---
        st.markdown("""
        <div class="custom-card" style="border-left: 5px solid #17B794; background: linear-gradient(to right, #0d2818 0%, #1c2128 100%);">
            <h3 style="color: #17B794; margin-top: 0;">🛡️ Phase 2: The ROI Optimizer</h3>
            <p style="color: #e6edf3; font-size: 1rem;"><strong>The Question:</strong> "Who should we save with our limited budget?"</p>
            <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;'>We use Mathematical Optimization (MILP) to find the exact combination of employees that yields maximum ROI. It's cheaper to give a 10% raise than to replace them.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input("Enter Total Retention Budget (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
        with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Run Optimizer", type="primary")
        
        if optimize_btn:
            with st.spinner("🧮 Running Mathematical Optimization..."):
                X = df.drop('left', axis=1)
                raw_probs = pipeline.predict_proba(X)[:, 1]
                probas = calibrate_probability_array(raw_probs, temperature=0.55)
                
                opt_df = df.copy(); opt_df['risk'] = probas
                high_risk_df = opt_df[opt_df['risk'] > 0.5].copy()
                
                if len(high_risk_df) == 0:
                    st.success("🎉 Great news! The AI predicts workforce is highly stable.")
                else:
                    if 'salary' in df.columns:
                        high_risk_df['salary_val'] = high_risk_df['salary'].map(salary_map)
                    else:
                        high_risk_df['salary_val'] = 500000 
                    
                    high_risk_df['cost_to_retain'] = high_risk_df['salary_val'] * 0.10
                    high_risk_df['expected_loss'] = high_risk_df['risk'] * (high_risk_df['salary_val'] * 0.50)
                    high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['cost_to_retain']
                    
                    candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
                    
                    if len(candidates) == 0:
                        st.warning("⚠️ Based on current risk probabilities, a 10% raise is not mathematically justifiable for the at-risk group.")
                    else:
                        n = len(candidates)
                        c = -candidates['net_savings'].values 
                        A = np.array([candidates['cost_to_retain'].values])
                        b = np.array([budget])
                        integrality = np.ones(n) 
                        
                        try:
                            res = milp(c=c, constraints=LinearConstraint(A, lb=-np.inf, ub=b), integrality=integrality)
                        except Exception as e:
                            st.error(f"Calculation Error: {e}"); res = None

                        if res and res.success:
                            selected_indices = np.where(res.x == 1)[0]
                            selected_employees = candidates.iloc[selected_indices]
                            
                            total_investment = selected_employees['cost_to_retain'].sum()
                            total_savings = selected_employees['net_savings'].sum()
                            
                            st.success("✅ **Optimization Complete.** Here is your action plan:")
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Investment Needed", f"₹{total_investment:,.0f}", delta=f"{(total_investment/budget)*100:.1f}% of Budget")
                            m2.metric("Projected Savings", f"₹{total_savings:,.0f}", delta="ROI Positive")
                            m3.metric("Lives Saved", f"{len(selected_employees)} People", delta="Prevented Exit")
                            
                            st.markdown("### 📋 Target List")
                            st.caption("These employees have been mathematically proven to be cheaper to retain than to replace.")
                            
                            display_df = selected_employees[['Department', 'salary', 'risk', 'cost_to_retain', 'net_savings']].copy()
                            display_df.rename(columns={
                                'salary': 'Salary Tier', 
                                'risk': 'AI Risk %', 
                                'cost_to_retain': 'Cost to Retain (₹)', 
                                'net_savings': 'Net Savings (₹)'
                            }, inplace=True)
                            display_df['AI Risk %'] = (display_df['AI Risk %'] * 100).round(1).astype(str) + "%"
                            display_df['Cost to Retain (₹)'] = display_df['Cost to Retain (₹)'].apply(lambda x: f"₹{x:,.0f}")
                            display_df['Net Savings (₹)'] = display_df['Net Savings (₹)'].apply(lambda x: f"₹{x:,.0f}")
                            st.dataframe(display_df, use_container_width=True)
                            
                            # NEW: Export optimization results
                            export_opt = selected_employees[['Department', 'salary', 'risk', 'cost_to_retain', 'net_savings']].copy()
                            st.markdown(get_download_link(export_opt, "retention_targets.csv", "📥 Export Target List"), unsafe_allow_html=True)
                        else:
                            st.error("❌ Optimization failed. Budget may be too low to save anyone.")
                            
    if page == "AI Assistant":
        st.header("🤖 AI Assistant")
        st.markdown("<p style='color: #9ca3af;'>Tools to ensure reliability and simplify communication.</p>", unsafe_allow_html=True)
        st.markdown("### ✍️ Draft Retention Communication")
        st.write("Select a scenario, and we'll draft a message for you.")
        with st.form("llm_form"):
            c1, c2 = st.columns(2)
            with c1:
                emp_name = st.text_input("Employee Name", value="Rahul Sharma")
                if 'Department' in df.columns: emp_dept = st.selectbox("Department", df['Department'].unique())
                else: emp_dept = st.text_input("Department", value="Sales")
            with c2:
                situation_input = st.selectbox("What is the situation?", ["Overworked & Burned out", "Seeking Higher Salary", "Low Morale / Unhappy", "Lack of Growth Opportunities"])
                solution_input = st.selectbox("Proposed Solution", ["Offer Flexible Hours", "Discuss Salary Adjustment", "Offer Promotion/Role Change", "Organize 1-on-1 Wellness Session"])
                cost_input = st.text_input("Estimated Annual Cost (Optional)", value="₹50,000")
            generate_btn = st.form_submit_button("🚀 Generate Email Draft")
            if generate_btn: run_groq_consultant(emp_name, emp_dept, situation_input, solution_input, cost_input)

    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Advanced modules for Strategy, Disruption, and Recruitment.</p>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Benchmarking", "🔬 Departmental Strategy", "🛡️ AI Disruption Defense", "📈 Data Drift Monitor"])  # NEW TAB
        
        with tab1:
            st.subheader("Algorithm Performance Comparison")
            if st.button("Run Benchmark", type="primary", key="run_benchmark"):
                with st.spinner("Training competing models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
                    rf_pipeline.fit(X_train_ref, y_train); y_pred_rf = rf_pipeline.predict(X_test_cur); proba_rf = rf_pipeline.predict_proba(X_test_cur)[:, 1]
                    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
                    lr_pipeline.fit(X_train_ref, y_train); y_pred_lr = lr_pipeline.predict(X_test_cur); proba_lr = lr_pipeline.predict_proba(X_test_cur)[:, 1]
                    
                    metrics = {'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'], 
                               'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)], 
                               'Precision': [precision_score(y_test, y_pred_lgbm), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_lr)], 
                               'Recall': [recall_score(y_test, y_pred_lgbm), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_lr)], 
                               'F1 Score': [f1_score(y_test, y_pred_lgbm), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_lr)], 
                               'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_lr)]}
                    results_df = pd.DataFrame(metrics)
                    
                    # NEW: Export metrics
                    st.markdown(get_download_link(results_df, "benchmark_results.csv", "📥 Export Benchmark Results"), unsafe_allow_html=True)
                    
                    st.markdown("### 📈 Performance Metrics")
                    st.dataframe(results_df.style.highlight_max(axis=0, color='#17B794'), use_container_width=True)
                    fig_metrics = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'), x='Metric', y='Score', color='Model', barmode='group', title="Model Comparison", template="plotly_dark", color_discrete_sequence=['#17B794', '#EEB76B', '#9C3D54'])
                    custome_layout(fig_metrics, title_size=24); st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # NEW: Confusion Matrices
                    st.markdown("### 🎯 Confusion Matrices")
                    cm_col1, cm_col2, cm_col3 = st.columns(3)
                    with cm_col1:
                        st.plotly_chart(plot_confusion_matrix(y_test, y_pred_lgbm, "LightGBM"), use_container_width=True)
                    with cm_col2:
                        st.plotly_chart(plot_confusion_matrix(y_test, y_pred_rf, "Random Forest"), use_container_width=True)
                    with cm_col3:
                        st.plotly_chart(plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression"), use_container_width=True)
                    
                    # NEW: ROC Curves
                    st.markdown("### 📊 ROC Curve Comparison")
                    roc_data = {
                        'LightGBM': proba_lgbm,
                        'Random Forest': proba_rf,
                        'Logistic Regression': proba_lr
                    }
                    st.plotly_chart(plot_roc_curve(y_test, roc_data, "ROC Curve: All Models"), use_container_width=True)
                    
                    # NEW: Precision-Recall Curves
                    st.markdown("### 📈 Precision-Recall Curve")
                    st.plotly_chart(plot_precision_recall_curve(y_test, roc_data, "Precision-Recall: All Models"), use_container_width=True)
                    
                    st.success("🏆 **Conclusion:** LightGBM was selected as the primary model due to its superior balance of Precision and Recall.")

        with tab2:
            st.subheader("🔬 Departmental Strategy Deep Dive")
            if 'Department' not in df.columns: st.warning("Department column not found in this dataset. Cannot run deep dive.")
            else:
                st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Don't just guess why a team is struggling. Use AI to uncover the <strong>specific reasons</strong> why employees are leaving a particular department and get tailored strategies.</p>", unsafe_allow_html=True)
                selected_dept_name = st.selectbox("Select Department to Analyze", options=sorted(df['Department'].unique()))
                if st.button("Generate Department Strategy", type="primary"):
                    with st.spinner("Analyzing departmental dynamics..."):
                        dept_data = df[df['Department'] == selected_dept_name]; dept_count = len(dept_data); dept_attrition = (dept_data['left'].sum() / dept_count) * 100; company_attrition = (df['left'].sum() / len(df)) * 100; delta = dept_attrition - company_attrition
                        delta_color = "normal" if delta <= 0 else "inverse"; delta_text = f"{delta:+.1f}% vs Company Avg"
                        c_m1, c_m2, c_m3 = st.columns(3)
                        c_m1.metric(f"{selected_dept_name} Workforce", f"{dept_count} Employees"); c_m2.metric("Attrition Rate", f"{dept_attrition:.1f}%", delta=delta_text, delta_color=delta_color); c_m3.metric("Risk Level", "High" if dept_attrition > 20 else "Moderate" if dept_attrition > 10 else "Low")
                        st.markdown("---"); shap_vals, X_proc_df = get_shap_explanations(pipeline, df)
                        
                        if shap_vals is not None:
                            target_col = None
                            dept_normalized = selected_dept_name.lower().replace('_', '').replace(' ', '')
                            
                            for col in X_proc_df.columns:
                                col_normalized = col.lower().replace('_', '').replace(' ', '')
                                if dept_normalized in col_normalized:
                                    target_col = col
                                    break
                            
                            if not target_col:
                                st.error(f"Could not find data for {selected_dept_name} in the model features.")
                                with st.expander("🔧 Debug Info"):
                                    st.write("**Department searched for:**", selected_dept_name)
                                    st.write("**Normalized search term:**", dept_normalized)
                                    st.write("**Available columns:**", X_proc_df.columns.tolist())
                            else:
                                dept_mask = X_proc_df[target_col] == 1
                                if dept_mask.sum() == 0: st.warning(f"Not enough data to analyze {selected_dept_name} specifically.")
                                else:
                                    if isinstance(shap_vals, list): dept_shap = shap_vals[1][dept_mask]
                                    else: dept_shap = shap_vals[dept_mask]
                                    mean_shap = np.abs(dept_shap).mean(axis=0)
                                    importance_df = pd.DataFrame({'Feature': X_proc_df.columns, 'Impact_Score': mean_shap})
                                    importance_df = importance_df[~importance_df['Feature'].str.contains('Department', case=False)]
                                    importance_df = importance_df[~importance_df['Feature'].str.match(r'^x\d')]
                                    importance_df.sort_values('Impact_Score', ascending=False, inplace=True); top_3_drivers = importance_df.head(3)
                                    chart_df = importance_df.head(5).iloc[::-1] 
                                    fig = px.bar(chart_df, x='Impact_Score', y='Feature', orientation='h', title=f"What is driving attrition in {selected_dept_name}?", template="plotly_dark", color_discrete_sequence=['#17B794'])
                                    fig.update_layout(xaxis_title="Relative Impact (Higher = More Important)", yaxis_title="", height=400, margin=dict(l=0, r=0, t=40, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown("### 💡 Recommended Retention Strategy"); st.write(f"Based on the AI analysis for the <strong>{selected_dept_name}</strong> team:", unsafe_allow_html=True)
                                    def get_driver_advice(feature_raw):
                                        feature_clean = feature_raw.replace('_', ' ').title()
                                        if 'Satisfaction' in feature_raw: return "Improve Employee Engagement", "The AI detects low morale as a primary driver. Initiate 'Stay Interviews', conduct anonymous pulse surveys, and review manager-employee relationships.", "🗣️"
                                        elif 'Hour' in feature_raw or 'Time' in feature_raw: return "Address Workload & Burnout", "Overwork is the leading cause. Review project allocation, consider hiring support staff, and enforce 'Right to Disconnect' policies.", "⏰"
                                        elif 'Project' in feature_raw: return "Optimize Work Distribution", "Employees are either bored or overwhelmed. Rebalance project assignments to ensure the 'Goldilocks' zone of productivity.", "📂"
                                        elif 'Evaluation' in feature_raw: return "Clarify Performance Expectations", "Unclear goals are causing stress. Implement clearer KPIs and more frequent, constructive feedback loops.", "📊"
                                        elif 'Salary' in feature_raw: return "Review Compensation Competitiveness", "Pay is a major factor. Conduct a market salary analysis for this specific department and adjust bands if necessary.", "💰"
                                        elif 'Tenure' in feature_raw or 'Spend' in feature_raw: return "Focus on Career Growth", "Long-tenured employees feel stagnant. Create clear internal promotion pathways or rotation programs.", "📈"
                                        else: return f"Monitor {feature_clean}", f"AI identified {feature_clean} as a key differentiator. Investigate department-specific policies related to this metric.", "🔍"
                                    c1, c2, c3 = st.columns(3); cols = [c1, c2, c3]
                                    for index, col in enumerate(cols):
                                        if index < len(top_3_drivers):
                                            driver_row = top_3_drivers.iloc[index]; feature_name = driver_row['Feature']; impact_score = driver_row['Impact_Score']
                                            icon, title, advice = get_driver_advice(feature_name)
                                            card_html = f"<div class='custom-card' style='border-top: 4px solid #17B794;'><div style='display: flex; align-items: center; margin-bottom: 10px;'><span style='font-size: 1.5rem; margin-right: 10px;'>{icon}</span><h4 style='margin: 0; color: #fff;'>{title}</h4></div><p style='color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;'>{advice}</p><small style='color: #8b949e;'>Driver: {feature_name.replace('_', ' ').title()}</small></div></div>"
                                            with col: st.markdown(card_html, unsafe_allow_html=True)

        # AI DISRUPTION DEFENSE
        with tab3:
            st.subheader("🛡️ AI Disruption Defense")
            st.caption("Prove to leadership that reskilling is cheaper than mass layoffs.")
            st.error("**The Fear:** CEO asks, 'Can we just replace half the team with AI tools?'")
            st.success("**The Reality:** AI replaces *tasks*, not jobs. And layoffs cost way more than you think.")
            st.write("")
            
            if 'Department' in df.columns and 'satisfaction_level' in df.columns:
                st.markdown("### Step 1: Department Vulnerability Assessment")
                dept_vuln = []
                for dept in df['Department'].unique():
                    dept_data = df[df['Department'] == dept]
                    avg_sat = dept_data['satisfaction_level'].mean()
                    avg_eval = dept_data['last_evaluation'].mean() if 'last_evaluation' in dept_data.columns else 0.5
                    avg_tenure = dept_data['time_spend_company'].mean() if 'time_spend_company' in dept_data.columns else 3.0
                    vuln_score = ((1 - avg_sat) * 40) + ((1 - avg_eval) * 30) + ((1 - (avg_tenure/10)) * 30)
                    dept_vuln.append({'Department': dept, 'Vulnerability Score': round(vuln_score), 'Employees': len(dept_data)})
                vuln_df = pd.DataFrame(dept_vuln).sort_values('Vulnerability Score', ascending=False)
                fig_vuln = px.bar(vuln_df, x='Vulnerability Score', y='Department', orientation='h', title="Which departments have roles most easily replaced by AI?", color='Vulnerability Score', color_continuous_scale=['#17B794', '#EEB76B'], template="plotly_dark", height=400)
                fig_vuln.update_layout(xaxis_title="0 = Safe (Complex Work) | 100 = At Risk (Repetitive Work)", yaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_vuln, use_container_width=True)
                st.info("**How to read this:** A score of 70+ means the department likely has many repetitive tasks. A score under 30 means the work is too complex for current AI.")

            st.markdown("---")
            st.markdown("""
            <div class="custom-card" style="border-left: 5px solid #9ca3ca; background: linear-gradient(to right, #151d28 0%, #1c2128 100%);">
                <h4 style="color: #9ca3ca; margin-top: 0;">🧠 The 'Proof of Work' Defense Strategy</h4>
                <p style="color: #e6edf3; font-size: 1rem; font-weight: 600; margin-bottom: 10px;">What to say when the CEO asks: "Can we just use AI to cut headcount?"</p>
                <p style="color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; margin-bottom: 10px;">
                "We shouldn't try to compete with AI on speed; we must compete on strategy. AI can write code in 10 seconds, but it takes a human 3 weeks to understand the business context, manage stakeholder politics, and ensure compliance. If we lay off 200 people to save ₹12 Cr, we lose their domain expertise. If we give 50 people AI tools, we increase their capacity by 30%, allowing us to <strong>absorb the workload of the 200 people who left last year</strong> without losing institutional knowledge."
                </p>
                <p style="color: #8b949e; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0;">
                <strong>Bottom Line:</strong> AI doesn't replace jobs; it replaces tasks. Our goal isn't to have fewer people; it's to have highly utilized people. We should pitch AI as a way to make our existing team 30% faster, not as a way to fire people.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Step 2: The Billion-Rupee Calculator (Reskill vs. Layoff)")
            c_inp1, c_inp2 = st.columns(2)
            with c_inp1:
                num_employees = st.number_input("How many employees are at risk of AI replacement?", min_value=5, max_value=500, value=50, step=5)
                avg_salary = st.number_input("Their average annual salary (₹)", min_value=200000, max_value=3000000, value=600000, step=50000)
            with c_inp2: st.markdown("**Industry Standard Costs:**"); st.caption("Severance: 3 months salary | Hiring AI Engineer: 2x salary | Training: 1.5x salary | Morale dip: 1 month salary")
            st.markdown("---")
            severance_cost = (avg_salary / 12) * 3 * num_employees
            new_hire_cost = (avg_salary * 2) * (num_employees * 0.1)
            total_layoff_cost = severance_cost + new_hire_cost
            training_cost = (avg_salary / 12) * 1.5 * num_employees
            productivity_dip = (avg_salary / 12) * 1 * num_employees
            total_reskill_cost = training_cost + productivity_dip
            savings = total_layoff_cost - total_reskill_cost
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Total Cost of Layoffs", f"₹{total_layoff_cost/10000000:.2f} Cr")
            c_res2.metric("Total Cost of Reskilling", f"₹{total_reskill_cost/10000000:.2f} Cr")
            c_res3.metric("Money Saved by Reskilling", f"₹{savings/10000000:.2f} Cr", delta="Reskill wins")
            st.markdown("---")
            if savings > 0: st.success(f"**The Verdict: Reskill.** By choosing to reskill instead of laying off, you save ₹{savings/10000000:.2f} Crores.")
            else: st.warning(f"**The Verdict: Layoffs are technically cheaper here. However, consider hidden costs before proceeding.")
            if st.button("✍️ Generate Strategy Memo for CEO", type="primary", key="gen_ai_memo"):
                with st.spinner("Drafting strategy memo..."):
                    try:
                        api_key = st.secrets.get("GROQ_API_KEY", None)
                        if not api_key: st.warning("🔑 System Error: API Key missing. Showing generic template.")
                        else:
                            llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5, timeout=30)
                            template = """You are an HR Director writing to the CEO. Keep it strictly under 400 words. Use bullet points. No fluff. **Context:** - We analyzed {num_employees} employees at risk of AI automation. - Layoff + rehiring AI talent costs: {total_layoff_cost}. - Reskilling the same team costs: {total_reskill_cost}. - Reskilling saves us {savings}. **Task:** Recommend reskilling. Briefly explain why layoffs are a trap. Propose a 6-month pilot reskilling program."""
                            prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
                            response = chain.invoke({"num_employees": num_employees, "total_layoff_cost": f"₹{total_layoff_cost/10000000:.1f} Cr", "total_reskill_cost": f"₹{total_reskill_cost/10000000:.1f} Cr", "savings": f"₹{savings/10000000:.1f} Cr"})
                            st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        if "rate_limit" in str(e).lower() or "429" in str(e): st.warning("⏳ **AI is busy right now.** Please wait 30 seconds and try again.")
                        else: st.error(f"❌ Error generating memo: {e}")
        
        # ====================================================================
        # NEW TAB: DATA DRIFT MONITORING (EVIDENTLY AI - NOW ACTUALLY IMPLEMENTED)
        # ====================================================================
        with tab4:
            st.subheader("📈 Data Drift Monitor")
            st.markdown("""
            <div class="custom-card" style="border-left: 5px solid #9C3D54;">
                <h4 style="color: #9C3D54; margin-top: 0;">Why Data Drift Matters?</h4>
                <p style="color: #c9d1d9; margin-bottom: 0;">
                ML models degrade over time as the real-world data distribution changes. An employee who was "satisfied" in 2019 
                may have completely different expectations in 2024. This tab helps you detect when your model needs retraining.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### How It Works")
            st.write("""
            We compare your **training data** (reference) against a **simulated current dataset** (current). 
            If the statistical distributions of features change significantly, the model's predictions become unreliable.
            """)
            
            if st.button("🔍 Run Drift Detection", type="primary"):
                with st.spinner("Analyzing data drift using Evidently AI..."):
                    # Create a synthetic "current" dataset by slightly modifying the original
                    # (In production, this would be actual new data)
                    reference_data = df.copy()
                    
                    # Simulate drift by adding noise and shifting distributions
                    np.random.seed(42)
                    current_data = df.copy()
                    numerical_cols = current_data.select_dtypes(include=np.number).columns.drop('left', errors='ignore')
                    
                    for col in numerical_cols:
                        # Add random drift (5-15% shift)
                        drift_factor = np.random.uniform(0.05, 0.15)
                        mean_shift = current_data[col].mean() * drift_factor
                        current_data[col] = current_data[col] + np.random.normal(mean_shift, current_data[col].std() * 0.1, len(current_data))
                    
                    # Run Evidently AI
                    drift_report, success = run_data_drift_analysis(reference_data, current_data, 'left')
                    
                    if success and drift_report:
                        st.success("✅ Drift analysis completed successfully!")
                        
                        # Display the report
                        st.markdown("### 📊 Drift Report")
                        drift_report.show()
                        
                        # Custom summary
                        st.markdown("---")
                        st.markdown("### 📋 Interpretation Guide")
                        st.markdown("""
                        <div class="custom-card">
                            <h4 style="color: #17B794; margin-top: 0;">How to Read This Report</h4>
                            <ul style="color: #c9d1d9; line-height: 1.8;">
                                <li><strong style="color: #17B794;">Green/Low Drift:</strong> Model is still reliable. No action needed.</li>
                                <li><strong style="color: #EEB76B;">Medium Drift:</strong> Some features are changing. Monitor closely.</li>
                                <li><strong style="color: #FF4B4B;">High Drift:</strong> Model needs retraining ASAP. Predictions are unreliable.</li>
                            </ul>
                            <p style="color: #8b949e; margin-top: 15px; margin-bottom: 0;">
                            <strong>Common Drift Causes:</strong> Economic changes, company policy changes, pandemic effects, 
                            new competitor in market, generational shift in workforce expectations.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Note about simulated data
                        st.info("📌 **Note:** This demo uses simulated drift on your dataset. In production, compare against actual new HR data.")
                    else:
                        st.warning("Could not generate drift report. Check if Evidently AI is properly installed.")
                        st.code("pip install evidently", language="bash")

    # ====================================================================
    # NEW PAGE: RESULTS SUMMARY (For M.Tech Viva)
    # ====================================================================
    if page == "📊 Results Summary":
        render_results_summary(pipeline, df, X_train_ref, X_test_cur, y_train, y_test)

    # ====================================================================
    # NEW PAGE: DOCUMENTATION (For M.Tech Thesis)
    # ====================================================================
    if page == "📋 Documentation":
        render_documentation_page()

    # ====================================================================
    # Page: STRATEGIC ROADMAP
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Future Planning & Projections")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>A simple tool to show leadership exactly what happens if we take action vs. if we do nothing.</p>", unsafe_allow_html=True)
        st.markdown("### 📋 Step 1: Get Your 6-Month Action Plan")
        issues = []
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.6: issues.append("Low Employee Satisfaction")
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: issues.append("Employee Burnout (High Working Hours)")
        if len(issues) == 0: issues.append("Standard Workforce Stabilization")
        issues_str = ", ".join(issues)
        st.markdown(f"""<div class="custom-card"><h4 style="color: #17B794; margin-top: 0;">🩺 AI Diagnostic Summary</h4><p style="color: #c9d1d9; line-height: 1.6;">Before making a plan, here is what the AI flagged as your biggest risks:<br><strong style="color: #EEB76B;">➤ {issues_str}</strong></p></div>""", unsafe_allow_html=True)
        if st.button("✍️ Draft My 6-Month HR Action Plan", type="primary"):
            with st.spinner("Drafting your 6-month strategy..."):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if api_key:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                        template = """You are an expert HR Strategist. **Context:** Our AI identified these attrition drivers: {issues}. **Task:** Create a 6-month execution roadmap. Break it into phases. For each month give: 1. Phase Name, 2. Actionable Steps (2-3 bullets), 3. Success Metrics. **Tone:** Plain English. Practical HR actions."""
                        prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({"issues": issues_str})
                        st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    else: st.warning("🔑 API Key missing. Showing generic template.")
                except Exception as e: st.error(f"Error: {e}")

        st.markdown("---"); st.markdown("### 📈 Step 2: See the Future Impact (12-Month Projection)")
        col_f1, col_f2 = st.columns(2)
        with col_f1: intervention_efficacy = st.slider("If we take action, how many at-risk people will we actually save? (%)", 10, 50, 20, 5)
        with col_f2: natural_attrition_rate = st.slider("People who leave for personal reasons (%)", 0.5, 2.0, 1.0, 0.1)
        if st.button("📈 Show Me the 12-Month Projection", type="primary"):
            months = list(range(1, 13)); current_workforce = len(df)
            raw_risk_scores = pipeline.predict_proba(df.drop('left', axis=1))[:, 1]
            total_risk_score = calibrate_probability_array(raw_risk_scores, temperature=0.55).sum()
            monthly_leavers_no_action = total_risk_score / 12.0
            monthly_leavers_with_action = monthly_leavers_no_action * (1 - (intervention_efficacy / 100.0))
            forecast_bau, forecast_intervention, temp_bau, temp_int = [], [], float(current_workforce), float(current_workforce)
            for m in months:
                natural_leavers_bau = temp_bau * (natural_attrition_rate / 100.0)
                natural_leavers_int = temp_int * (natural_attrition_rate / 100.0)
                temp_bau -= monthly_leavers_no_action + natural_leavers_bau
                temp_int -= monthly_leavers_with_action + natural_leavers_int
                forecast_bau.append(temp_bau); forecast_intervention.append(temp_int)
            forecast_df = pd.DataFrame({'Month': months, 'If We Do Nothing (Status Quo)': forecast_bau, 'If We Follow the Plan': forecast_intervention}).melt(id_vars='Month', var_name='Scenario', value_name='Workforce Count')
            fig_forecast = px.line(forecast_df, x='Month', y='Workforce Count', color='Scenario', title="Projected Workforce Size Over the Next 12 Months", template="plotly_dark", markers=True, color_discrete_map={'If We Do Nothing (Status Quo)': "#EEB76B", 'If We Follow the Plan': "#17B794"})
            fig_forecast.update_layout(yaxis_title="Total Employee Headcount", xaxis=dict(dtick=1)); st.plotly_chart(fig_forecast, use_container_width=True)
            saved_employees = forecast_intervention[-1] - forecast_bau[-1]
            if 'salary' in df.columns: avg_salary = df['salary'].map({'low': 400000, 'medium': 600000, 'high': 900000}).mean()
            else: avg_salary = 500000 
            replacement_cost_per_emp = avg_salary * 0.5; total_money_saved = int(saved_employees) * replacement_cost_per_emp
            st.markdown("---"); st.markdown("### 🏢 HR Director Summary (For Leadership)")
            col_sum_1, col_sum_2 = st.columns(2)
            with col_sum_1: st.metric("Employees Saved by Plan", f"{int(saved_employees)} People")
            with col_sum_2: st.metric("Estimated Recruitment Costs Prevented", f"₹{total_money_saved:,.0f}", delta="Financial Value")
            st.success(f"**The Bottom Line:** Retaining {intervention_efficacy}% saves {int(saved_employees)} employees and prevents **₹{total_money_saved:,.0f}** in costs.")
            
            # NEW: Export projection data
            proj_export = pd.DataFrame({
                'Month': months,
                'Status Quo (No Action)': [round(x, 0) for x in forecast_bau],
                'With Intervention': [round(x, 0) for x in forecast_intervention]
            })
            st.markdown(get_download_link(proj_export, "workforce_projection.csv", "📥 Export Projection Data"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
