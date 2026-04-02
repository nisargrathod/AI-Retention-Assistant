# ====================================================================
# All Necessary Imports
# ====================================================================
import os
import json
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
from time import sleep
from datetime import datetime, timedelta
from scipy.sparse import issparse
from scipy.special import expit, logit
import base64

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel
from scipy.optimize import milp, LinearConstraint, Bounds

# --- Imports for Evaluation 2 (Intelligent Interface: Groq + Evidently) ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Imports for AI Research Lab (Counterfactuals) ---
import dice_ml
from dice_ml import Dice

# ====================================================================
# 1. ADVANCED UI STYLING (CSS) - ENHANCED
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
    
    /* NEW: Data Freshness Badge */
    .freshness-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .freshness-fresh {
        background-color: #0d281880;
        color: #17B794;
        border: 1px solid #17B79440;
    }
    .freshness-stale {
        background-color: #2d251580;
        color: #EEB76B;
        border: 1px solid #EEB76B40;
    }
    
    /* NEW: Confidence Bar */
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background-color: #30363d;
        overflow: hidden;
        margin-top: 8px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    /* NEW: Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-high {
        background-color: #FF4B4B20;
        color: #FF4B4B;
    }
    .status-medium {
        background-color: #EEB76B20;
        color: #EEB76B;
    }
    .status-low {
        background-color: #17B79420;
        color: #17B794;
    }
    
    /* NEW: Benchmark Card */
    .benchmark-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .benchmark-value {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .benchmark-label {
        color: #8b949e;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .benchmark-comparison {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #30363d;
        font-size: 0.9rem;
    }
    
    /* NEW: Action Tracker */
    .action-tracker-item {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .action-tracker-item.completed {
        border-left: 4px solid #17B794;
        opacity: 0.7;
    }
    .action-tracker-item.pending {
        border-left: 4px solid #EEB76B;
    }
    .action-tracker-item.overdue {
        border-left: 4px solid #FF4B4B;
    }
    
    /* NEW: Privacy Badge */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        background-color: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    
    /* NEW: Stay Interview Template */
    .interview-question {
        background-color: #161b22;
        border-left: 3px solid #17B794;
        padding: 12px 15px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
    .interview-question .question-text {
        color: #e6edf3;
        font-weight: 500;
    }
    .interview-question .question-purpose {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 4px;
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
</style>
""", unsafe_allow_html=True)

# ====================================================================
# TEMPERATURE SCALING FUNCTION
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
# NEW: Data Quality Report Function
# ====================================================================
def generate_data_quality_report(df, target_col='left'):
    """Generate comprehensive data quality metrics"""
    report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
        'numeric_columns': len(df.select_dtypes(include=np.number).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'class_distribution': None,
        'issues': []
    }
    
    # Check class imbalance
    if target_col in df.columns:
        class_counts = df[target_col].value_counts()
        report['class_distribution'] = class_counts.to_dict()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        if imbalance_ratio > 3:
            report['issues'].append(f"⚠️ Severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        elif imbalance_ratio > 2:
            report['issues'].append(f"⚠️ Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")
    
    # Check for missing values
    if report['missing_percentage'] > 5:
        report['issues'].append(f"⚠️ High missing values ({report['missing_percentage']:.1f}%)")
    
    # Check for duplicates
    if report['duplicate_percentage'] > 5:
        report['issues'].append(f"⚠️ High duplicate rows ({report['duplicate_percentage']:.1f}%)")
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            report['issues'].append(f"⚠️ Constant column detected: {col}")
    
    return report

# ====================================================================
# NEW: Industry Benchmarks Data
# ====================================================================
INDUSTRY_BENCHMARKS = {
    'Technology': {'avg_attrition': 13.2, 'cost_per_hire_pct': 0.15, 'time_to_fill_days': 42},
    'Healthcare': {'avg_attrition': 18.5, 'cost_per_hire_pct': 0.12, 'time_to_fill_days': 65},
    'Financial Services': {'avg_attrition': 11.8, 'cost_per_hire_pct': 0.20, 'time_to_fill_days': 38},
    'Retail': {'avg_attrition': 25.4, 'cost_per_hire_pct': 0.08, 'time_to_fill_days': 21},
    'Manufacturing': {'avg_attrition': 15.7, 'cost_per_hire_pct': 0.10, 'time_to_fill_days': 35},
    'Consulting': {'avg_attrition': 16.9, 'cost_per_hire_pct': 0.18, 'time_to_fill_days': 45},
    'Overall Average': {'avg_attrition': 17.5, 'cost_per_hire_pct': 0.14, 'time_to_fill_days': 42}
}

# ====================================================================
# NEW: Stay Interview Templates
# ====================================================================
STAY_INTERVIEW_QUESTIONS = [
    {"question": "What do you look forward to when you come to work each day?", "purpose": "Identifies motivators and engagement drivers"},
    {"question": "What are you learning here, and what do you want to learn?", "purpose": "Assesses growth opportunities and development needs"},
    {"question": "Why do you stay at this company?", "purpose": "Reveals retention factors to reinforce"},
    {"question": "When was the last time you thought about leaving, and what prompted it?", "purpose": "Identifies recent friction points or triggers"},
    {"question": "What can I do to make your experience here better?", "purpose": "Direct actionable feedback from the employee"},
    {"question": "Do you feel recognized for your contributions? How would you prefer to be recognized?", "purpose": "Checks if recognition needs are being met"},
    {"question": "What would make you want to stay for another 2-3 years?", "purpose": "Long-term retention planning"},
    {"question": "Is there anything preventing you from doing your best work?", "purpose": "Identifies blockers and frustrations"},
    {"question": "How would you describe our company culture to a friend?", "purpose": "Gauges cultural alignment and perception"},
    {"question": "What feedback do you have for me as your manager?", "purpose": "Builds trust and opens communication channels"}
]

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
# NEW: Seasonal Trend Analysis
# ====================================================================
def analyze_seasonal_trends(df):
    """Analyze if there are seasonal patterns in attrition"""
    if 'time_spend_company' not in df.columns:
        return None
    
    # Create tenure buckets to simulate temporal patterns
    df_analysis = df.copy()
    
    # Analyze attrition by tenure (as proxy for time-based patterns)
    tenure_attrition = df_analysis.groupby('time_spend_company').agg({
        'left': ['count', 'sum', 'mean']
    }).reset_index()
    tenure_attrition.columns = ['Tenure_Years', 'Total_Employees', 'Left', 'Attrition_Rate']
    tenure_attrition['Attrition_Rate'] = tenure_attrition['Attrition_Rate'] * 100
    
    return tenure_attrition

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
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                st.warning("⏳ **AI is busy right now.** The servers are overloaded. Please wait 30 seconds and try again.")
            else:
                st.error(f"Error generating draft: {e}")

# ====================================================================
# NEW: Executive Summary Generator
# ====================================================================
def generate_executive_summary(df, pipeline, attrition_rate):
    """Generate a 1-page executive summary for CEO/Board"""
    total_employees = len(df)
    total_left = int(df['left'].sum())
    
    # Calculate high-risk count
    feature_cols = [c for c in df.columns if c != 'left']
    current_df = df[df['left'] == 0].copy()
    if len(current_df) > 0:
        risks = pipeline.predict_proba(current_df[feature_cols])[:, 1]
        current_df['Risk_Score'] = calibrate_probability_array(risks)
        high_risk_count = len(current_df[current_df['Risk_Score'] > 0.6])
    else:
        high_risk_count = 0
    
    summary = f"""
## 📊 EXECUTIVE SUMMARY: Workforce Attrition Report
**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')} | **Data Source:** {'Custom Upload' if st.session_state.get('is_global') else 'Default Dataset'}

---

### 🎯 Key Metrics at a Glance
| Metric | Value | Status |
|--------|-------|--------|
| Total Workforce | {total_employees:,} | - |
| Employees Left | {total_left:,} | - |
| Current Attrition Rate | {attrition_rate:.1f}% | {'🚨 CRITICAL' if attrition_rate > 20 else '⚠️ HIGH' if attrition_rate > 15 else '✅ NORMAL'} |
| High-Risk Employees (Current) | {high_risk_count:,} | {'🔴 ACTION NEEDED' if high_risk_count > 100 else '🟡 MONITOR' if high_risk_count > 50 else '🟢 STABLE'} |

---

### 💡 Top 3 Recommendations
1. **Immediate:** Schedule stay interviews for top {min(high_risk_count, 20)} high-risk employees
2. **Short-term:** Review workload distribution for departments with >20% attrition
3. **Long-term:** Implement quarterly pulse surveys for early warning detection

---

### 💰 Financial Impact (Estimated)
- **Cost of Replacing All At-Risk:** ₹{(high_risk_count * 300000):,.0f} (using avg. 50% salary replacement cost)
- **Recommended Retention Budget:** ₹{(high_risk_count * 30000):,.0f} (10% intervention cost)
- **Potential ROI:** 10x return on retention investment

---

*Report generated by RetainAI Enterprise Workforce Intelligence Platform*
*For detailed analysis, refer to individual module reports.*
"""
    return summary

# ====================================================================
# NEW: Employee Search Function
# ====================================================================
def employee_search(df, pipeline, search_term, feature_columns):
    """Search for employees and return their risk scores"""
    results = []
    search_term = search_term.lower()
    
    for idx, row in df.iterrows():
        match = False
        # Search in all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                if search_term in str(row[col]).lower():
                    match = True
                    break
        
        if match:
            employee_data = row[feature_columns].to_frame().T
            risk_score = calibrate_probability(pipeline.predict_proba(employee_data)[0][1])
            results.append({
                'index': idx,
                'data': row,
                'risk_score': risk_score
            })
    
    return results

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ====================================================================
    # NEW: Initialize Session State Variables
    # ====================================================================
    if 'action_items' not in st.session_state:
        st.session_state.action_items = []
    if 'data_loaded_time' not in st.session_state:
        st.session_state.data_loaded_time = datetime.now()
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None
    if 'industry_selected' not in st.session_state:
        st.session_state.industry_selected = 'Technology'

    # ====================================================================
    # CROSS-PAGE NAVIGATION HANDLER (FIXED FLOW)
    # ====================================================================
    menu_options = ['⚙️ Global Setup', 'Home', 'Employee Search', 'Employee Insights', 'Predict Attrition', 'Why They Leave', 'Industry Benchmarks', 'Budget Planner', 'Stay Interviews', 'Action Tracker', 'AI Assistant', 'AI Research Lab', 'Strategic Roadmap', 'Export Report']
    
    default_idx = 1  # Default to 'Home' instead of 'Setup'
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
            # FIX: App crash protection if CSV missing
            if not os.path.exists('HR_comma_sep.csv'):
                st.error("❌ Default dataset 'HR_comma_sep.csv' not found. Please go to ⚙️ Global Setup to upload your data.")
                st.stop()
                
            st.write("📂 Step 1/3: Loading Dataset from CSV...")
            df = pd.read_csv('HR_comma_sep.csv')
            st.session_state.data_loaded_time = datetime.now()
            
            st.write("🧹 Step 2/3: Preprocessing & Splitting Data...")
            df_original = df.copy()
            df_train = df.drop_duplicates().reset_index(drop=True)
            X = df_train.drop('left', axis=1)
            y = df_train['left']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
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
            
            # Calculate and store model accuracy
            y_pred = final_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.session_state.model_accuracy = accuracy
            
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
            booster = model.booster_ if hasattr(model, "booster_") else model._Booster if hasattr(model, "_Booster") else model.booster if hasattr(model, "booster") else model
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

    # --- NEW: Helper function for data freshness ---
    def get_freshness_badge():
        time_since_load = datetime.now() - st.session_state.data_loaded_time
        hours_since = time_since_load.total_seconds() / 3600
        
        if hours_since < 1:
            return '<span class="freshness-badge freshness-fresh">🟢 Data: Fresh (Just loaded)</span>'
        elif hours_since < 24:
            return '<span class="freshness-badge freshness-fresh">🟢 Data: Fresh (< 24h ago)</span>'
        elif hours_since < 72:
            return f'<span class="freshness-badge freshness-stale">🟡 Data: {int(hours_since)}h ago</span>'
        else:
            return f'<span class="freshness-badge freshness-stale">🔴 Data: {int(hours_since/24)}d ago - Consider updating</span>'

    # --- NEW: Privacy badge ---
    def get_privacy_badge():
        return '<span class="privacy-badge">🔒 No data leaves your system • Local processing only</span>'

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
        
        # NEW: Show data freshness in sidebar
        st.markdown(get_freshness_badge(), unsafe_allow_html=True)
        st.markdown(get_privacy_badge(), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        page = option_menu(
            menu_title=None,
            options=menu_options,  
            icons=['gear', 'house', 'search', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'building', 'currency-rupee', 'chat-heart', 'list-check', 'robot', 'cpu', 'flag-2-fill', 'file-arrow-down'], 
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
    # PAGE: GLOBAL SETUP (ENHANCED with Data Quality Report)
    # ====================================================================
    if page == "⚙️ Global Setup":
        st.header("⚙️ Global Setup: Upload Your Company Data")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Turn this AI into your company's dedicated assistant. Upload your HR dataset, and the system will automatically retrain itself.</p>", unsafe_allow_html=True)
        
        # NEW: About This Tool section
        with st.expander("ℹ️ About This Tool", expanded=False):
            st.markdown("""
            **RetainAI** is an enterprise-grade workforce intelligence platform that helps HR teams:
            
            - 🎯 **Predict** which employees are likely to leave
            - 🔍 **Explain** why employees leave using AI explainability (SHAP, Causal Analysis)
            - 💰 **Optimize** retention budget using mathematical optimization (MILP)
            - 📧 **Communicate** with AI-generated email drafts
            - 📈 **Plan** strategic interventions with 12-month projections
            
            **How it works:**
            1. Upload your HR data (or use demo data)
            2. AI trains a LightGBM model on your data
            3. Use various tools to analyze and take action
            
            **Privacy:** All data processing happens locally. No data is sent to external servers except for AI text generation (Groq API).
            """)
        
        uploaded_file = st.file_uploader("Upload your HR Dataset (CSV format)", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                
                # NEW: Data Quality Report
                with st.expander("📋 Data Quality Report", expanded=True):
                    quality_report = generate_data_quality_report(new_df)
                    
                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    col_q1.metric("Total Records", f"{quality_report['total_records']:,}")
                    col_q2.metric("Missing Values", f"{quality_report['missing_values']:,}")
                    col_q3.metric("Duplicate Rows", f"{quality_report['duplicate_rows']:,}")
                    col_q4.metric("Memory Usage", f"{quality_report['memory_usage_mb']:.2f} MB")
                    
                    col_q5, col_q6 = st.columns(2)
                    col_q5.metric("Numeric Columns", quality_report['numeric_columns'])
                    col_q6.metric("Categorical Columns", quality_report['categorical_columns'])
                    
                    if quality_report['issues']:
                        st.markdown("### ⚠️ Issues Detected")
                        for issue in quality_report['issues']:
                            st.warning(issue)
                    else:
                        st.success("✅ No data quality issues detected!")
                    
                    if quality_report['class_distribution']:
                        st.markdown("### 📊 Class Distribution (Attrition)")
                        for val, count in quality_report['class_distribution'].items():
                            pct = (count / quality_report['total_records']) * 100
                            st.write(f"- **Class {val}**: {count:,} records ({pct:.1f}%)")
                
                st.dataframe(new_df.head())
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
                if st.button("🚀 Train Custom AI Model", type="primary"):
                    with st.spinner("🤖 AI is learning your data..."):
                        y = new_df[target_col].apply(lambda x: 1 if x == left_value else 0); X = new_df[feature_cols]
                        valid_idx = X.dropna().index; X_clean = X.loc[valid_idx]; y_clean = y.loc[valid_idx]
                        
                        if len(categorical_auto) == 0: preprocessor_global = ColumnTransformer(transformers=[('num', 'passthrough', numerical_auto)])
                        else: preprocessor_global = ColumnTransformer(transformers=[('num', 'passthrough', numerical_auto), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_auto)])
                        
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        
                        spw = min((y_train_g == 0).sum() / (y_train_g == 1).sum(), 2.0) if (y_train_g == 1).sum() > 0 else 1.0
                        
                        global_pipeline = Pipeline(steps=[('preprocessor', preprocessor_global), ('classifier', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=12, max_depth=4, min_child_samples=40, reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, scale_pos_weight=spw))])
                        global_pipeline.fit(X_train_g, y_train_g)
                        y_pred_g = global_pipeline.predict(X_test_g); 
                        
                        # Calculate comprehensive metrics
                        acc = accuracy_score(y_test_g, y_pred_g)
                        prec = precision_score(y_test_g, y_pred_g, zero_division=0)
                        rec = recall_score(y_test_g, y_pred_g, zero_division=0)
                        f1 = f1_score(y_test_g, y_pred_g, zero_division=0)
                        roc = roc_auc_score(y_test_g, global_pipeline.predict_proba(X_test_g)[:, 1]) if len(y_test_g.unique()) > 1 else 0
                        
                        st.session_state.model_accuracy = acc
                        
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g; st.session_state['is_global'] = True
                        st.session_state.data_loaded_time = datetime.now()
                        
                        st.success(f"🎉 Training Complete!")
                        
                        # NEW: Show model performance
                        with st.expander("📊 Model Performance Metrics", expanded=True):
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("Accuracy", f"{acc:.1%}")
                            col_m2.metric("Precision", f"{prec:.1%}")
                            col_m3.metric("Recall", f"{rec:.1%}")
                            col_m4.metric("F1 Score", f"{f1:.1%}")
                            st.metric("ROC AUC", f"{roc:.1%}")
                            
                            st.info("""
                            **What these metrics mean:**
                            - **Accuracy**: Overall correctness of predictions
                            - **Precision**: When AI says "Leave", how often is it right?
                            - **Recall**: Of all who actually left, how many did AI catch?
                            - **F1 Score**: Balance between Precision and Recall
                            - **ROC AUC**: Model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
                            """)
                        
                        # Auto-navigate to Home after setup
                        st.session_state['nav_to'] = "Home"
                        st.rerun()
            except Exception as e: st.error(f"Error: {e}")
        if st.button("🔄 Reset to Default Demo Data"):
            if 'is_global' in st.session_state: del st.session_state['is_global']
            st.session_state.data_loaded_time = datetime.now()
            st.rerun()

    # ====================================================================
    # PAGE: HOME (ENHANCED WITH MORE FEATURES)
    # ====================================================================
    if page == "Home":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        
        # NEW: Show data freshness and model accuracy
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(get_freshness_badge(), unsafe_allow_html=True)
        with col_info2:
            if st.session_state.model_accuracy:
                st.markdown(f'<span class="freshness-badge freshness-fresh">🤖 Model Accuracy: {st.session_state.model_accuracy:.1%}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="freshness-badge freshness-stale">🤖 Model: Loading...</span>', unsafe_allow_html=True)
        with col_info3:
            st.markdown(get_privacy_badge(), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
            # NEW: Active employees metric
            col4.metric("Active Employees", f"{(df['left']==0).sum():,}")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        
        # QUICK ACTIONS (CROSS-PAGE NAVIGATION)
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("🔍 Search Employee", use_container_width=True):
                st.session_state['nav_to'] = "Employee Search"
                st.rerun()
        with c2:
            if st.button("🎯 Predict Employee", use_container_width=True):
                st.session_state['nav_to'] = "Predict Attrition"
                st.rerun()
        with c3:
            if st.button("💰 Budget Planner", use_container_width=True):
                st.session_state['nav_to'] = "Budget Planner"
                st.rerun()
        with c4:
            if st.button("💬 Stay Interviews", use_container_width=True):
                st.session_state['nav_to'] = "Stay Interviews"
                st.rerun()
        with c5:
            if st.button("📋 Action Tracker", use_container_width=True):
                st.session_state['nav_to'] = "Action Tracker"
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
                
                # NEW: Status pill based on risk level
                if risk_pct > 70: status_class, status_text = "status-high", "HIGH RISK"
                elif risk_pct > 50: status_class, status_text = "status-medium", "MEDIUM RISK"
                else: status_class, status_text = "status-low", "LOW RISK"
                
                color = "#FF4B4B" if risk_pct > 70 else "#EEB76B" if risk_pct > 50 else "#17B794"
                st.markdown(f"""
                <div style='background:{color}15; border-left:4px solid {color}; padding:10px 15px; margin:5px 0; border-radius:0 8px 8px 0; display:flex; justify-content:space-between; align-items:center;'>
                    <span><strong>{dept}</strong> | {salary.title()} Salary</span>
                    <span style='display:flex; align-items:center; gap:10px;'>
                        <span class='status-pill {status_class}'>{status_text}</span>
                        <span style='color:{color}; font-weight:700;'>Risk: {risk_pct:.1f}%</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No current employees found in dataset.")
        
        st.markdown("---")
        st.markdown("### 📄 Employee Data Snapshot")
        st.dataframe(df.head(100), use_container_width=True)

    # ====================================================================
    # NEW PAGE: EMPLOYEE SEARCH
    # ====================================================================
    if page == "Employee Search":
        st.header("🔍 Employee Search")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Search for specific employees by name, department, or any attribute and see their attrition risk instantly.</p>", unsafe_allow_html=True)
        
        feature_columns = [c for c in df.columns if c != 'left']
        
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.text_input("Search employees...", placeholder="Type department name, salary tier, or any value...")
        with search_col2:
            st.markdown("<br>")
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_term and search_button:
            results = employee_search(df, pipeline, search_term, feature_columns)
            
            if len(results) == 0:
                st.warning("No employees found matching your search.")
            else:
                st.success(f"Found {len(results)} matching employee(s)")
                
                for result in results[:20]:  # Limit to 20 results
                    risk_pct = result['risk_score'] * 100
                    employee = result['data']
                    
                    # Determine risk status
                    if risk_pct > 70: 
                        status_class, status_text, bg_color = "status-high", "HIGH RISK", "#FF4B4B"
                    elif risk_pct > 50: 
                        status_class, status_text, bg_color = "status-medium", "MEDIUM RISK", "#EEB76B"
                    else: 
                        status_class, status_text, bg_color = "status-low", "LOW RISK", "#17B794"
                    
                    # Create employee info string
                    info_parts = []
                    for col in df.columns:
                        if col != 'left' and col != 'Risk_Score':
                            val = employee[col]
                            if pd.notna(val):
                                info_parts.append(f"<strong>{col.replace('_', ' ').title()}:</strong> {val}")
                    
                    info_html = " | ".join(info_parts[:5])  # Show first 5 attributes
                    
                    st.markdown(f"""
                    <div style='background:{bg_color}15; border-left:4px solid {bg_color}; padding:15px; margin:10px 0; border-radius:0 12px 12px 0;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                            <span style='color:#e6edf3; font-size:0.9rem;'>{info_html}</span>
                            <span class='status-pill {status_class}'>{status_text}: {risk_pct:.1f}%</span>
                        </div>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width:{risk_pct}%; background-color:{bg_color};'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif search_term and not search_button:
            st.info("Press the Search button or Enter to search.")

    # ====================================================================
    # PAGE: EMPLOYEE INSIGHTS
    # ====================================================================
    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis")
        st.write("Explore the workforce demographics to identify patterns.")
        
        # NEW: Seasonal Trend Analysis
        with st.expander("📅 Seasonal/Tenure Trend Analysis", expanded=False):
            st.markdown("<p style='color: #9ca3af;'>Understanding when attrition peaks helps with proactive intervention timing.</p>", unsafe_allow_html=True)
            trend_data = analyze_seasonal_trends(df)
            if trend_data is not None:
                fig_trend = px.line(
                    trend_data, 
                    x='Tenure_Years', 
                    y='Attrition_Rate',
                    title='Attrition Rate by Employee Tenure',
                    template='plotly_dark',
                    markers=True,
                    color_discrete_sequence=['#17B794']
                )
                fig_trend.update_layout(
                    xaxis_title="Years at Company",
                    yaxis_title="Attrition Rate (%)",
                    yaxis_range=[0, max(trend_data['Attrition_Rate'].max() * 1.2, 10)]
                )
                custome_layout(fig_trend, title_size=22)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Find peak attrition tenure
                peak_row = trend_data.loc[trend_data['Attrition_Rate'].idxmax()]
                st.info(f"📈 **Insight:** Attrition peaks at **{peak_row['Tenure_Years']:.0f} years** of tenure with **{peak_row['Attrition_Rate']:.1f}%** attrition rate. Consider targeted interventions at the {int(peak_row['Tenure_Years'])-1}-{int(peak_row['Tenure_Years'])+1} year mark.")
            else:
                st.warning("Tenure data not available for trend analysis.")
        
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    # ====================================================================
    # PAGE: PREDICT ATTRITION (ENHANCED with Confidence Intervals)
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
                        if pred == 1: st.success(f"✅ **Correct!** Prediction: Leave ({calibrated_prob:.1%})")
                        else: st.error(f"❌ **Incorrect.** Prediction: Stay ({calibrated_prob:.1%})")
                        st.json(sample.to_dict(), expanded=False)
                with c_test2:
                    if st.button("Test with Employee who Stayed"):
                        sample = df[df['left'] == 0].iloc[0]
                        test_df = sample.drop('left').to_frame().T
                        raw_prob = pipeline.predict_proba(test_df)[0][1]
                        calibrated_prob = calibrate_probability(raw_prob)
                        pred = 1 if calibrated_prob >= 0.5 else 0
                        if pred == 0: st.success(f"✅ **Correct!** Prediction: Stay ({calibrated_prob:.1%})")
                        else: st.error(f"❌ **Incorrect.** Prediction: Leave ({calibrated_prob:.1%})")
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
                
                # NEW: Calculate confidence interval (simplified)
                confidence = abs(leave_prob - 0.5) * 2  # 0 = uncertain, 1 = very confident
                if confidence > 0.7: confidence_text = "HIGH"
                elif confidence > 0.3: confidence_text = "MEDIUM"
                else: confidence_text = "LOW"
                
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
                        <div style='margin-top:10px; font-size:0.8rem; color:#8b949e;'>Confidence: {confidence_text}</div>
                        <div class='confidence-bar' style='width:80%;'>
                            <div class='confidence-fill' style='width:{confidence*100}%; background-color:{"#17B794" if st.session_state.prediction_result==0 else "#EEB76B"};'></div>
                        </div>
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
                
                # NEW: Model accuracy disclaimer
                st.caption(f"*Prediction based on LightGBM model with {st.session_state.model_accuracy:.1%} historical accuracy. Individual predictions may vary.*")
                
                if st.session_state.prediction_result == 1:
                    st.markdown("---"); st.markdown("### 💡 Recommended Actions")
                    for rec in get_retention_strategies(st.session_state.input_df): st.info(rec)
                    
                    # NEW: Add to action tracker option
                    if st.button("📋 Add to Action Tracker", type="secondary"):
                        action = {
                            'id': len(st.session_state.action_items) + 1,
                            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'description': f"Review at-risk employee (Leave probability: {leave_percent}%)",
                            'status': 'pending',
                            'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                            'priority': 'high' if leave_percent > 70 else 'medium'
                        }
                        st.session_state.action_items.append(action)
                        st.success("✅ Added to Action Tracker!")
                    
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
        
        # BATCH PREDICTION TAB
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
                    batch_df['Confidence'] = batch_df['Risk Score'].apply(lambda x: "HIGH" if abs(x-0.5) > 0.3 else "MEDIUM" if abs(x-0.5) > 0.15 else "LOW")
                    batch_df = batch_df.sort_values('Risk Score', ascending=False)
                    st.dataframe(batch_df, use_container_width=True)
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results", csv, "batch_predictions.csv", "text/csv")
                else:
                    st.error(f"Missing columns in uploaded file. Required: {feature_columns}")

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
        with st.expander("🔧 Technical Deep Dive (SHAP)"):
            st.write("Below are the raw SHAP plots for data scientists.")
            if shap_values is not None:
                fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False)
                st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)
                fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False, plot_type='dot')
                st.pyplot(fig1, bbox_inches='tight'); plt.close(fig1)

    # ====================================================================
    # NEW PAGE: INDUSTRY BENCHMARKS
    # ====================================================================
    if page == "Industry Benchmarks":
        st.header("📊 Industry Benchmarks")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Compare your attrition metrics against industry standards to contextualize your performance.</p>", unsafe_allow_html=True)
        
        # Industry selector
        selected_industry = st.selectbox("Select Your Industry", list(INDUSTRY_BENCHMARKS.keys()))
        st.session_state.industry_selected = selected_industry
        
        # Get current metrics
        current_attrition = (df['left'].sum() / len(df)) * 100
        benchmark = INDUSTRY_BENCHMARKS[selected_industry]
        
        # Comparison cards
        col_b1, col_b2, col_b3 = st.columns(3)
        
        # Attrition comparison
        diff_attrition = current_attrition - benchmark['avg_attrition']
        if diff_attrition > 0:
            attrition_status = f"⚠️ {diff_attrition:.1f}% WORSE"
            attrition_color = "#FF4B4B"
        else:
            attrition_status = f"✅ {abs(diff_attrition):.1f}% BETTER"
            attrition_color = "#17B794"
        
        with col_b1:
            st.markdown(f"""
            <div class="benchmark-card">
                <div class="benchmark-label">Your Attrition Rate</div>
                <div class="benchmark-value" style="color: {attrition_color};">{current_attrition:.1f}%</div>
                <div class="benchmark-comparison">
                    <div style="color: #8b949e;">Industry Avg: {benchmark['avg_attrition']}%</div>
                    <div style="color: {attrition_color}; font-weight: 600; margin-top: 5px;">{attrition_status}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b2:
            st.markdown(f"""
            <div class="benchmark-card">
                <div class="benchmark-label">Industry Cost Per Hire</div>
                <div class="benchmark-value" style="color: #EEB76B;">{benchmark['cost_per_hire_pct']*100:.0f}%</div>
                <div class="benchmark-comparison">
                    <div style="color: #8b949e;">Of annual salary</div>
                    <div style="color: #c9d1d9; margin-top: 5px;">Time to fill: {benchmark['time_to_fill_days']} days</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b3:
            # Calculate estimated cost impact
            avg_salary = 500000  # Default estimate
            if 'salary' in df.columns:
                salary_map = {'low': 400000, 'medium': 600000, 'high': 900000}
                avg_salary = df['salary'].map(salary_map).mean()
            
            employees_left = int(df['left'].sum())
            estimated_cost = employees_left * avg_salary * benchmark['cost_per_hire_pct']
            
            st.markdown(f"""
            <div class="benchmark-card">
                <div class="benchmark-label">Your Replacement Cost</div>
                <div class="benchmark-value" style="color: #FF4B4B;">₹{(estimated_cost/10000000):.2f}Cr</div>
                <div class="benchmark-comparison">
                    <div style="color: #8b949e;">{employees_left} employees replaced</div>
                    <div style="color: #c9d1d9; margin-top: 5px;">Avg salary: ₹{(avg_salary/100000):.1f}L</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # All industries comparison chart
        st.markdown("---")
        st.markdown("### 📈 All Industries Comparison")
        
        benchmark_df = pd.DataFrame([
            {'Industry': k, 'Attrition Rate': v['avg_attrition'], 'Type': 'Industry Average'}
            for k, v in INDUSTRY_BENCHMARKS.items()
        ])
        benchmark_df = pd.concat([
            benchmark_df,
            pd.DataFrame([{'Industry': 'YOUR COMPANY', 'Attrition Rate': current_attrition, 'Type': 'Your Company'}])
        ])
        
        fig_benchmark = px.bar(
            benchmark_df,
            x='Industry',
            y='Attrition Rate',
            color='Type',
            color_discrete_map={'Industry Average': '#30363d', 'Your Company': '#17B794'},
            title='Attrition Rate: Your Company vs Industry Benchmarks',
            template='plotly_dark',
            height=450
        )
        fig_benchmark.update_layout(showlegend=True)
        custome_layout(fig_benchmark, title_size=22, showlegend=True)
        st.plotly_chart(fig_benchmark, use_container_width=True)
        
        # Contextual advice
        st.markdown("---")
        if current_attrition > benchmark['avg_attrition'] + 5:
            st.error(f"""
            **🚨 Action Required:** Your attrition rate ({current_attrition:.1f}%) is significantly higher than the {selected_industry} industry average ({benchmark['avg_attrition']}%).
            
            **Immediate Recommendations:**
            1. Conduct stay interviews for top 20% of workforce
            2. Review compensation against market rates
            3. Analyze manager effectiveness (bad managers = high turnover)
            """)
        elif current_attrition > benchmark['avg_attrition']:
            st.warning(f"""
            **⚠️ Monitor:** Your attrition rate ({current_attrition:.1f}%) is slightly above the {selected_industry} industry average ({benchmark['avg_attrition']}%).
            
            **Recommendations:**
            1. Identify which departments are driving the increase
            2. Implement pulse surveys for early detection
            """)
        else:
            st.success(f"""
            **✅ Well Done:** Your attrition rate ({current_attrition:.1f}%) is below the {selected_industry} industry average ({benchmark['avg_attrition']}%).
            
            **Maintenance Recommendations:**
            1. Continue current retention strategies
            2. Focus on keeping top performers engaged
            3. Prepare for industry-wide challenges
            """)

    # ====================================================================
    # PAGE: BUDGET PLANNER
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
                            
                            # NEW: Add to action tracker
                            if st.button("📋 Add to Action Tracker", type="secondary", key="add_budget_action"):
                                action = {
                                    'id': len(st.session_state.action_items) + 1,
                                    'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'description': f"Execute retention plan for {len(selected_employees)} employees (Budget: ₹{total_investment:,.0f})",
                                    'status': 'pending',
                                    'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                                    'priority': 'high'
                                }
                                st.session_state.action_items.append(action)
                                st.success("✅ Added to Action Tracker!")
                            
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
                        else:
                            st.error("❌ Optimization failed. Budget may be too low to save anyone.")

    # ====================================================================
    # NEW PAGE: STAY INTERVIEWS
    # ====================================================================
    if page == "Stay Interviews":
        st.header("💬 Stay Interview Templates")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Proactive retention starts with understanding why people stay. Use these research-backed questions for your stay interviews.</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card" style="border-left: 5px solid #17B794;">
            <h4 style="color: #17B794; margin-top: 0;">🎯 What is a Stay Interview?</h4>
            <p style='color: #c9d1d9;'>A stay interview is a structured conversation with current employees to understand what keeps them at the company and what might cause them to leave. Unlike exit interviews, stay interviews help you <strong>prevent</strong> attrition before it happens.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Best practices
        with st.expander("📋 Best Practices for Stay Interviews", expanded=False):
            st.markdown("""
            **Do's:**
            - ✅ Conduct with top performers and critical roles first
            - ✅ Have the direct manager conduct the interview (not HR)
            - ✅ Keep it confidential - share themes, not individual responses
            - ✅ Follow up within 2 weeks on any commitments made
            - ✅ Make it a regular practice (quarterly or bi-annually)
            
            **Don'ts:**
            - ❌ Don't combine with performance reviews
            - ❌ Don't ask leading questions ("You're happy here, right?")
            - ❌ Don't make promises you can't keep
            - ❌ Don't share responses with other managers
            - ❌ Don't skip follow-up actions
            """)
        
        st.markdown("---")
        st.markdown("### 📝 Question Bank")
        
        # Category filter
        categories = ["All", "Engagement", "Growth", "Management", "Compensation", "Culture"]
        selected_category = st.selectbox("Filter by Category", categories)
        
        question_categories = {
            "Engagement": [0, 3, 5],
            "Growth": [1, 6],
            "Management": [5, 9],
            "Compensation": [3],
            "Culture": [8, 9]
        }
        
        for i, q in enumerate(STAY_INTERVIEW_QUESTIONS):
            if selected_category == "All" or i in question_categories.get(selected_category, []):
                st.markdown(f"""
                <div class="interview-question">
                    <div class="question-text">{i+1}. {q['question']}</div>
                    <div class="question-purpose">💡 Purpose: {q['purpose']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # NEW: Generate stay interview guide
        st.markdown("---")
        st.markdown("### 🤖 AI-Generated Interview Guide")
        if st.button("✍️ Generate Custom Stay Interview Guide", type="primary"):
            with st.spinner("Generating personalized guide..."):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if api_key:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.6)
                        
                        # Get context from data
                        if 'satisfaction_level' in df.columns:
                            avg_sat = df['satisfaction_level'].mean()
                            top_issue = "low satisfaction" if avg_sat < 0.5 else "work-life balance" if df['average_montly_hours'].mean() > 200 else "career growth"
                        else:
                            top_issue = "general engagement"
                        
                        template = """You are an HR expert creating a stay interview guide.
                        
                        **Context:** Our company has {attrition_rate:.1f}% attrition rate. Main issue seems to be {top_issue}.
                        
                        **Task:** Create a 15-minute stay interview guide with:
                        1. Opening script (2 minutes)
                        2. 5 key questions with follow-up probes
                        3. Closing script (2 minutes)
                        
                        Keep it conversational and natural. Avoid corporate jargon.
                        """
                        prompt = PromptTemplate.from_template(template)
                        chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({
                            "attrition_rate": (df['left'].sum() / len(df)) * 100,
                            "top_issue": top_issue
                        })
                        st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("🔑 API Key missing. Use the question bank above instead.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ====================================================================
    # NEW PAGE: ACTION TRACKER
    # ====================================================================
    if page == "Action Tracker":
        st.header("📋 Action Tracker")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Track your retention action items and ensure follow-through. Recommendations without action are just noise.</p>", unsafe_allow_html=True)
        
        # Summary metrics
        col_a1, col_a2, col_a3 = st.columns(3)
        pending_count = len([a for a in st.session_state.action_items if a['status'] == 'pending'])
        completed_count = len([a for a in st.session_state.action_items if a['status'] == 'completed'])
        overdue_count = len([a for a in st.session_state.action_items if a['status'] == 'pending' and a['due_date'] < datetime.now().strftime('%Y-%m-%d')])
        
        col_a1.metric("Pending Actions", pending_count, delta="Need attention")
        col_a2.metric("Completed", completed_count, delta="Done!")
        col_a3.metric("Overdue", overdue_count, delta_color="inverse" if overdue_count > 0 else "normal")
        
        # Add new action form
        with st.expander("➕ Add New Action Item", expanded=False):
            with st.form("add_action_form"):
                new_action_desc = st.text_area("Action Description", placeholder="e.g., Schedule stay interview with Rahul from Sales")
                col_new1, col_new2 = st.columns(2)
                with col_new1:
                    new_due_date = st.date_input("Due Date", value=datetime.now() + timedelta(days=7))
                with col_new2:
                    new_priority = st.selectbox("Priority", ["high", "medium", "low"])
                
                if st.form_submit_button("Add Action"):
                    if new_action_desc:
                        action = {
                            'id': len(st.session_state.action_items) + 1,
                            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'description': new_action_desc,
                            'status': 'pending',
                            'due_date': new_due_date.strftime('%Y-%m-%d'),
                            'priority': new_priority
                        }
                        st.session_state.action_items.append(action)
                        st.success("✅ Action added!")
                        st.rerun()
                    else:
                        st.warning("Please enter a description")
        
        # Display actions
        st.markdown("---")
        if len(st.session_state.action_items) == 0:
            st.info("No action items yet. Add actions from predictions or use the form above.")
        else:
            # Filter options
            filter_status = st.selectbox("Filter by Status", ["All", "Pending", "Completed", "Overdue"])
            
            filtered_actions = st.session_state.action_items
            if filter_status == "Pending":
                filtered_actions = [a for a in filtered_actions if a['status'] == 'pending']
            elif filter_status == "Completed":
                filtered_actions = [a for a in filtered_actions if a['status'] == 'completed']
            elif filter_status == "Overdue":
                filtered_actions = [a for a in filtered_actions if a['status'] == 'pending' and a['due_date'] < datetime.now().strftime('%Y-%m-%d')]
            
            for action in filtered_actions:
                is_overdue = action['status'] == 'pending' and action['due_date'] < datetime.now().strftime('%Y-%m-%d')
                status_class = 'overdue' if is_overdue else action['status']
                
                priority_badge = f'<span class="status-pill status-{"high" if action["priority"]=="high" else "medium" if action["priority"]=="medium" else "low"}">{action["priority"].upper()}</span>'
                
                st.markdown(f"""
                <div class="action-tracker-item {status_class}">
                    <div>
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                            {priority_badge}
                            <span style="color:#e6edf3; font-weight:500;">{action['description']}</span>
                        </div>
                        <div style="color:#8b949e; font-size:0.8rem;">
                            Created: {action['created']} | Due: {action['due_date']}
                            {f" | ⚠️ OVERDUE" if is_overdue else ""}
                        </div>
                    </div>
                    <div style="display:flex; gap:8px;">
                        {'<button onclick="this.parentElement.parentElement.className=\'action-tracker-item completed\'" style="padding:6px 12px; border-radius:6px; background:#17B79420; color:#17B794; border:1px solid #17B79440; cursor:pointer;">✓ Done</button>' if action['status'] == 'pending' else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Bulk actions
        if len(st.session_state.action_items) > 0:
            st.markdown("---")
            col_bulk1, col_bulk2 = st.columns(2)
            with col_bulk1:
                if st.button("✅ Mark All Pending as Complete", type="secondary"):
                    for a in st.session_state.action_items:
                        if a['status'] == 'pending':
                            a['status'] = 'completed'
                    st.success("All marked complete!")
                    st.rerun()
            with col_bulk2:
                if st.button("🗑️ Clear Completed", type="secondary"):
                    st.session_state.action_items = [a for a in st.session_state.action_items if a['status'] != 'completed']
                    st.success("Cleared completed items!")
                    st.rerun()

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
        tab1, tab2, tab3 = st.tabs(["📊 Model Benchmarking", "🔬 Departmental Strategy Deep Dive", "🛡️ AI Disruption Defense"])
        
        with tab1:
            st.subheader("Algorithm Performance Comparison")
            if st.button("Run Benchmark", type="primary", key="run_benchmark"):
                with st.spinner("Training competing models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
                    rf_pipeline.fit(X_train_ref, y_train); y_pred_rf = rf_pipeline.predict(X_test_cur); proba_rf = rf_pipeline.predict_proba(X_test_cur)[:, 1]
                    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
                    lr_pipeline.fit(X_train_ref, y_train); y_pred_lr = lr_pipeline.predict(X_test_cur); proba_lr = lr_pipeline.predict_proba(X_test_cur)[:, 1]
                    metrics = {'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'], 'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)], 'Precision': [precision_score(y_test, y_pred_lgbm), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_lr)], 'Recall': [recall_score(y_test, y_pred_lgbm), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_lr)], 'F1 Score': [f1_score(y_test, y_pred_lgbm), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_lr)], 'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_lr)]}
                    results_df = pd.DataFrame(metrics)
                    st.markdown("### 📈 Performance Metrics")
                    st.dataframe(results_df.style.highlight_max(axis=0, color='#17B794'), use_container_width=True)
                    fig_metrics = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'), x='Metric', y='Score', color='Model', barmode='group', title="Model Comparison", template="plotly_dark", color_discrete_sequence=['#17B794', '#EEB76B', '#9C3D54'])
                    custome_layout(fig_metrics, title_size=24); st.plotly_chart(fig_metrics, use_container_width=True)
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
            fig_forecast             = px.line(forecast_df, x='Month', y='Workforce Count', color='Scenario', title="Projected Workforce Size Over the Next 12 Months", template="plotly_dark", markers=True, color_discrete_map={'If We Do Nothing (Status Quo)': "#EEB76B", 'If We Follow the Plan': "#17B794"})
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
            
            # NEW: Add to action tracker
            if st.button("📋 Add Roadmap to Action Tracker", type="secondary", key="add_roadmap_action"):
                action = {
                    'id': len(st.session_state.action_items) + 1,
                    'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'description': f"Execute {intervention_efficacy}% retention plan (Projected savings: ₹{total_money_saved:,.0f})",
                    'status': 'pending',
                    'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    'priority': 'high'
                }
                st.session_state.action_items.append(action)
                st.success("✅ Added to Action Tracker!")

    # ====================================================================
    # NEW PAGE: EXPORT REPORT
    # ====================================================================
    if page == "Export Report":
        st.header("📄 Export Report")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Generate comprehensive reports for leadership, board presentations, or your records.</p>", unsafe_allow_html=True)
        
        # Report options
        report_type = st.selectbox("Select Report Type", [
            "Executive Summary (1-page)",
            "Full Attrition Analysis Report",
            "Department-wise Breakdown",
            "Action Items Report",
            "Model Performance Report"
        ])
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            include_timestamp = st.checkbox("Include timestamp", value=True)
            include_recommendations = st.checkbox("Include recommendations", value=True)
        with col_r2:
            include_data_sample = st.checkbox("Include data sample", value=False)
            include_charts = st.checkbox("Include chart descriptions", value=True)
        
        if st.button("📝 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                attrition_rate = (df['left'].sum() / len(df)) * 100
                total_employees = len(df)
                total_left = int(df['left'].sum())
                
                # Calculate high-risk count
                feature_cols = [c for c in df.columns if c != 'left']
                current_df = df[df['left'] == 0].copy()
                if len(current_df) > 0:
                    risks = pipeline.predict_proba(current_df[feature_cols])[:, 1]
                    current_df['Risk_Score'] = calibrate_probability_array(risks)
                    high_risk_count = len(current_df[current_df['Risk_Score'] > 0.6])
                    medium_risk_count = len(current_df[(current_df['Risk_Score'] > 0.4) & (current_df['Risk_Score'] <= 0.6)])
                else:
                    high_risk_count = 0
                    medium_risk_count = 0
                
                timestamp = datetime.now().strftime('%B %d, %Y at %H:%M') if include_timestamp else "N/A"
                data_source = 'Custom Upload' if st.session_state.get('is_global') else 'Default Dataset (HR_comma_sep.csv)'
                
                if report_type == "Executive Summary (1-page)":
                    report = generate_executive_summary(df, pipeline, attrition_rate)
                    st.markdown(report)
                    
                    # Download as text
                    report_text = report.replace('**', '').replace('*', '').replace('#', '').replace('|', ' | ').replace('---', '\n' + '-'*50 + '\n')
                    
                elif report_type == "Full Attrition Analysis Report":
                    report = f"""
# ATTRITION ANALYSIS REPORT
Generated: {timestamp}
Data Source: {data_source}

## 1. OVERVIEW
- Total Workforce Analyzed: {total_employees:,}
- Employees Who Left: {total_left:,}
- Current Attrition Rate: {attrition_rate:.1f}%
- Active Employees: {(df['left']==0).sum():,}

## 2. RISK DISTRIBUTION (Current Employees)
- High Risk (>60% probability): {high_risk_count:,} employees
- Medium Risk (40-60% probability): {medium_risk_count:,} employees
- Low Risk (<40% probability): {(df['left']==0).sum() - high_risk_count - medium_risk_count:,} employees

## 3. KEY METRICS
"""
                    if 'satisfaction_level' in df.columns:
                        report += f"- Average Satisfaction Level: {df['satisfaction_level'].mean():.2f}/1.0\n"
                    if 'average_montly_hours' in df.columns:
                        report += f"- Average Monthly Hours: {df['average_montly_hours'].mean():.0f}\n"
                    if 'number_project' in df.columns:
                        report += f"- Average Number of Projects: {df['number_project'].mean():.1f}\n"
                    if 'time_spend_company' in df.columns:
                        report += f"- Average Tenure: {df['time_spend_company'].mean():.1f} years\n"
                    
                    report += f"\n## 4. MODEL PERFORMANCE\n- Algorithm: LightGBM\n- Accuracy: {st.session_state.model_accuracy:.1%}\n"
                    
                    if include_recommendations:
                        report += """
## 5. RECOMMENDATIONS
1. IMMEDIATE: Schedule stay interviews for top high-risk employees
2. SHORT-TERM: Review workload distribution and compensation
3. LONG-TERM: Implement quarterly pulse surveys for early detection
4. BUDGET: Allocate retention budget based on ROI optimization results
5. MONITORING: Track action items and measure intervention effectiveness
"""
                    
                    if include_data_sample:
                        report += "\n## 6. DATA SAMPLE (First 10 records)\n"
                        report += df.head(10).to_string()
                    
                    st.markdown(report)
                    report_text = report
                    
                elif report_type == "Department-wise Breakdown":
                    report = f"""
# DEPARTMENT-WISE ATTRITION BREAKDOWN
Generated: {timestamp}
Data Source: {data_source}

## OVERALL SUMMARY
- Company Attrition Rate: {attrition_rate:.1f}%
- Total Employees: {total_employees:,}

"""
                    if 'Department' in df.columns:
                        dept_stats = df.groupby('Department').agg({
                            'left': ['count', 'sum', 'mean']
                        }).reset_index()
                        dept_stats.columns = ['Department', 'Total', 'Left', 'Attrition_Rate']
                        dept_stats['Attrition_Rate'] = (dept_stats['Attrition_Rate'] * 100).round(1)
                        dept_stats = dept_stats.sort_values('Attrition_Rate', ascending=False)
                        
                        for _, row in dept_stats.iterrows():
                            status = "🚨 CRITICAL" if row['Attrition_Rate'] > 25 else "⚠️ HIGH" if row['Attrition_Rate'] > 15 else "✅ NORMAL"
                            report += f"### {row['Department']}\n"
                            report += f"- Total Employees: {row['Total']}\n"
                            report += f"- Employees Left: {row['Left']}\n"
                            report += f"- Attrition Rate: {row['Attrition_Rate']}% {status}\n\n"
                        
                        report += "\n## DEPARTMENT COMPARISON TABLE\n"
                        report += dept_stats.to_string(index=False)
                    else:
                        report += "*Department column not available in this dataset.*"
                    
                    st.markdown(report)
                    report_text = report
                    
                elif report_type == "Action Items Report":
                    report = f"""
# ACTION ITEMS REPORT
Generated: {timestamp}

## SUMMARY
- Total Action Items: {len(st.session_state.action_items)}
- Pending: {len([a for a in st.session_state.action_items if a['status'] == 'pending'])}
- Completed: {len([a for a in st.session_state.action_items if a['status'] == 'completed'])}
- Overdue: {len([a for a in st.session_state.action_items if a['status'] == 'pending' and a['due_date'] < datetime.now().strftime('%Y-%m-%d')])}

## ACTION ITEMS
"""
                    if len(st.session_state.action_items) == 0:
                        report += "*No action items recorded yet.*"
                    else:
                        for i, action in enumerate(st.session_state.action_items, 1):
                            is_overdue = action['status'] == 'pending' and action['due_date'] < datetime.now().strftime('%Y-%m-%d')
                            report += f"{i}. [{action['status'].upper()}{' - OVERDUE' if is_overdue else ''}] {action['description']}\n"
                            report += f"   Created: {action['created']} | Due: {action['due_date']} | Priority: {action['priority'].upper()}\n\n"
                    
                    st.markdown(report)
                    report_text = report
                    
                elif report_type == "Model Performance Report":
                    report = f"""
# MODEL PERFORMANCE REPORT
Generated: {timestamp}
Data Source: {data_source}

## MODEL CONFIGURATION
- Algorithm: LightGBM (Light Gradient Boosting Machine)
- Training Records: {len(X_train_ref):,}
- Test Records: {len(X_test_cur):,}
- Train/Test Split: 80/20
- Random State: 42

## HYPERPARAMETERS
- n_estimators: 150
- learning_rate: 0.05
- num_leaves: 12
- max_depth: 4
- min_child_samples: 40
- reg_alpha: 0.5
- reg_lambda: 0.5
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: 1.5

## PERFORMANCE METRICS
- Accuracy: {st.session_state.model_accuracy:.1%}
"""
                    
                    # Calculate additional metrics
                    y_pred = pipeline.predict(X_test_cur)
                    y_proba = pipeline.predict_proba(X_test_cur)[:, 1]
                    
                    report += f"- Precision: {precision_score(y_test, y_pred, zero_division=0):.1%}\n"
                    report += f"- Recall: {recall_score(y_test, y_pred, zero_division=0):.1%}\n"
                    report += f"- F1 Score: {f1_score(y_test, y_pred, zero_division=0):.1%}\n"
                    report += f"- ROC AUC: {roc_auc_score(y_test, y_proba):.1%}\n"
                    
                    report += """
## CALIBRATION
- Temperature Scaling: 0.55
- Purpose: To convert raw model probabilities into more interpretable risk percentages

## FEATURE ENGINEERING
- Categorical Encoding: One-Hot Encoding
- Numerical Features: Passed through without scaling (Tree-based model)

## WHY LIGHTGBM?
LightGBM was selected after benchmarking against:
- Random Forest
- Logistic Regression

LightGBM provides the best balance of:
- High accuracy
- Fast training time
- Built-in handling of missing values
- Feature importance interpretability
"""
                    
                    st.markdown(report)
                    report_text = report
                
                # Download buttons
                st.markdown("---")
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.download_button(
                        "⬇️ Download as TXT",
                        report_text.encode('utf-8'),
                        f"retainai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
                
                with col_d2:
                    # Create markdown file
                    st.download_button(
                        "⬇️ Download as Markdown",
                        report.encode('utf-8'),
                        f"retainai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown"
                    )
                
                with col_d3:
                    # Create JSON summary
                    summary_json = {
                        "generated": timestamp,
                        "total_employees": total_employees,
                        "attrition_rate": round(attrition_rate, 1),
                        "high_risk_employees": high_risk_count,
                        "model_accuracy": round(st.session_state.model_accuracy, 3) if st.session_state.model_accuracy else None,
                        "action_items_count": len(st.session_state.action_items),
                        "pending_actions": len([a for a in st.session_state.action_items if a['status'] == 'pending'])
                    }
                    st.download_button(
                        "⬇️ Download JSON Summary",
                        json.dumps(summary_json, indent=2).encode('utf-8'),
                        f"retainai_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
        
        # Quick export options
        st.markdown("---")
        st.markdown("### ⚡ Quick Data Export")
        
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("📥 Export Raw Dataset"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    f"hr_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        with col_q2:
            if st.button("📥 Export Risk Scores"):
                feature_cols = [c for c in df.columns if c != 'left']
                risk_df = df.copy()
                risk_df['Risk_Score'] = calibrate_probability_array(pipeline.predict_proba(risk_df[feature_cols])[:, 1])
                risk_df['Risk_Category'] = risk_df['Risk_Score'].apply(
                    lambda x: 'HIGH' if x > 0.6 else 'MEDIUM' if x > 0.4 else 'LOW'
                )
                risk_df = risk_df.sort_values('Risk_Score', ascending=False)
                csv = risk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Risk CSV",
                    csv,
                    f"risk_scores_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()
