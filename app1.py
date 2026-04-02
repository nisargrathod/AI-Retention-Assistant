# ====================================================================
# All Necessary Imports
# ====================================================================
import os
import sys
import json
import base64
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
from io import BytesIO

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
    
    /* NEW: Onboarding Styles */
    .onboarding-container {
        background: linear-gradient(135deg, #0d2818 0%, #161b22 50%, #1c2128 100%);
        border: 2px solid #17B794;
        border-radius: 20px;
        padding: 40px;
        margin: 20px 0;
        text-align: center;
    }
    
    .onboarding-step {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    .onboarding-step:hover {
        border-color: #17B794;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(23, 183, 148, 0.2);
    }
    
    .step-number {
        background: linear-gradient(135deg, #17B794, #11998e);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 15px;
    }
    
    /* NEW: Search Styles */
    .search-result-item {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .search-result-item:hover {
        border-color: #17B794;
        background-color: #21262d;
    }
    
    /* NEW: Notification Badge */
    .notification-badge {
        background-color: #FF4B4B;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 5px;
    }
    
    /* NEW: Glossary Styles */
    .glossary-item {
        background-color: #161b22;
        border-left: 3px solid #17B794;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .glossary-term {
        color: #17B794;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    .glossary-definition {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* NEW: Succession Card */
    .succession-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .readiness-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #30363d;
        overflow: hidden;
        margin-top: 10px;
    }
    .readiness-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* NEW: D&I Metric Card */
    .di-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
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
        .onboarding-container {
            padding: 20px;
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
# NEW: UTILITY FUNCTIONS
# ====================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'is_global': False,
        'nav_to': None,
        'first_visit': True,
        'alert_history': [],
        'search_results': None,
        'app_settings': {
            'currency': '₹',
            'currency_name': 'INR',
            'salary_low': 400000,
            'salary_medium': 600000,
            'salary_high': 900000,
            'attrition_threshold': 20,
            'satisfaction_threshold': 0.5,
            'hours_threshold': 200,
        },
        'prediction_history': [],
        'onboarding_complete': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def add_to_alert_history(level, message, timestamp=None):
    """Add alert to history"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    alert = {'level': level, 'message': message, 'timestamp': timestamp}
    st.session_state.alert_history.append(alert)
    # Keep only last 50 alerts
    if len(st.session_state.alert_history) > 50:
        st.session_state.alert_history = st.session_state.alert_history[-50:]

def get_currency_symbol():
    """Get currency symbol from settings"""
    return st.session_state.get('app_settings', {}).get('currency', '₹')

def format_currency(amount):
    """Format amount with currency"""
    symbol = get_currency_symbol()
    if amount >= 10000000:
        return f"{symbol}{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"{symbol}{amount/100000:.2f} L"
    else:
        return f"{symbol}{amount:,.0f}"

def calculate_confidence_interval(prob, n_samples=100, method='bootstrap'):
    """Calculate confidence interval for prediction"""
    # Simplified CI calculation based on binomial proportion
    z = 1.96  # 95% CI
    margin = z * np.sqrt((prob * (1 - prob)) / max(n_samples, 1))
    lower = max(0, prob - margin)
    upper = min(1, prob + margin)
    return lower, upper

def get_groq_llm():
    """Get Groq LLM with error handling"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return None, "API_KEY_MISSING"
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            timeout=30
        )
        return llm, None
    except Exception as e:
        return None, str(e)

def dataframe_to_csv_download(df, filename="data.csv"):
    """Convert dataframe to CSV for download"""
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

def get_feature_description(feature_name):
    """Get human-readable description for features"""
    descriptions = {
        'satisfaction_level': 'How satisfied the employee is with their job (0-1 scale)',
        'last_evaluation': 'Performance rating from last review (0-1 scale)',
        'number_project': 'Number of projects the employee is currently working on',
        'average_montly_hours': 'Average hours worked per month',
        'time_spend_company': 'Years the employee has been with the company',
        'Work_accident': 'Whether the employee had a workplace accident (0=No, 1=Yes)',
        'promotion_last_5years': 'Whether the employee was promoted in last 5 years (0=No, 1=Yes)',
        'Department': 'The department the employee works in',
        'salary': 'Salary level (low/medium/high)',
        'left': 'Whether the employee left the company (0=Stayed, 1=Left)',
    }
    return descriptions.get(feature_name, f'The {feature_name} metric')

def get_risk_label(probability):
    """Get human-readable risk label"""
    if probability >= 0.75:
        return "Critical", "#FF4B4B"
    elif probability >= 0.5:
        return "High", "#EEB76B"
    elif probability >= 0.3:
        return "Medium", "#FFA500"
    else:
        return "Low", "#17B794"

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
# NEW: EMPLOYEE SEARCH FUNCTION
# ====================================================================
def search_employees(df, search_query, search_by="all"):
    """Search employees by various criteria"""
    if not search_query:
        return df.head(20)
    
    search_query = search_query.lower()
    results = pd.DataFrame()
    
    if search_by == "all" or search_by == "department":
        if 'Department' in df.columns:
            mask = df['Department'].str.lower().str.contains(search_query, na=False)
            results = pd.concat([results, df[mask]])
    
    if search_by == "all" or search_by == "salary":
        if 'salary' in df.columns:
            mask = df['salary'].str.lower().str.contains(search_query, na=False)
            results = pd.concat([results, df[mask]])
    
    # Numeric search
    try:
        num_query = float(search_query)
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col != 'left':
                mask = df[col] == num_query
                results = pd.concat([results, df[mask]])
    except:
        pass
    
    # Remove duplicates
    results = results.drop_duplicates()
    return results if len(results) > 0 else df.head(0)

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
        try:
            model_sal = CausalModel(data=df_model, treatment='salary_num', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_sal = model_sal.estimate_effect(model_sal.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
            effects['Salary'] = abs(est_sal.value)
        except Exception as e:
            st.warning(f"Salary causal analysis skipped: {str(e)[:50]}...")
            effects['Salary'] = 0.1
        
        try:
            model_sat = CausalModel(data=df_model, treatment='satisfaction_level', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_sat = model_sat.estimate_effect(model_sat.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
            effects['Satisfaction'] = abs(est_sat.value)
        except Exception as e:
            st.warning(f"Satisfaction causal analysis skipped: {str(e)[:50]}...")
            effects['Satisfaction'] = 0.1
            
        try:
            model_hr = CausalModel(data=df_model, treatment='average_montly_hours', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_hr = model_hr.estimate_effect(model_hr.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
            effects['Overwork'] = abs(est_hr.value) * 10
        except Exception as e:
            st.warning(f"Overwork causal analysis skipped: {str(e)[:50]}...")
            effects['Overwork'] = 0.1

        sorted_effects = sorted(effects.items(), key=lambda item: item[1], reverse=True)
        def get_display_info(rank, factor, value):
            if rank == 1: color = "#FF4B4B"; status = "CRITICAL DRIVER"; advice = "This is the #1 reason people leave."
            elif rank == 2: color = "#FFA500"; status = "MAJOR FACTOR"; advice = "Important to address."
            else: color = "#FFD700"; status = "MODERATE FACTOR"; advice = "Monitor this factor."
            return color, status, advice

        c1, c2, c3 = st.columns(3)
        for idx, (col, factor_value) in enumerate(sorted_effects[:3]):
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
    elif "growth" in situation.lower():
        root_cause = "Lack of Career Development"
    else:
        root_cause = "Attrition Risk Factors"
        
    action_description = solution
    cost_str = budget

    llm, error = get_groq_llm()
    if error:
        if error == "API_KEY_MISSING":
            st.warning("🔑 **API Key Not Configured**. To use AI features:\n\n1. Go to your Streamlit Cloud secrets\n2. Add `GROQ_API_KEY`\n3. Get a free key from [groq.com](https://console.groq.com)")
            st.markdown("""
            <div class="custom-card" style="border-left: 4px solid #8b949e;">
                <h4 style="color: #8b949e; margin-top: 0;">📋 Sample Template (Without AI)</h4>
                <p style="color: #c9d1d9;">Subject: Checking In - Your Wellbeing Matters<br><br>
                Dear {name},<br><br>
                I hope this message finds you well. I wanted to reach out personally to check in on how you're doing. Your contributions to {department} have been valuable, and we want to ensure you have the support you need.<br><br>
                I understand that {situation} can be challenging. We'd like to discuss {solution} to help address this.<br><br>
                Could we schedule a brief 15-minute chat this week?<br><br>
                Best regards,<br>HR Team</p>
            </div>
            """.format(name=employee_name, department=department, situation=situation.lower(), solution=solution.lower()), unsafe_allow_html=True)
        else:
            st.error(f"Connection Error: {error}")
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
    Include a clear call-to-action (schedule a meeting, etc.).
    
    **Tone:** Professional, Supportive, Human.
    **Length:** 150-200 words max.
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
            
            # Add copy button
            st.code(response, language=None)
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                st.warning("⏳ **AI is busy right now.** The servers are overloaded. Please wait 30 seconds and try again.")
            else:
                st.error(f"Error generating draft: {e}")


# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Initialize session state
    initialize_session_state()

    # ====================================================================
    # CROSS-PAGE NAVIGATION HANDLER
    # ====================================================================
    menu_options = ['⚙️ Settings', '🏠 Home', '👥 People', '📊 Employee Insights', '🎯 Predict Attrition', '🔍 Why They Leave', '💰 Budget Planner', '🤖 AI Assistant', '🧪 AI Research Lab', '🚀 Strategic Roadmap', '📖 Data Dictionary']
    
    default_idx = 1  # Default to 'Home'
    if 'nav_to' in st.session_state and st.session_state['nav_to']:
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
        feature_columns = [c for c in df.columns if c != 'left']
        st.toast("✅ Using Custom Uploaded Company Data", icon="📊")
    else:
        @st.cache_data
        def load_data_and_train_model(_model_version="v4_calibrated"):
            # FIX: App crash protection if CSV missing
            if not os.path.exists('HR_comma_sep.csv'):
                st.error("❌ Default dataset 'HR_comma_sep.csv' not found. Please go to ⚙️ Settings to upload your data.")
                st.stop()
                
            st.write("📂 Step 1/3: Loading Dataset from CSV...")
            df = pd.read_csv('HR_comma_sep.csv')
            
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
            feature_columns = list(X.columns)
            
            return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, feature_columns

        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat, feature_columns = load_data_and_train_model(_model_version="v4_calibrated")
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
        if 'satisfaction_level' in employee_data.index and employee_data['satisfaction_level'] <= 0.45: strategies.append("🗣️ Conduct 1-on-1 meeting to understand concerns")
        if 'number_project' in employee_data.index:
            if employee_data['number_project'] <= 2: strategies.append("📈 Discuss career aspirations and growth opportunities")
            if employee_data['number_project'] >= 6: strategies.append("⚠️ Assess workload and consider project rebalancing")
        if 'time_spend_company' in employee_data.index and 'promotion_last_5years' in employee_data.index:
            if employee_data['time_spend_company'] >= 4 and employee_data['promotion_last_5years'] == 0: strategies.append("📊 Develop clear career path and promotion roadmap")
        if 'last_evaluation' in employee_data.index and 'satisfaction_level' in employee_data.index:
            if employee_data['last_evaluation'] >= 0.8 and employee_data['satisfaction_level'] < 0.6: strategies.append("🏆 Acknowledge high performance with recognition/reward")
        if 'average_montly_hours' in employee_data.index and employee_data['average_montly_hours'] > 240: strategies.append("⏰ Address burnout risk - consider flexible hours or time off")
        if not strategies: strategies.append("✅ Employee appears stable - continue regular engagement monitoring")
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
        
        # NEW: Notification badge
        alert_count = len(st.session_state.alert_history)
        menu_icons = ['gear', 'house', 'people-fill', 'bar-chart-line-fill', "graph-up-arrow", 'search', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill', 'book']
        
        page = option_menu(
            menu_title=None,
            options=menu_options,  
            icons=menu_icons, 
            menu_icon="cast", default_index=default_idx, 
            styles={
                "container": {"padding": "0!important", "background-color": 'transparent'},
                "icon": {"color": "#17B794", "font-size": "18px"},
                "nav-link": {"color": "#c9d1d9", "font-size": "14px", "text-align": "left", "margin": "0px", "margin-bottom": "8px"},
                "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"},
            }
        )
        
        # NEW: Quick Stats in Sidebar
        st.markdown("<br><hr style='border-color: #30363d; margin: 10px 0;'><br>", unsafe_allow_html=True)
        st.markdown("**📊 Quick Stats**", unsafe_allow_html=True)
        total_emp = len(df)
        attrition = (df['left'].sum() / total_emp * 100) if 'left' in df.columns else 0
        st.metric("Total Employees", f"{total_emp:,}")
        st.metric("Attrition Rate", f"{attrition:.1f}%")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding:20px; border-top:1px solid #2d333b;'><div style='font-size:0.85rem; color:#8b949e;'>Built by</div><div style='font-size:1.6rem; font-weight:600; color:#00E5A8; margin-bottom:10px;'>Nisarg Rathod</div><div style='display:flex; justify-content:center; gap:15px;'><a href='https://www.linkedin.com/in/nisarg-rathod/' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#0A66C2; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='16' height='16' style='filter:invert(1);'/>LinkedIn</a><a href='https://github.com/nisargrathod' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#24292e; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='16' height='16' style='filter:invert(1);'/>GitHub</a></div></div>", unsafe_allow_html=True)

    # ====================================================================
    # PAGE: SETTINGS (Renamed from Global Setup)
    # ====================================================================
    if page == "⚙️ Settings":
        st.header("⚙️ System Settings")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Configure your RetainAI instance.</p>", unsafe_allow_html=True)
        
        tab_data, tab_config, tab_api = st.tabs(["📁 Data Upload", "⚙️ Configuration", "🔑 API Keys"])
        
        with tab_data:
            st.markdown("### Upload Your Company Data")
            st.markdown("<p style='color: #8b949e;'>Turn this AI into your company's dedicated assistant.</p>", unsafe_allow_html=True)
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
                    
                    # Data quality checks
                    st.markdown("---")
                    st.markdown("### 📋 Data Quality Report")
                    missing = new_df.isnull().sum()
                    if missing.sum() > 0:
                        st.warning(f"⚠️ Found {missing.sum()} missing values across {len(missing[missing > 0])} columns")
                        with st.expander("See missing values"):
                            st.dataframe(missing[missing > 0])
                    else:
                        st.success("✅ No missing values detected")
                    
                    duplicates = new_df.duplicated().sum()
                    if duplicates > 0:
                        st.info(f"ℹ️ Found {duplicates} duplicate rows (will be removed during training)")
                    
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
                            y_pred_g = global_pipeline.predict(X_test_g)
                            
                            # Calculate multiple metrics
                            acc = accuracy_score(y_test_g, y_pred_g)
                            prec = precision_score(y_test_g, y_pred_g, zero_division=0)
                            rec = recall_score(y_test_g, y_pred_g, zero_division=0)
                            f1 = f1_score(y_test_g, y_pred_g, zero_division=0)
                            
                            final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                            st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                            st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                            st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g
                            st.session_state['is_global'] = True
                            
                            st.success(f"🎉 Training Complete!")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{acc:.1%}")
                            c2.metric("Precision", f"{prec:.1%}")
                            c3.metric("Recall", f"{rec:.1%}")
                            c4.metric("F1 Score", f"{f1:.1%}")
                            
                            st.session_state['nav_to'] = "🏠 Home"
                            sleep(1)
                            st.rerun()
                except Exception as e: st.error(f"Error: {e}")
            if st.button("🔄 Reset to Default Demo Data"):
                if 'is_global' in st.session_state: del st.session_state['is_global']
                st.rerun()
        
        with tab_config:
            st.markdown("### 🌍 Regional Settings")
            st.markdown("<p style='color: #8b949e;'>Configure currency and thresholds for your region.</p>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                currency_options = {'₹ INR': '₹', '$ USD': '$', '€ EUR': '€', '£ GBP': '£'}
                selected_currency = st.selectbox("Currency", list(currency_options.keys()))
                st.session_state.app_settings['currency'] = currency_options[selected_currency]
                st.session_state.app_settings['currency_name'] = selected_currency.split(' ')[1]
                
                st.markdown("---")
                st.markdown("### 💰 Salary Bands (Annual)")
                st.session_state.app_settings['salary_low'] = st.number_input("Low Salary", value=st.session_state.app_settings['salary_low'], step=50000)
                st.session_state.app_settings['salary_medium'] = st.number_input("Medium Salary", value=st.session_state.app_settings['salary_medium'], step=50000)
                st.session_state.app_settings['salary_high'] = st.number_input("High Salary", value=st.session_state.app_settings['salary_high'], step=50000)
            
            with c2:
                st.markdown("### ⚠️ Alert Thresholds")
                st.session_state.app_settings['attrition_threshold'] = st.slider("Attrition Rate Alert (%)", 5, 40, st.session_state.app_settings['attrition_threshold'], 1)
                st.session_state.app_settings['satisfaction_threshold'] = st.slider("Satisfaction Alert", 0.1, 0.9, st.session_state.app_settings['satisfaction_threshold'], 0.05)
                st.session_state.app_settings['hours_threshold'] = st.slider("Hours Alert", 150, 300, st.session_state.app_settings['hours_threshold'], 10)
                
                st.markdown("---")
                if st.button("💾 Save Settings"):
                    st.success("✅ Settings saved!")
            
            st.markdown("---")
            st.markdown("### Current Configuration")
            st.json(st.session_state.app_settings)
        
        with tab_api:
            st.markdown("### 🔑 API Key Configuration")
            st.markdown("<p style='color: #8b949e;'>Configure API keys for AI features.</p>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="custom-card">
                <h4 style='color: #17B794; margin-top: 0;'>🤖 Groq AI (for Email Drafting)</h4>
                <p style='margin-bottom: 10px;'>Used for: AI Assistant, Strategic Roadmap, AI Research Lab</p>
                <p><strong>Get free API key:</strong> <a href='https://console.groq.com' target='_blank' style='color: #17B794;'>console.groq.com</a></p>
                <p><strong>Setup:</strong> Add to Streamlit Secrets as <code>GROQ_API_KEY</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Test API key
            if st.button("🔗 Test API Connection"):
                llm, error = get_groq_llm()
                if error == "API_KEY_MISSING":
                    st.error("❌ API key not configured")
                elif error:
                    st.error(f"❌ Connection failed: {error}")
                else:
                    st.success("✅ API connection successful!")

    # ====================================================================
    # PAGE: HOME (ENHANCED WITH ONBOARDING & ALERTS)
    # ====================================================================
    if page == "🏠 Home":
        # NEW: First-time onboarding
        if st.session_state.first_visit and not st.session_state.onboarding_complete:
            st.markdown("""
            <div class="onboarding-container">
                <h1 style='color: #17B794; margin-bottom: 10px;'>👋 Welcome to RetainAI</h1>
                <p style='color: #c9d1d9; font-size: 1.1rem; margin-bottom: 30px;'>Your AI-powered workforce intelligence platform. Here's how to get started:</p>
                
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; text-align: left;'>
                    <div class='onboarding-step'>
                        <div class='step-number'>1</div>
                        <h4 style='color: white; margin: 10px 0;'>Upload Your Data</h4>
                        <p style='color: #8b949e;'>Go to ⚙️ Settings to upload your HR dataset, or use the demo data to explore.</p>
                    </div>
                    <div class='onboarding-step'>
                        <div class='step-number'>2</div>
                        <h4 style='color: white; margin: 10px 0;'>Explore Insights</h4>
                        <p style='color: #8b949e;'>Check the 🏠 Home dashboard for alerts, then explore 📊 Employee Insights.</p>
                    </div>
                    <div class='onboarding-step'>
                        <div class='step-number'>3</div>
                        <h4 style='color: white; margin: 10px 0;'>Predict & Act</h4>
                        <p style='color: #8b949e;'>Use 🎯 Predict Attrition to assess individual employees and get retention strategies.</p>
                    </div>
                </div>
                
                <br>
                <p style='color: #8b949e;'>💡 <strong>Tip:</strong> Visit 📖 Data Dictionary anytime to understand what each metric means.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✅ Got it, take me to the dashboard", type="primary"):
                st.session_state.first_visit = False
                st.session_state.onboarding_complete = True
                st.rerun()
            st.stop()
        
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        
        # NEW: Last updated timestamp
        st.caption(f"📊 Data as of: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
        
        total_employees = len(df)
        attrition_rate = (df['left'].sum() / len(df)) * 100 if 'left' in df.columns else 0
        
        # Get settings
        settings = st.session_state.app_settings
        
        # PROACTIVE ALERT SYSTEM
        alerts = []
        if attrition_rate > settings['attrition_threshold']:
            alert_msg = f"Overall Attrition rate is {attrition_rate:.1f}% (Exceeds {settings['attrition_threshold']}% threshold)"
            alerts.append(("🚨 CRITICAL", alert_msg))
            add_to_alert_history("CRITICAL", alert_msg)
        
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < settings['satisfaction_threshold']:
            alert_msg = f"Average Employee Satisfaction is critically low ({df['satisfaction_level'].mean():.2f} < {settings['satisfaction_threshold']})"
            alerts.append(("⚠️ WARNING", alert_msg))
            add_to_alert_history("WARNING", alert_msg)
            
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > settings['hours_threshold']:
            alert_msg = f"High Average Working Hours detected ({df['average_montly_hours'].mean():.0f} > {settings['hours_threshold']} - Burnout risk)"
            alerts.append(("⚠️ WARNING", alert_msg))
            add_to_alert_history("WARNING", alert_msg)
        
        # Display alerts
        if alerts:
            for level, msg in alerts:
                if "CRITICAL" in level: st.error(msg)
                else: st.warning(msg)
            with st.expander("📋 View Alert History"):
                if st.session_state.alert_history:
                    alert_df = pd.DataFrame(st.session_state.alert_history)
                    alert_df.columns = ['Level', 'Message', 'Timestamp']
                    st.dataframe(alert_df, use_container_width=True)
                    if st.button("Clear History"):
                        st.session_state.alert_history = []
                        st.rerun()
                else:
                    st.info("No alert history")
        
        # Metrics
        if 'satisfaction_level' in df.columns:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
            if 'average_montly_hours' in df.columns:
                col4.metric("Avg. Monthly Hours", f"{df['average_montly_hours'].mean():.0f}")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        
        # QUICK ACTIONS
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("👤 Find Employee", use_container_width=True):
                st.session_state['nav_to'] = "👥 People"
                st.rerun()
        with c2:
            if st.button("🎯 Predict Risk", use_container_width=True):
                st.session_state['nav_to'] = "🎯 Predict Attrition"
                st.rerun()
        with c3:
            if st.button("💰 Budget Plan", use_container_width=True):
                st.session_state['nav_to'] = "💰 Budget Planner"
                st.rerun()
        with c4:
            if st.button("📧 Draft Email", use_container_width=True):
                st.session_state['nav_to'] = "🤖 AI Assistant"
                st.rerun()
        with c5:
            if st.button("🚀 6-Month Plan", use_container_width=True):
                st.session_state['nav_to'] = "🚀 Strategic Roadmap"
                st.rerun()

        # AT-RISK EMPLOYEE LIST
        st.markdown("---")
        st.markdown("### 🔴 Top 10 At-Risk Employees (Currently Working)")
        st.caption("Predicted by AI as most likely to leave. Click 'Find Employee' to search for specific people.")
        
        current_df = df[df['left'] == 0].copy()
        if len(current_df) > 0:
            feature_columns_home = [c for c in df.columns if c != 'left']
            risks = pipeline.predict_proba(current_df[feature_columns_home])[:, 1]
            current_df['Risk_Score'] = calibrate_probability_array(risks)
            
            # Add confidence intervals
            current_df['Risk_Lower'] = current_df['Risk_Score'].apply(lambda x: calculate_confidence_interval(x)[0])
            current_df['Risk_Upper'] = current_df['Risk_Score'].apply(lambda x: calculate_confidence_interval(x)[1])
            
            top_risk = current_df.nlargest(10, 'Risk_Score')
            
            for idx, row in top_risk.iterrows():
                risk_pct = row['Risk_Score'] * 100
                risk_label, risk_color = get_risk_label(row['Risk_Score'])
                dept = row.get('Department', 'N/A')
                salary = row.get('salary', 'N/A')
                confidence_range = f"{row['Risk_Lower']*100:.0f}-{row['Risk_Upper']*100:.0f}%"
                
                st.markdown(f"""
                <div style='background:{risk_color}10; border-left:4px solid {risk_color}; padding:12px 15px; margin:8px 0; border-radius:0 8px 8px 0; display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <strong>{dept}</strong> | {salary.title()} Salary
                        <br><small style='color: #8b949e;'>Confidence: {confidence_range}</small>
                    </div>
                    <div style='text-align: right;'>
                        <span style='color:{risk_color}; font-weight:700; font-size: 1.1rem;'>{risk_label}: {risk_pct:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No current employees found in dataset.")
        
        st.markdown("---")
        with st.expander("📄 View Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)
            csv = dataframe_to_csv_download(df.head(100), "employee_data.csv")
            st.download_button("⬇️ Download Data", csv, "employee_data.csv", "text/csv")

    # ====================================================================
    # NEW PAGE: PEOPLE (Employee Search & Lookup)
    # ====================================================================
    if page == "👥 People":
        st.header("👥 People Directory")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Search and explore individual employee profiles.</p>", unsafe_allow_html=True)
        
        # Search bar
        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            search_query = st.text_input("🔍 Search employees...", placeholder="Search by department, salary, or any value...")
        with col_search2:
            search_by = st.selectbox("Filter by", ["all", "department", "salary"], label_visibility="collapsed")
        
        # Apply search
        if search_query:
            search_results = search_employees(df, search_query, search_by)
        else:
            search_results = df.head(50)
        
        # Display results count
        st.caption(f"Showing {len(search_results)} of {len(df)} employees")
        
        if len(search_results) > 0:
            # Calculate risk scores for displayed employees
            feature_cols_search = [c for c in df.columns if c != 'left']
            search_display = search_results.copy()
            risks = pipeline.predict_proba(search_display[feature_cols_search])[:, 1]
            search_display['Risk Score'] = calibrate_probability_array(risks)
            search_display['Risk Level'] = search_display['Risk Score'].apply(lambda x: get_risk_label(x)[0])
            
            # Display as cards
            for idx, row in search_display.head(20).iterrows():
                risk_label, risk_color = get_risk_label(row['Risk Score'])
                risk_pct = row['Risk Score'] * 100
                
                # Build info string
                info_parts = []
                if 'Department' in row.index: info_parts.append(f"🏢 {row['Department']}")
                if 'salary' in row.index: info_parts.append(f"💰 {row['salary'].title()}")
                if 'time_spend_company' in row.index: info_parts.append(f"📅 {int(row['time_spend_company'])} years")
                if 'number_project' in row.index: info_parts.append(f"📂 {int(row['number_project'])} projects")
                if 'satisfaction_level' in row.index: info_parts.append(f"😊 {row['satisfaction_level']:.2f}")
                
                status = "Left" if row.get('left', 0) == 1 else "Active"
                status_color = "#FF4B4B" if status == "Left" else "#17B794"
                
                st.markdown(f"""
                <div class='search-result-item' style='border-left: 4px solid {risk_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong style='color: white;'>Employee #{idx}</strong>
                            <span style='color: {status_color}; margin-left: 10px; font-size: 0.85rem;'>● {status}</span>
                            <br>
                            <small style='color: #8b949e;'>{' | '.join(info_parts)}</small>
                        </div>
                        <div style='text-align: right;'>
                            <div style='color: {risk_color}; font-weight: 700; font-size: 1.1rem;'>{risk_label}</div>
                            <div style='color: #8b949e; font-size: 0.85rem;'>Risk: {risk_pct:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(search_display) > 20:
                st.info(f"Showing top 20 results. Refine your search for more specific results.")
            
            # Download search results
            csv = dataframe_to_csv_download(search_display, "search_results.csv")
            st.download_button("⬇️ Download Search Results", csv, "search_results.csv", "text/csv")
        else:
            st.info("No employees found matching your search criteria.")

    # ====================================================================
    # PAGE: EMPLOYEE INSIGHTS
    # ====================================================================
    if page == "📊 Employee Insights":
        st.header("📉 Employee Data Analysis")
        st.write("Explore the workforce demographics to identify patterns.")
        
        # NEW: Summary stats
        with st.expander("📋 Quick Summary Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    # ====================================================================
    # PAGE: PREDICT ATTRITION (ENHANCED WITH CONFIDENCE & HISTORY)
    # ====================================================================
    if page == "🎯 Predict Attrition":
        tab_ind, tab_batch, tab_history = st.tabs(["🎯 Individual Prediction", "📦 Batch Upload", "📜 Prediction History"])
        
        with tab_ind:
            st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9ca3af;'>Enter employee details to see if they will Stay or Leave, with confidence levels.</p>", unsafe_allow_html=True)
            
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
                with c_test2:
                    if st.button("Test with Employee who Stayed"):
                        sample = df[df['left'] == 0].iloc[0]
                        test_df = sample.drop('left').to_frame().T
                        raw_prob = pipeline.predict_proba(test_df)[0][1]
                        calibrated_prob = calibrate_probability(raw_prob)
                        pred = 1 if calibrated_prob >= 0.5 else 0
                        if pred == 0: st.success(f"✅ **Correct!** Prediction: Stay ({(1-calibrated_prob)*100:.1f}%)")
                        else: st.error(f"❌ **Incorrect.** Prediction: Leave ({calibrated_prob*100:.1f}%)")

            st.markdown("---")
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
                    sleep(0.5)
                    input_df = input_df[feature_columns] 
                    raw_probas = pipeline.predict_proba(input_df)[0]
                    calibrated_stay = calibrate_probability(raw_probas[0], temperature=0.55)
                    calibrated_leave = 1 - calibrated_stay
                    prediction = 1 if calibrated_leave >= 0.5 else 0
                    
                    st.session_state.prediction_result = prediction
                    st.session_state.input_df = input_df
                    st.session_state.prediction_probas = [calibrated_stay, calibrated_leave]
                    
                    # NEW: Add to history
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'input': input_data.copy(),
                        'prediction': 'LEAVE' if prediction == 1 else 'STAY',
                        'confidence': calibrated_leave if prediction == 1 else calibrated_stay
                    }
                    st.session_state.prediction_history.append(history_entry)

            if st.session_state.prediction_result is not None:
                st.markdown("---")
                
                stay_prob = st.session_state.prediction_probas[0]
                leave_prob = st.session_state.prediction_probas[1]
                
                # NEW: Calculate confidence interval
                ci_lower, ci_upper = calculate_confidence_interval(leave_prob, len(df))
                confidence_margin = (ci_upper - ci_lower) / 2
                
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
                        <div style='color: #8b949e; font-size: 0.8rem; margin-top: 5px;'>±{confidence_margin*100:.0f}% confidence margin</div>
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
                
                # NEW: Confidence explanation
                with st.expander("📊 Understanding the Confidence Level"):
                    st.markdown(f"""
                    **What does ±{confidence_margin*100:.0f}% mean?**
                    
                    - The AI is {leave_percent}% confident the employee will leave
                    - However, the true probability could be between **{ci_lower*100:.0f}%** and **{ci_upper*100:.0f}**
                    - This range is based on the sample size ({len(df)} employees) and prediction certainty
                    
                    **How to use this:**
                    - If the lower bound (>30%) still indicates risk → Take preventive action
                    - If the upper bound (<40%) is below threshold → Monitor but don't overreact
                    """)
                
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
                                                    if 'satisfaction' in col_lower: action_text = f"🤝 <strong>Boost Engagement</strong>: Improve satisfaction from <strong>{orig_val:.2f}</strong> to <strong>{new_val:.2f}</strong>."
                                                    elif 'hours' in col_lower:
                                                        diff = orig_val - new_val
                                                        if diff > 0: action_text = f"⏰ <strong>Reduce Workload</strong>: Cut monthly hours by ~<strong>{abs(diff):.0f}</strong>."
                                                        else: action_text = f"⏰ <strong>Adjust Hours</strong>: Set to ~<strong>{new_val:.0f}</strong>."
                                                    elif 'project' in col_lower: action_text = f"📂 <strong>Rebalance Projects</strong>: Adjust to <strong>{int(new_val)}</strong> projects."
                                                    elif 'evaluation' in col_lower: action_text = f"📊 <strong>Coaching</strong>: Guide evaluation to <strong>{new_val:.2f}</strong>."
                                                    else: action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: {orig_val:.2f} → {new_val:.2f}."
                                                    if action_text: changes.append(action_text)
                                            else:
                                                if orig_val != new_val:
                                                    if 'department' in col.lower(): has_high_effort = True; action_text = f"🏢 <strong>Transfer</strong>: Move to <strong>{new_val}</strong>. <span style='color:#EEB76B;'>(High Effort)</span>"
                                                    else: action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: {orig_val} → {new_val}."
                                                    changes.append(action_text)
                                        if not changes: changes.append("• (AI suggests maintaining current status with minor supervision)")
                                        changes_str = "".join([f"<div class='action-item {'action-item-high-effort' if has_high_effort else ''}'>{c}</div>" for c in changes])
                                        scenarios_html.append(f"<div class='custom-card' style='border-color: #17B794;'><h4 style='color: #17B794; margin-top:0;'>Strategy {i+1}</h4><p style='color: #c9d1d9; font-size: 0.9rem; line-height: 1.6;'>{changes_str}</p><div style='margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;'><small style='color: #17B794;'><strong>Result:</strong> AI predicts employee will <strong>STAY</strong>.</small></div></div>")
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    cols_list = [col_s1, col_s2, col_s3]
                                    for i, html in enumerate(scenarios_html):
                                        with cols_list[i]: st.markdown(html, unsafe_allow_html=True)
                            except Exception as e: st.error(f"Error generating strategies: {e}")
        
        with tab_batch:
            st.markdown("### 📦 Batch Prediction")
            st.write("Upload a CSV of multiple employees to get predictions instantly.")
            batch_file = st.file_uploader("Upload Employee CSV", type=["csv"], key="batch_uploader")
            if batch_file:
                try:
                    batch_df = pd.read_csv(batch_file)
                    req_cols = [c for c in feature_columns if c in batch_df.columns]
                    missing_cols = [c for c in feature_columns if c not in batch_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns: {missing_cols}")
                        st.info(f"Required columns: {feature_columns}")
                    elif len(req_cols) == len(feature_columns):
                        probs = pipeline.predict_proba(batch_df[feature_columns])[:, 1]
                        batch_df['Risk Score'] = calibrate_probability_array(probs)
                        batch_df['Risk Level'] = batch_df['Risk Score'].apply(lambda x: get_risk_label(x)[0])
                        batch_df['Prediction'] = batch_df['Risk Score'].apply(lambda x: "LEAVE" if x > 0.5 else "STAY")
                        batch_df = batch_df.sort_values('Risk Score', ascending=False)
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Summary stats
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Processed", len(batch_df))
                        c2.metric("High Risk", len(batch_df[batch_df['Risk Level'].isin(['Critical', 'High'])]))
                        c3.metric("Low Risk", len(batch_df[batch_df['Risk Level'] == 'Low']))
                        
                        csv = dataframe_to_csv_download(batch_df, "batch_predictions.csv")
                        st.download_button("⬇️ Download Results", csv, "batch_predictions.csv", "text/csv")
                    else:
                        st.error(f"Column mismatch. Required: {feature_columns}")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        # NEW: Prediction History Tab
        with tab_history:
            st.markdown("### 📜 Prediction History")
            st.write("View your recent individual predictions.")
            
            if st.session_state.prediction_history:
                for i, entry in enumerate(reversed(st.session_state.prediction_history[-10:])):
                    pred_color = "#FF4B4B" if entry['prediction'] == 'LEAVE' else "#17B794"
                    st.markdown(f"""
                    <div style='background: {pred_color}10; border-left: 3px solid {pred_color}; padding: 10px 15px; margin: 5px 0; border-radius: 0 8px 8px 0;'>
                        <strong>{entry['prediction']}</strong> - {entry['timestamp']}
                        <br><small style='color: #8b949e;'>Confidence: {entry['confidence']*100:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
            else:
                st.info("No predictions made yet. Go to the Individual Prediction tab to start.")

    if page == "🔍 Why They Leave":
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
    # REBUILT PAGE: BUDGET PLANNER
    # ====================================================================
    if page == "💰 Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-bottom: 30px;'>A two-step financial playbook to secure executive budget approval and optimize your spend.</p>", unsafe_allow_html=True)
        
        # Get salary settings
        settings = st.session_state.app_settings
        salary_map = {
            'low': settings['salary_low'],
            'medium': settings['salary_medium'],
            'high': settings['salary_high']
        }
        currency = settings['currency']
        
        # --- PHASE 1: TRUE COST OF TURNOVER ---
        st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid #FF4B4B; background: linear-gradient(to right, #2d1515 0%, #1c2128 100%);">
            <h3 style="color: #FF4B4B; margin-top: 0;">💸 Phase 1: The Burn Rate</h3>
            <p style="color: #e6edf3; font-size: 1rem;"><strong>The Question:</strong> "How much money are we actually losing, and where?"</p>
            <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;">CFOs speak in money. Translating attrition into hard numbers gets budget approval.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'Department' in df.columns:
            df_cost = df.copy()
            if 'salary' in df_cost.columns:
                df_cost['annual_salary'] = df_cost['salary'].map(salary_map)
            else:
                df_cost['annual_salary'] = (settings['salary_low'] + settings['salary_high']) / 2
            
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
                c_shock1.metric("Total Money Lost to Attrition", format_currency(grand_total), delta=f"{total_left} Employees Left", delta_color="inverse")
                c_shock2.metric("Avg. Cost Per Exit", format_currency(avg_exit_cost), delta="Industry Standard Math")
                c_shock3.metric("Most Expensive Dept.", dept_costs.iloc[0]['Department'], delta=f"{format_currency(dept_costs.iloc[0]['total_cost'])} Lost", delta_color="inverse")
                
                with st.expander("📐 How we calculate this (Click to verify)"):
                    st.markdown(f"""
                    We use the **Standard HR Industry Formula** for turnover cost:\n
                    > `Cost = 50% of Annual Salary × (1 + Years of Tenure × 10%)`
                    
                    *   **Base Replacement Cost:** 50% of salary (covers recruiting, onboarding, admin).
                    *   **Experience Penalty:** +10% for every year of experience (institutional knowledge loss).
                    *   **Cap:** Multiplier capped at 2.0x for senior employees.
                    
                    *Salary Bands Used:* Low: {currency}{salary_map['low']:,} | Medium: {currency}{salary_map['medium']:,} | High: {currency}{salary_map['high']:,}
                    
                    *Example:* A 5-year employee earning {currency}{salary_map['medium']:,} costs `0.5 × {salary_map['medium']:,} × 1.5 = {currency}{int(salary_map['medium']*0.75):,}` to replace.
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
                    yaxis_title=f"Total Loss ({currency})",
                    xaxis={'categoryorder': 'total descending'},
                    xaxis_title="",
                    showlegend=False
                )
                custome_layout(fig_cost, title_size=24, showlegend=False); 
                st.plotly_chart(fig_cost, use_container_width=True)
                
                top_dept = dept_costs.iloc[0]
                st.markdown(f"""<div class="custom-card" style="border-left: 4px solid #FF4B4B;"><h4 style="color: #FF4B4B; margin-top: 0;">Executive Talking Point</h4><p style="color: #e6edf3; font-size: 1.1rem;"><strong>{top_dept['Department']} attrition cost us {format_currency(top_dept['total_cost'])} last year.</strong> That's the price of losing {top_dept['employees_left']} employees and their accumulated experience.</p></div>""", unsafe_allow_html=True)
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
            <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;">We use Mathematical Optimization (MILP) to find the exact combination of employees that yields maximum ROI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input(f"Enter Total Retention Budget ({currency})", min_value=100000, max_value=10000000, value=1000000, step=50000)
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
                        high_risk_df['salary_val'] = (settings['salary_low'] + settings['salary_high']) / 2
                    
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
                            m1.metric("Investment Needed", format_currency(total_investment), delta=f"{(total_investment/budget)*100:.1f}% of Budget")
                            m2.metric("Projected Savings", format_currency(total_savings), delta="ROI Positive")
                            m3.metric("Lives Saved", f"{len(selected_employees)} People", delta="Prevented Exit")
                            
                            st.markdown("### 📋 Target List")
                            st.caption("These employees have been mathematically proven to be cheaper to retain than to replace.")
                            
                            display_df = selected_employees[['Department', 'salary', 'risk', 'cost_to_retain', 'net_savings']].copy()
                            display_df.rename(columns={
                                'salary': 'Salary Tier', 
                                'risk': 'AI Risk %', 
                                'cost_to_retain': f'Cost to Retain ({currency})', 
                                'net_savings': f'Net Savings ({currency})'
                            }, inplace=True)
                            display_df['AI Risk %'] = (display_df['AI Risk %'] * 100).round(1).astype(str) + "%"
                            display_df[f'Cost to Retain ({currency})'] = display_df[f'Cost to Retain ({currency})'].apply(lambda x: format_currency(x))
                            display_df[f'Net Savings ({currency})'] = display_df[f'Net Savings ({currency})'].apply(lambda x: format_currency(x))
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Download option
                            csv = dataframe_to_csv_download(display_df, "retention_targets.csv")
                            st.download_button("⬇️ Download Target List", csv, "retention_targets.csv", "text/csv")
                        else:
                            st.error("❌ Optimization failed. Budget may be too low to save anyone.")
                            
    if page == "🤖 AI Assistant":
        st.header("🤖 AI Assistant")
        st.markdown("<p style='color: #9ca3af;'>Tools to simplify HR communication.</p>", unsafe_allow_html=True)
        st.markdown("### ✍️ Draft Retention Communication")
        st.write("Select a scenario, and we'll draft a message for you.")
        
        # Check API status
        llm, api_error = get_groq_llm()
        if api_error == "API_KEY_MISSING":
            st.info("💡 To use AI features, configure your Groq API key in ⚙️ Settings → 🔑 API Keys")
        
        with st.form("llm_form"):
            c1, c2 = st.columns(2)
            with c1:
                emp_name = st.text_input("Employee Name", value="Rahul Sharma")
                if 'Department' in df.columns: emp_dept = st.selectbox("Department", df['Department'].unique())
                else: emp_dept = st.text_input("Department", value="Sales")
            with c2:
                situation_input = st.selectbox("What is the situation?", ["Overworked & Burned out", "Seeking Higher Salary", "Low Morale / Unhappy", "Lack of Growth Opportunities"])
                solution_input = st.selectbox("Proposed Solution", ["Offer Flexible Hours", "Discuss Salary Adjustment", "Offer Promotion/Role Change", "Organize 1-on-1 Wellness Session"])
                cost_input = st.text_input("Estimated Annual Cost (Optional)", value=format_currency(50000))
            generate_btn = st.form_submit_button("🚀 Generate Email Draft")
            if generate_btn: run_groq_consultant(emp_name, emp_dept, situation_input, solution_input, cost_input)

    if page == "🧪 AI Research Lab":
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
                st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Use AI to uncover the <strong>specific reasons</strong> why employees are leaving a particular department.</p>", unsafe_allow_html=True)
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
                                    fig.update_layout(xaxis_title="Relative Impact", yaxis_title="", height=400, margin=dict(l=0, r=0, t=40, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown("### 💡 Recommended Retention Strategy")
                                    def get_driver_advice(feature_raw):
                                        if 'Satisfaction' in feature_raw: return "Improve Engagement", "Conduct 'Stay Interviews' and pulse surveys.", "🗣️"
                                        elif 'Hour' in feature_raw or 'Time' in feature_raw: return "Address Burnout", "Review project allocation and enforce 'Right to Disconnect'.", "⏰"
                                        elif 'Project' in feature_raw: return "Optimize Work", "Rebalance assignments to the 'Goldilocks' zone.", "📂"
                                        elif 'Evaluation' in feature_raw: return "Clarify Expectations", "Implement clearer KPIs and feedback loops.", "📊"
                                        elif 'Salary' in feature_raw: return "Review Compensation", "Conduct market salary analysis.", "💰"
                                        else: return f"Monitor {feature_raw}", "Investigate department-specific policies.", "🔍"
                                    c1, c2, c3 = st.columns(3); cols = [c1, c2, c3]
                                    for index, col in enumerate(cols):
                                        if index < len(top_3_drivers):
                                            driver_row = top_3_drivers.iloc[index]; feature_name = driver_row['Feature']
                                            icon, title, advice = get_driver_advice(feature_name)
                                            card_html = f"<div class='custom-card' style='border-top: 4px solid #17B794;'><div style='display: flex; align-items: center; margin-bottom: 10px;'><span style='font-size: 1.5rem; margin-right: 10px;'>{icon}</span><h4 style='margin: 0; color: #fff;'>{title}</h4></div><p style='color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;'>{advice}</p><small style='color: #8b949e;'>Driver: {feature_name.replace('_', ' ').title()}</small></div>"
                                            with col: st.markdown(card_html, unsafe_allow_html=True)

        with tab3:
            st.subheader("🛡️ AI Disruption Defense")
            st.caption("Prove to leadership that reskilling is cheaper than mass layoffs.")
            st.error("**The Fear:** CEO asks, 'Can we just replace half the team with AI tools?'")
            st.success("**The Reality:** AI replaces *tasks*, not jobs. And layoffs cost way more than you think.")
            st.write("")
            
            settings = st.session_state.app_settings
            avg_salary_default = (settings['salary_low'] + settings['salary_high']) / 2
            currency = settings['currency']
            
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
                fig_vuln.update_layout(xaxis_title="0 = Safe | 100 = At Risk", yaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_vuln, use_container_width=True)

            st.markdown("---")
            st.markdown("""
            <div class="custom-card" style="border-left: 5px solid #9ca3ca; background: linear-gradient(to right, #151d28 0%, #1c2128 100%);">
                <h4 style="color: #9ca3ca; margin-top: 0;">🧠 The 'Proof of Work' Defense Strategy</h4>
                <p style="color: #c9d1d9; font-size: 0.95rem; line-height: 1.6;">
                "AI can write code in 10 seconds, but it takes a human 3 weeks to understand the business context. If we lay off 200 people to save costs, we lose their domain expertise. If we give 50 people AI tools, we increase their capacity by 30%, allowing us to <strong>absorb the workload of the 200 people who left last year</strong> without losing institutional knowledge."
                </p>
                <p style="color: #8b949e; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0;">
                <strong>Bottom Line:</strong> AI doesn't replace jobs; it replaces tasks. Our goal isn't fewer people; it's highly utilized people.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Step 2: The Calculator (Reskill vs. Layoff)")
            c_inp1, c_inp2 = st.columns(2)
            with c_inp1:
                num_employees = st.number_input("Employees at risk of AI replacement?", min_value=5, max_value=500, value=50, step=5)
                avg_salary = st.number_input(f"Their average annual salary ({currency})", min_value=200000, max_value=3000000, value=int(avg_salary_default), step=50000)
            with c_inp2: st.markdown("**Industry Costs:**"); st.caption("Severance: 3 months | Hiring AI Engineer: 2x salary | Training: 1.5x salary | Morale dip: 1 month salary")
            
            if st.button("Calculate", type="primary", key="calc_reskill"):
                severance_cost = (avg_salary / 12) * 3 * num_employees
                new_hire_cost = (avg_salary * 2) * (num_employees * 0.1)
                total_layoff_cost = severance_cost + new_hire_cost
                training_cost = (avg_salary / 12) * 1.5 * num_employees
                productivity_dip = (avg_salary / 12) * 1 * num_employees
                total_reskill_cost = training_cost + productivity_dip
                savings = total_layoff_cost - total_reskill_cost
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Cost of Layoffs", format_currency(total_layoff_cost))
                c_res2.metric("Cost of Reskilling", format_currency(total_reskill_cost))
                c_res3.metric("Saved by Reskilling", format_currency(savings), delta="Reskill wins")
                if savings > 0: st.success(f"**Verdict: Reskill.** Saves {format_currency(savings)}.")
                else: st.warning(f"**Verdict:** Layoffs technically cheaper, but consider hidden costs.")
            
            if st.button("✍️ Generate Strategy Memo for CEO", type="primary", key="gen_ai_memo"):
                llm, error = get_groq_llm()
                if error:
                    st.warning("🔑 API key required for AI memo generation. See ⚙️ Settings.")
                else:
                    with st.spinner("Drafting strategy memo..."):
                        try:
                            template = """You are an HR Director writing to the CEO. Keep it under 400 words. Use bullet points. **Context:** - {num_employees} employees at risk of AI automation. - Layoff cost: {total_layoff_cost}. - Reskilling cost: {total_reskill_cost}. - Reskilling saves: {savings}. **Task:** Recommend reskilling. Explain why layoffs are a trap. Propose a 6-month pilot."""
                            prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
                            response = chain.invoke({"num_employees": num_employees, "total_layoff_cost": format_currency(total_layoff_cost), "total_reskill_cost": format_currency(total_reskill_cost), "savings": format_currency(savings)})
                            st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                        except Exception as e:
                            if "rate_limit" in str(e).lower(): st.warning("⏳ AI busy. Wait 30 seconds and retry.")
                            else: st.error(f"Error: {e}")
                            
    # ====================================================================
    # Page: STRATEGIC ROADMAP
    # ====================================================================
    if page == "🚀 Strategic Roadmap":
        st.header("🚀 Future Planning & Projections")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>A tool to show leadership exactly what happens if we take action vs. if we do nothing.</p>", unsafe_allow_html=True)
        
        settings = st.session_state.app_settings
        currency = settings['currency']
        salary_map = {'low': settings['salary_low'], 'medium': settings['salary_medium'], 'high': settings['salary_high']}
        
        st.markdown("### 📋 Step 1: Get Your 6-Month Action Plan")
        issues = []
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.6: issues.append("Low Employee Satisfaction")
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: issues.append("Employee Burnout (High Working Hours)")
        if len(issues) == 0: issues.append("Standard Workforce Stabilization")
        issues_str = ", ".join(issues)
        st.markdown(f"""<div class="custom-card"><h4 style="color: #17B794; margin-top: 0;">🩺 AI Diagnostic Summary</h4><p style="color: #c9d1d9; line-height: 1.6;">AI flagged risks: <strong style="color: #EEB76B;">➤ {issues_str}</strong></p></div>""", unsafe_allow_html=True)
        
        llm, api_error = get_groq_llm()
        if api_error == "API_KEY_MISSING":
            st.info("💡 Configure Groq API key in ⚙️ Settings for AI-generated plans")
        
        if st.button("✍️ Draft My 6-Month HR Action Plan", type="primary"):
            if api_error:
                st.markdown("""
                <div class="llm-response">
<strong>📋 Generic 6-Month HR Action Plan Template</strong>

<strong>Month 1-2: Discovery</strong>
• Conduct organization-wide engagement survey
• Identify top 20 at-risk employees
• Hold 1-on-1 "stay interviews"

<strong>Month 3-4: Intervention</strong>
• Implement flexible work policies
• Launch manager training program
• Begin compensation review process

<strong>Month 5-6: Sustain</strong>
• Track key metrics weekly
• Celebrate early wins
• Prepare 90-day progress report for leadership
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Drafting your 6-month strategy..."):
                    try:
                        template = """You are an expert HR Strategist. **Context:** Our AI identified these attrition drivers: {issues}. **Task:** Create a 6-month execution roadmap. Break into phases. For each month: 1. Phase Name, 2. Actionable Steps (2-3 bullets), 3. Success Metrics. **Tone:** Practical HR actions."""
                        prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({"issues": issues_str})
                        st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    except Exception as e: st.error(f"Error: {e}")

        st.markdown("---"); st.markdown("### 📈 Step 2: See the Future Impact (12-Month Projection)")
        col_f1, col_f2 = st.columns(2)
        with col_f1: intervention_efficacy = st.slider("If we take action, how many at-risk people will we save? (%)", 10, 50, 20, 5)
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
            fig_forecast = px.line(forecast_df, x='Month', y='Workforce Count', color='Scenario', title="Projected Workforce Size Over 12 Months", template="plotly_dark", markers=True, color_discrete_map={'If We Do Nothing (Status Quo)': "#EEB76B", 'If We Follow the Plan': "#17B794"})
            fig_forecast.update_layout(yaxis_title="Total Employee Headcount", xaxis=dict(dtick=1)); st.plotly_chart(fig_forecast, use_container_width=True)
            saved_employees = forecast_intervention[-1] - forecast_bau[-1]
            if 'salary' in df.columns: avg_salary_proj = df['salary'].map(salary_map).mean()
            else: avg_salary_proj = (settings['salary_low'] + settings['salary_high']) / 2
            replacement_cost_per_emp = avg_salary_proj * 0.5; total_money_saved = int(saved_employees) * replacement_cost_per_emp
            st.markdown("---"); st.markdown("### 🏢 HR Director Summary (For Leadership)")
            col_sum_1, col_sum_2 = st.columns(2)
            with col_sum_1: st.metric("Employees Saved by Plan", f"{int(saved_employees)} People")
            with col_sum_2: st.metric("Estimated Costs Prevented", format_currency(total_money_saved), delta="Financial Value")
            st.success(f"**Bottom Line:** Retaining {intervention_efficacy}% saves {int(saved_employees)} employees and prevents **{format_currency(total_money_saved)}** in costs.")

    # ====================================================================
    # NEW PAGE: DATA DICTIONARY
    # ====================================================================
    if page == "📖 Data Dictionary":
        st.header("📖 Data Dictionary")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Understand what each metric means and how to interpret it.</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card" style="border-left: 4px solid #17B794;">
            <h4 style="color: #17B794; margin-top: 0;">❓ What is this page?</h4>
            <p style="color: #c9d1d9;">This glossary explains all the terms and metrics used in RetainAI. Refer to this page whenever you're unsure about what a number means.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Core Metrics
        st.markdown("### 📊 Core Employee Metrics")
        core_metrics = [
            ('Satisfaction Level', 'A score from 0 to 1 measuring how happy the employee is with their job. Below 0.5 is concerning.'),
            ('Last Evaluation', 'Performance rating from the most recent review (0-1 scale). Above 0.7 is good performance.'),
            ('Number of Projects', 'How many projects the employee is currently assigned. 2-5 is normal. 6+ indicates overload.'),
            ('Average Monthly Hours', 'Average hours worked per month. Above 200 hours indicates potential burnout risk.'),
            ('Time Spent at Company', 'Years of tenure. High turnover at 3-5 years indicates career growth issues.'),
            ('Work Accident', 'Whether the employee had a workplace accident (0=No, 1=Yes). Can impact morale.'),
            ('Promotion Last 5 Years', 'Whether promoted recently (0=No, 1=Yes). Lack of promotion is a retention risk for high performers.'),
        ]
        
        for term, definition in core_metrics:
            st.markdown(f"""
            <div class="glossary-item">
                <div class="glossary-term">{term}</div>
                <div class="glossary-definition">{definition}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Metrics
        st.markdown("### 🤖 AI Prediction Metrics")
        ai_metrics = [
            ('Risk Score', 'A percentage (0-100%) indicating how likely the employee is to leave. Above 50% = High Risk.'),
            ('Risk Level', 'Categorization: Critical (75%+), High (50-74%), Medium (30-49%), Low (0-29%).'),
            ('Confidence Interval', 'A range showing the uncertainty in our prediction. Wider range = less certain.'),
            ('SHAP Value', 'A technical measure showing how much each factor contributed to a specific prediction.'),
            ('Counterfactual', 'A "what-if" scenario showing what changes would make an employee stay.'),
        ]
        
        for term, definition in ai_metrics:
            st.markdown(f"""
            <div class="glossary-item">
                <div class="glossary-term">{term}</div>
                <div class="glossary-definition">{definition}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Financial Metrics
        st.markdown("### 💰 Financial Metrics")
        financial_metrics = [
            ('Cost to Retain', 'Typically 10% of annual salary. The cost of interventions to keep an employee.'),
            ('Expected Loss', 'Risk Score × 50% of Salary. The expected cost if the employee leaves.'),
            ('Net Savings', 'Expected Loss - Cost to Retain. Positive = Retention is cheaper than replacement.'),
            ('Replacement Cost', 'Typically 50-200% of annual salary, depending on role and tenure.'),
            ('ROI Optimizer', 'Mathematical algorithm that finds the best employees to retain within a budget.'),
        ]
        
        for term, definition in financial_metrics:
            st.markdown(f"""
            <div class="glossary-item">
                <div class="glossary-term">{term}</div>
                <div class="glossary-definition">{definition}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("### 📈 Model Performance Terms")
        model_metrics = [
            ('Accuracy', 'Percentage of correct predictions out of all predictions.'),
            ('Precision', 'Of all "Leave" predictions, how many actually left. Important to avoid false alarms.'),
            ('Recall', 'Of all who actually left, how many did we correctly predict. Important to not miss at-risk employees.'),
            ('F1 Score', 'Balance between Precision and Recall. Higher is better.'),
            ('ROC AUC', 'Overall model quality measure. 0.5 = random guessing, 1.0 = perfect predictions.'),
        ]
        
        for term, definition in model_metrics:
            st.markdown(f"""
            <div class="glossary-item">
                <div class="glossary-term">{term}</div>
                <div class="glossary-definition">{definition}</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
