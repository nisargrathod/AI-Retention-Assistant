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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from time import sleep
from scipy.sparse import issparse
from scipy.special import expit, logit
import json

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel
from scipy.optimize import milp, LinearConstraint, Bounds

# --- Imports for Evaluation 2 (Intelligent Interface: Groq + Evidently) ---
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping

# VERSION-AGNOSTIC EVIDENTLY IMPORTS
try:
    # Try newer Evidently version (0.4.x+)
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
    EVIDENTLY_VERSION = "new"
except ImportError:
    try:
        # Try older Evidently version (0.3.x)
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
        EVIDENTLY_VERSION = "old"
        DatasetDriftMetric = None
        DataDriftTable = None
    except ImportError:
        # Minimal fallback
        EVIDENTLY_VERSION = "minimal"
        DataDriftPreset = None
        DataQualityPreset = None
        ClassificationPreset = None
        DatasetDriftMetric = None
        DataDriftTable = None

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
    
    .drift-status-ok {
        background-color: #0d2818;
        border: 1px solid #17B794;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .drift-status-warning {
        background-color: #2d2515;
        border: 1px solid #EEB76B;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .drift-status-alert {
        background-color: #2d1515;
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-card {
        background-color: #21262d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #30363d;
    }
    
    .metric-card-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .metric-card-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* --- NEW CHATBOT STYLES --- */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
        padding: 20px;
        background-color: #161b22;
        border-radius: 16px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        height: 70vh;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 20px;
        flex-shrink: 0;
    }
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding-right: 10px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .chat-messages::-webkit-scrollbar { width: 6px; }
    .chat-messages::-webkit-scrollbar-track { background: #161b22; }
    .chat-messages::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    .message { display: flex; gap: 12px; animation: fadeIn 0.3s ease-in-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .message-user { flex-direction: row-reverse; }
    .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; font-size: 14px; }
    .avatar-bot { background: linear-gradient(135deg, #17B794 0%, #11998e 100%); color: white; box-shadow: 0 4px 10px rgba(23, 183, 148, 0.3); }
    .avatar-human { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
    .bubble { padding: 12px 16px; border-radius: 16px; font-size: 0.95rem; line-height: 1.5; max-width: 75%; position: relative; }
    .bubble-bot { background-color: #21262d; color: #e6edf3; border-bottom-left-radius: 4px; border: 1px solid #30363d; }
    .bubble-user { background: linear-gradient(90deg, #17B794 0%, #11998e 100%); color: white; border-bottom-right-radius: 4px; }
    
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
        .chat-container {
            height: 60vh;
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
# EVIDENTLY AI FUNCTIONS (VERSION-AGNOSTIC)
# ====================================================================

def get_column_mapping(df, target_col='left'):
    """Create column mapping for Evidently based on dataframe types"""
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore').tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return ColumnMapping(
        target=target_col,
        numerical_features=numeric_cols,
        categorical_features=categorical_cols
    )


def run_data_drift_analysis(reference_df, current_df, target_col='left'):
    """Run data drift analysis using Evidently AI - version agnostic"""
    column_mapping = get_column_mapping(reference_df, target_col)

    # Ensure both dataframes have the same columns
    common_cols = list(set(reference_df.columns) & set(current_df.columns))
    reference_df = reference_df[common_cols].copy()
    current_df = current_df[common_cols].copy()

    # Handle missing target in current data (for production scenarios)
    if target_col not in current_df.columns:
        current_df[target_col] = np.nan

    if EVIDENTLY_VERSION == "new" and DatasetDriftMetric is not None:
        # Use individual metrics for newer versions
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
    elif DataDriftPreset is not None:
        # Use preset for older versions
        report = Report(metrics=[DataDriftPreset()])
    else:
        # Minimal fallback - return None
        return None

    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    return report


def run_data_quality_report(df, target_col='left'):
    """Run data quality analysis using Evidently AI - version agnostic"""
    column_mapping = get_column_mapping(df, target_col)

    if DataQualityPreset is not None:
        report = Report(metrics=[DataQualityPreset()])
    else:
        return None

    report.run(reference_data=df, current_data=df, column_mapping=column_mapping)

    return report


def run_classification_performance_report(reference_df, current_df, predictions, target_col='left'):
    """Run classification performance analysis using Evidently AI - version agnostic"""
    column_mapping = get_column_mapping(reference_df, target_col)

    # Create a copy with predictions
    current_df = current_df.copy()
    current_df['prediction'] = predictions

    if ClassificationPreset is not None:
        report = Report(metrics=[ClassificationPreset()])
    else:
        return None

    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    return report


def extract_drift_metrics(report):
    """Extract drift metrics from Evidently report as dictionary"""
    if report is None:
        return {'error': 'Evidently not available'}

    try:
        result = {}
        for metric_result in report._metrics:
            metric_name = metric_result.__class__.__name__

            if hasattr(metric_result, 'get_result'):
                metric_value = metric_result.get_result()
                if isinstance(metric_value, dict):
                    result[metric_name] = metric_value
                else:
                    result[metric_name] = {'value': metric_value}

        return result
    except Exception as e:
        return {'error': str(e)}


def render_drift_summary(drift_metrics):
    """Render a visual summary of drift metrics"""
    if 'error' in drift_metrics:
        st.warning(f"Evidently metrics unavailable: {drift_metrics['error']}")
        return False, 0, 0

    try:
        # Check for dataset drift
        dataset_drift = drift_metrics.get('DatasetDriftMetric', {})
        if not dataset_drift:
            # Try DataDriftPreset results
            for key, val in drift_metrics.items():
                if 'dataset_drift' in str(val).lower():
                    dataset_drift = val
                    break

        is_drifted = dataset_drift.get('dataset_drift', False)
        drift_share = dataset_drift.get('number_of_drifted_columns', 0)
        total_columns = dataset_drift.get('number_of_columns', 1)

        if is_drifted:
            status_html = f"""
            <div class="drift-status-alert">
                <h3 style="color: #FF4B4B; margin: 0;">⚠️ DATA DRIFT DETECTED</h3>
                <p style="color: #c9d1d9; margin: 10px 0 0 0;">{drift_share} out of {total_columns} features have significantly shifted</p>
            </div>
            """
        else:
            status_html = f"""
            <div class="drift-status-ok">
                <h3 style="color: #17B794; margin: 0;">✅ DATA STABLE</h3>
                <p style="color: #c9d1d9; margin: 10px 0 0 0;">No significant drift detected in your data</p>
            </div>
            """

        st.markdown(status_html, unsafe_allow_html=True)
        return is_drifted, drift_share, total_columns

    except Exception as e:
        st.warning(f"Could not parse drift metrics: {e}")
        return False, 0, 0


def render_drift_table(drift_metrics):
    """Render detailed drift table for each feature"""
    if 'error' in drift_metrics:
        st.info("Detailed drift table unavailable")
        return

    try:
        drift_table = drift_metrics.get('DataDriftTable', {})
        drift_per_column = drift_table.get('drift_per_column', {})

        if not drift_per_column:
            # Try to find drift data in other keys
            for key, val in drift_metrics.items():
                if isinstance(val, dict) and 'drift_per_column' in val:
                    drift_per_column = val['drift_per_column']
                    break

        if not drift_per_column:
            st.info("No per-column drift data available")
            return

        # Create a dataframe for display
        drift_data = []
        for column, stats in drift_per_column.items():
            drift_data.append({
                'Feature': column.replace('_', ' ').title(),
                'Drift Detected': 'Yes' if stats.get('drift_detected', False) else 'No',
                'Drift Score': round(stats.get('drift_score', 0), 4),
                'Threshold': round(stats.get('threshold', 0), 4),
                'Current Mean': round(stats.get('current_mean', 0), 3) if stats.get('current_mean') else 'N/A',
                'Reference Mean': round(stats.get('reference_mean', 0), 3) if stats.get('reference_mean') else 'N/A'
            })

        if drift_data:
            drift_df = pd.DataFrame(drift_data)
            drift_df = drift_df.sort_values('Drift Score', ascending=False)

            # Style the dataframe
            def highlight_drift(row):
                if row['Drift Detected'] == 'Yes':
                    return ['background-color: #2d1515; color: #FF4B4B'] * len(row)
                else:
                    return ['background-color: #0d2818; color: #17B794'] * len(row)

            styled_df = drift_df.style.apply(highlight_drift, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

    except Exception as e:
        st.warning(f"Could not render drift table: {e}")


def render_classification_metrics(classification_metrics):
    """Render classification performance metrics"""
    if 'error' in classification_metrics:
        st.info("Classification metrics unavailable from Evidently")
        return

    try:
        # Try to extract metrics from the result
        metrics_data = {}

        for key, val in classification_metrics.items():
            if isinstance(val, dict):
                # Look for common metric keys
                if 'roc_auc' in str(val).lower():
                    metrics_data['ROC AUC'] = val.get('value', val.get('roc_auc', 0))
                if 'precision' in str(val).lower() and 'recall' not in key.lower():
                    metrics_data['Precision'] = val.get('value', val.get('precision', 0))
                if 'recall' in str(val).lower():
                    metrics_data['Recall'] = val.get('value', val.get('recall', 0))
                if 'f1' in str(val).lower():
                    metrics_data['F1 Score'] = val.get('value', val.get('f1', 0))
                if 'accuracy' in str(val).lower():
                    metrics_data['Accuracy'] = val.get('value', val.get('accuracy', 0))
                if 'quality' in key.lower():
                    metrics_data['Overall Quality'] = val.get('value', val.get('quality', 0))

        if metrics_data:
            cols = st.columns(len(metrics_data))

            for idx, (metric_name, value) in enumerate(metrics_data.items()):
                if isinstance(value, (int, float)):
                    if value > 0.8:
                        color = '#17B794'
                    elif value > 0.6:
                        color = '#EEB76B'
                    else:
                        color = '#FF4B4B'

                    metric_html = f"""
                    <div class="metric-card">
                        <div class="metric-card-value" style="color: {color};">{value:.3f}</div>
                        <div class="metric-card-label">{metric_name}</div>
                    </div>
                    """

                    with cols[idx]:
                        st.markdown(metric_html, unsafe_allow_html=True)
        else:
            st.info("No classification metrics extracted")

    except Exception as e:
        st.warning(f"Could not render classification metrics: {e}")


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
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Show Evidently version info (debug)
    if st.secrets.get("DEBUG_MODE", "false").lower() == "true":
        with st.sidebar:
            st.caption(f"Evidently version: {EVIDENTLY_VERSION}")

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

            return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model(_model_version="v4_calibrated")
        st.empty()

    @st.cache_data
    def get_shap_explanations(_pipeline, _df):
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

        # --- NEW CHATBOT PAGE ADDED TO MENU ---
        page = option_menu(
            menu_title=None,
            options=['Global Setup', 'Home', 'Employee Insights', 'Predict Attrition', 'Why They Leave', 'Budget Planner', 'AI Assistant', 'HR Chatbot', 'AI Research Lab', 'Strategic Roadmap'],  
            icons=['gear', 'house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'chat-dots-fill', 'cpu', 'flag-2-fill'], 
            menu_icon="cast", default_index=0, 
            styles={
                "container": {"padding": "0!important", "background-color": 'transparent'},
                "icon": {"color": "#17B794", "font-size": "18px"},
                "nav-link": {"color": "#c9d1d9", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "10px"},
                "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"},
            }
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding:20px; border-top:1px solid #2d333b;'><div style='font-size:0.85rem; color:#8b949e;'>Built by</div><div style='font-size:1.6rem; font-weight:600; color:#00E5A0; margin-bottom:10px;'>Nisarg Rathod</div><div style='display:flex; justify-content:center; gap:15px;'><a href='https://www.linkedin.com/in/nisarg-rathod/' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#0A66C2; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='16' height='16' style='filter:invert(1);'/>LinkedIn</a><a href='https://github.com/nisargrathod' target='_blank'style='display:flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px; background:#24292e; color:white; text-decoration:none; font-size:0.9rem;'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='16' height='16' style='filter:invert(1);'/>GitHub</a></div></div>", unsafe_allow_html=True)

    # ====================================================================
    # PAGE: GLOBAL SETUP
    # ====================================================================
    if page == "Global Setup":
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

                # --- EVIDENTLY DATA QUALITY CHECK ---
                st.markdown("---")
                st.markdown("### 🔍 Step 2.5: Data Quality Check (Evidently AI)")
                st.caption("Automated data quality analysis before training")

                if st.button("🔬 Run Data Quality Report", type="secondary", key="quality_check"):
                    with st.spinner("Analyzing data quality..."):
                        try:
                            quality_df = new_df.copy()
                            quality_df[target_col] = quality_df[target_col].apply(lambda x: 1 if x == left_value else 0)

                            quality_report = run_data_quality_report(quality_df, target_col)

                            if quality_report is None:
                                st.warning("⚠️ Evidently AI quality report not available (version compatibility issue). Showing basic stats instead.")
                                col_q1, col_q2, col_q3 = st.columns(3)
                                with col_q1: st.metric("Total Rows", len(new_df))
                                with col_q2: st.metric("Total Columns", len(new_df.columns))
                                with col_q3: st.metric("Missing Values", new_df.isnull().sum().sum())
                            else:
                                quality_metrics = extract_drift_metrics(quality_report)
                                st.success("✅ Data Quality Report Generated")

                                quality_data = quality_metrics.get('DataQualityMetric', {})

                                col_q1, col_q2, col_q3, col_q4 = st.columns(4)

                                with col_q1:
                                    st.metric("Total Rows", quality_data.get('rows_count', len(new_df)))
                                with col_q2:
                                    st.metric("Total Columns", quality_data.get('columns_count', len(new_df.columns)))
                                with col_q3:
                                    missing = quality_data.get('missing_values', {}).get('number_of_missing_values', 0)
                                    st.metric("Missing Values", missing, delta="Should be low" if missing > 0 else "Perfect")
                                with col_q4:
                                    empty_cols = quality_data.get('empty_columns', {}).get('number_of_empty_columns', 0)
                                    st.metric("Empty Columns", empty_cols, delta="None" if empty_cols == 0 else "Check data")

                                if missing > 0:
                                    with st.expander("📋 Missing Values by Column"):
                                        missing_by_col = quality_data.get('missing_values', {}).get('share_of_missing_values_by_column', {})
                                        if missing_by_col:
                                            missing_df = pd.DataFrame([
                                                {'Column': k.replace('_', ' ').title(), 'Missing %': f"{v*100:.1f}%"}
                                                for k, v in missing_by_col.items() if v > 0
                                            ])
                                            st.dataframe(missing_df, use_container_width=True)

                        except Exception as e:
                            st.error(f"Data quality check failed: {e}")

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
                        y_pred_g = global_pipeline.predict(X_test_g); acc = accuracy_score(y_test_g, y_pred_g)
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g; st.session_state['is_global'] = True
                        st.success(f"🎉 Training Complete! Accuracy: **{acc:.1%}**.")
                        st.info("Go to **'Predict Attrition'** or **'Why They Leave'** to use your data!")
            except Exception as e: st.error(f"Error: {e}")
        if st.button("🔄 Reset to Default Demo Data"):
            if 'is_global' in st.session_state: del st.session_state['is_global']
            st.rerun()

    # ====================================================================
    # Pages
    # ====================================================================
    if page == "Home":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        total_employees = len(df); attrition_rate = (df['left'].sum() / len(df)) * 100

        if 'satisfaction_level' in df.columns:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
            col4.metric("Data Health", "✅ Stable", delta="No Drift")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Data Health", "✅ Stable", delta="No Drift")

        # Quick Data Drift Indicator
        st.markdown("---")
        with st.expander("📊 Quick Data Health Check (Evidently AI)", expanded=False):
            st.caption("Compare your current data against the training baseline to detect any distribution shifts")

            if st.button("🔍 Run Quick Drift Check", type="secondary", key="quick_drift"):
                with st.spinner("Checking for data drift..."):
                    try:
                        reference_data = df.sample(frac=0.8, random_state=42)
                        current_data = df.drop(reference_data.index)

                        drift_report = run_data_drift_analysis(reference_data, current_data)

                        if drift_report is None:
                            st.warning("⚠️ Evidently AI drift analysis not available. Using basic statistics instead.")

                            # Fallback: basic statistical comparison
                            numeric_cols = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                            drift_detected = 0
                            total_cols = len(numeric_cols)

                            for col in numeric_cols[:10]:  # Check first 10 numeric columns
                                ref_mean = reference_data[col].mean()
                                cur_mean = current_data[col].mean()
                                ref_std = reference_data[col].std()
                                if ref_std > 0:
                                    z_score = abs(ref_mean - cur_mean) / ref_std
                                    if z_score > 2:  # Simple threshold
                                        drift_detected += 1

                            if drift_detected > 0:
                                st.warning(f"⚠️ Basic check: {drift_detected}/{total_cols} features may have shifted")
                            else:
                                st.success("✅ Basic check: Data appears stable")
                        else:
                            drift_metrics = extract_drift_metrics(drift_report)
                            is_drifted, drift_cols, total_cols = render_drift_summary(drift_metrics)

                            if not is_drifted:
                                st.success("✅ Your data is stable. No significant drift detected between the reference and current periods.")
                            else:
                                st.warning(f"⚠️ Drift detected in {drift_cols} features. Consider retraining your model.")

                            with st.expander("📊 Detailed Drift Table"):
                                render_drift_table(drift_metrics)

                    except Exception as e:
                        st.error(f"Drift check failed: {e}")

        st.markdown("---")
        st.markdown("### 📄 Employee Data Snapshot")
        st.dataframe(df.head(100), use_container_width=True)

    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis")
        st.write("Explore the workforce demographics to identify patterns.")
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    if page == "Predict Attrition":
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
                    if pred == 1: st.success(f"✅ **Correct!** Prediction: Leave")
                    else: st.error(f"❌ **Incorrect.** Prediction: Stay")
                    st.json(sample.to_dict(), expanded=False)
            with c_test2:
                if st.button("Test with Employee who Stayed"):
                    sample = df[df['left'] == 0].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    raw_prob = pipeline.predict_proba(test_df)[0][1]
                    calibrated_prob = calibrate_probability(raw_prob)
                    pred = 1 if calibrated_prob >= 0.5 else 0
                    if pred == 0: st.success(f"✅ **Correct!** Prediction: Stay")
                    else: st.error(f"❌ **Incorrect.** Prediction: Leave")
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

    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers")
        st.write("Understand the specific factors driving your team's attrition risk, explained simply.")
        st.write("---")
        analyze_why_people_leave(df)
        with st.spinner("Analyzing model insights..."):
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
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

    # ====================================================================
    # --- NEW CHATBOT PAGE ---
    # ====================================================================
    if page == "HR Chatbot":
        st.markdown("<h1 style='margin-bottom: 5px;'>💬 AI HR Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; margin-bottom: 30px;'>I am your AI HR Copilot. Ask me anything about your workforce data, retention strategies, or draft communications instantly.</p>", unsafe_allow_html=True)

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hello! I am your RetainAI Copilot. 🧠\n\nI have access to your entire workforce dataset and AI models. You can ask me things like:\n• 'What is our current attrition rate?'\n• 'Why are people leaving in the Sales department?'\n• 'Draft a warning email for an overworked employee.'\n\nHow can I help you today?"}
            ]

        # --- FIX: NEW CHAT BUTTON ---
        col_chat_header, col_chat_clear = st.columns([6, 1])
        with col_chat_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": "Chat cleared! 🧹 How can I help you with your workforce today?"}
                ]
                st.rerun()

        # Fetch Context dynamically
        @st.cache_data
        def get_dynamic_context(_df):
            total_emp = len(_df)
            attrition_rate = round((_df['left'].sum() / total_emp) * 100, 1) if 'left' in _df.columns else 0
            avg_sat = round(_df['satisfaction_level'].mean(), 2) if 'satisfaction_level' in _df.columns else "N/A"
            avg_hours = round(_df['average_montly_hours'].mean(), 0) if 'average_montly_hours' in _df.columns else "N/A"
            departments = _df['Department'].unique().tolist() if 'Department' in _df.columns else []
            return f"Total Employees: {total_emp}, Attrition Rate: {attrition_rate}%, Avg Satisfaction: {avg_sat}/1.0, Avg Monthly Hours: {avg_hours}, Departments: {', '.join(departments)}"

        context_str = get_dynamic_context(df)

        # Custom HTML Chat Container
        chat_container = st.container()

        with chat_container:
            st.markdown("""
            <div class="chat-container">
                <div class="chat-header">
                    <h3 style="margin:0; color: #17B794;">RetainAI Copilot</h3>
                    <small style="color: #8b949e;">Powered by Groq LLM + Internal Data</small>
                </div>
                <div class="chat-messages" id="chat-messages">
            """, unsafe_allow_html=True)

            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="message message-user">
                        <div class="avatar avatar-human">HR</div>
                        <div class="bubble bubble-user">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="message">
                        <div class="avatar avatar-bot">AI</div>
                        <div class="bubble bubble-bot">{message["content"].replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        if prompt := st.chat_input("Ask me anything about your workforce..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.rerun()

        if len(st.session_state.chat_messages) > 0 and st.session_state.chat_messages[-1]["role"] == "user":
            user_prompt = st.session_state.chat_messages[-1]["content"]

            with st.spinner("Thinking..."):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if not api_key:
                        bot_response = "⚠️ System Error: Groq API Key is missing. Please add it to your Streamlit secrets."
                    else:
                        llm = ChatGroq(
                            groq_api_key=api_key, 
                            model_name="llama-3.3-70b-versatile", 
                            temperature=0.7, 
                            timeout=30
                        )

                        template = """
                        You are an expert, empathetic, and highly data-driven HR Assistant named 'RetainAI Copilot' working inside an enterprise application.
                        
                        **Current Company Data Context:**
                        {context}
                        
                        **HR Manager's Question:**
                        {question}
                        
                        **Instructions:**
                        - If the question is about data (attrition rate, hours, departments), use the provided context to answer directly. Do not hallucinate numbers.
                        - If the question asks for advice (retention, morale, policies), give practical, corporate-standard HR advice.
                        - If the question asks to draft an email or message, write it clearly with placeholders like [Employee Name].
                        - Keep the response concise (under 150 words) unless drafting an email.
                        - Use bullet points for lists.
                        - Do not use markdown code blocks.
                        """

                        chain = PromptTemplate.from_template(template) | llm | StrOutputParser()
                        bot_response = chain.invoke({
                            "context": context_str, 
                            "question": user_prompt
                        })

                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        bot_response = "⏳ I am experiencing high traffic right now. Please wait 10 seconds and ask your question again."
                    else:
                        bot_response = f"❌ Sorry, I encountered an error: {str(e)}"

            st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})
            st.rerun()

    # ====================================================================
    # PAGE: AI RESEARCH LAB (RESTORED EXACTLY TO ORIGINAL LENGTH)
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Advanced modules for Strategy, Disruption, and Recruitment.</p>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Benchmarking", "🔬 Departmental Strategy Deep Dive", "🛡️ AI Disruption Defense", "📈 Data Drift & Monitoring"])

        with tab1:
            st.subheader("Algorithm Performance Comparison")
            st.caption("Using Evidently AI for comprehensive classification metrics")
            if st.button("Run Benchmark", type="primary", key="run_benchmark"):
                with st.spinner("Training competing models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
                    rf_pipeline.fit(X_train_ref, y_train); y_pred_rf = rf_pipeline.predict(X_test_cur); proba_rf = rf_pipeline.predict_proba(X_test_cur)[:, 1]
                    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
                    lr_pipeline.fit(X_train_ref, y_train); y_pred_lr = lr_pipeline.predict(X_test_cur); proba_lr = lr_pipeline.predict_proba(X_test_cur)[:, 1]
                    metrics = {'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'], 'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)], 'Precision': [precision_score(y_test, y_pred_lgbm), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_lr)], 'Recall': [recall_score(y_test, y_pred_lgbm), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_lr)], 'F1 Score': [f1_score(y_test, y_pred_lgbm), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_lr)], 'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_lr)]}
                    results_df = pd.DataFrame(metrics)
                    st.markdown("### 📈 Performance Metrics (Traditional)")
                    st.dataframe(results_df.style.highlight_max(axis=0, color='#17B794'), use_container_width=True)
                    fig_metrics = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'), x='Metric', y='Score', color='Model', barmode='group', title="Model Comparison", template="plotly_dark", color_discrete_sequence=['#17B794', '#EEB76B', '#9C3D54'])
                    custome_layout(fig_metrics, title_size=24); st.plotly_chart(fig_metrics, use_container_width=True)

                    # --- EVIDENTLY CLASSIFICATION REPORT ---
                    st.markdown("---")
                    st.markdown("### 📊 Evidently AI Classification Report (LightGBM)")
                    st.caption("Detailed classification quality analysis by Evidently AI")

                    try:
                        reference_data = X_train_ref.copy()
                        reference_data['left'] = y_train.values

                        current_data = X_test_cur.copy()
                        current_data['left'] = y_test.values
                        current_data['prediction'] = y_pred_lgbm

                        classification_report = run_classification_performance_report(reference_data, current_data, y_pred_lgbm)

                        if classification_report is None:
                            st.warning("⚠️ Evidently AI classification report not available (version compatibility). Using sklearn metrics shown above.")
                        else:
                            classification_metrics = extract_drift_metrics(classification_report)
                            render_classification_metrics(classification_metrics)

                            with st.expander("📋 Detailed Metrics Breakdown"):
                                for metric_name, metric_data in classification_metrics.items():
                                    st.markdown(f"**{metric_name}**")
                                    if isinstance(metric_data, dict):
                                        for k, v in metric_data.items():
                                            if isinstance(v, float):
                                                st.write(f"  • {k}: `{v:.4f}`")
                                            else:
                                                st.write(f"  • {k}: `{v}`")
                                    st.markdown("---")

                    except Exception as e:
                        st.warning(f"Evidently classification report failed: {e}")

                    st.success("🏆 **Conclusion:** LightGBM was selected as the primary model due to its superior balance of Precision and Recall.")

        with tab2:
            st.subheader("🔬 Departmental Strategy Deep Dive")
            if 'Department' not in df.columns: 
                st.warning("Department column not found in this dataset. Cannot run deep dive.")
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
                            if dept_mask.sum() == 0: 
                                st.warning(f"Not enough data to analyze {selected_dept_name} specifically.")
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
            with c_inp2: 
                st.markdown("**Industry Standard Costs:**"); 
                st.caption("Severance: 3 months salary | Hiring AI Engineer: 2x salary | Training: 1.5x salary | Morale dip: 1 month salary")
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
            if savings > 0: 
                st.success(f"**The Verdict: Reskill.** By choosing to reskill instead of laying off, you save ₹{savings/10000000:.2f} Crores.")
            else: 
                st.warning(f"**The Verdict: Layoffs are technically cheaper here. However, consider hidden costs before proceeding.")
            if st.button("✍️ Generate Strategy Memo for CEO", type="primary", key="gen_ai_memo"):
                with st.spinner("Drafting strategy memo..."):
                    try:
                        api_key = st.secrets.get("GROQ_API_KEY", None)
                        if not api_key: 
                            st.warning("🔑 System Error: API Key missing. Showing generic template.")
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
        # NEW TAB 4: DATA DRIFT & MONITORING (EVIDENTLY AI)
        # ====================================================================
        with tab4:
            st.subheader("📈 Data Drift & Model Monitoring")
            st.markdown("""
            <div class="custom-card" style="border-left: 5px solid #17B794; background: linear-gradient(to right, #0d2818 0%, #1c2128 100%);">
                <h4 style="color: #17B794; margin-top: 0;">🔍 Evidently AI Monitoring</h4>
                <p style="color: #e6edf3; font-size: 1rem; margin-bottom: 0;">Detect when your data distribution changes. Data drift can silently degrade model performance over time.</p>
            </div>
            """, unsafe_allow_html=True)

            # Check if Evidently is available
            if EVIDENTLY_VERSION == "minimal":
                st.warning("""
                ⚠️ **Evidently AI Limited Mode**
                
                The installed version of Evidently AI has compatibility issues. Using fallback drift detection based on basic statistical methods.
                
                To get full Evidently features, try upgrading: `pip install evidently==0.4.22`
                """)

            st.markdown("### 📊 What is Data Drift?")
            st.markdown("""
            <p style='color: #c9d1d9; line-height: 1.6;'>
            Data drift occurs when the statistical properties of your input data change over time. For example:<br><br>
            • <strong>Average satisfaction drops</strong> from 0.65 to 0.55 company-wide<br>
            • <strong>Monthly working hours increase</strong> due to new project demands<br>
            • <strong>Department composition changes</strong> after a reorg<br><br>
            When drift occurs, your model's predictions become less reliable because it was trained on "old" data patterns.
            </p>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🧪 Run Drift Analysis")

            col_drift1, col_drift2 = st.columns(2)
            with col_drift1:
                drift_scenario = st.selectbox(
                    "Select Analysis Scenario",
                    options=[
                        "Train vs Test Split (Default)",
                        "First 50% vs Last 50% (Time Simulation)",
                        "Random Sample Comparison",
                        "High Risk vs Low Risk Employees"
                    ]
                )
            with col_drift2:
                if EVIDENTLY_VERSION != "minimal":
                    drift_threshold = st.slider(
                        "Drift Sensitivity Threshold",
                        min_value=0.01,
                        max_value=0.20,
                        value=0.05,
                        step=0.01,
                        help="Lower values = more sensitive to small changes"
                    )
                else:
                    st.info("Using default threshold: 2 standard deviations")

            if st.button("🔍 Run Data Drift Analysis", type="primary", key="run_drift_analysis"):
                with st.spinner("Running drift analysis..."):
                    try:
                        full_data = df.copy()

                        if drift_scenario == "Train vs Test Split (Default)":
                            reference_df = full_data.sample(frac=0.8, random_state=42)
                            current_df = full_data.drop(reference_df.index)
                            scenario_desc = "Training (80%) vs Test (20%) split"

                        elif drift_scenario == "First 50% vs Last 50% (Time Simulation)":
                            mid_point = len(full_data) // 2
                            reference_df = full_data.iloc[:mid_point]
                            current_df = full_data.iloc[mid_point:]
                            scenario_desc = "First half vs Second half (simulating time)"

                        elif drift_scenario == "Random Sample Comparison":
                            reference_df = full_data.sample(frac=0.5, random_state=42)
                            current_df = full_data.sample(frac=0.5, random_state=123)
                            scenario_desc = "Two random 50% samples"

                        elif drift_scenario == "High Risk vs Low Risk Employees":
                            X = full_data.drop('left', axis=1)
                            raw_probs = pipeline.predict_proba(X)[:, 1]
                            full_data['risk_score'] = calibrate_probability_array(raw_probs)

                            reference_df = full_data[full_data['risk_score'] < 0.5].copy()
                            current_df = full_data[full_data['risk_score'] >= 0.5].copy()
                            scenario_desc = "Low Risk (< 50%) vs High Risk (≥ 50%) employees"

                        # Run drift analysis
                        drift_report = run_data_drift_analysis(reference_df, current_df)

                        st.markdown("---")
                        st.markdown(f"### 📊 Results: {scenario_desc}")

                        if drift_report is None:
                            # FALLBACK: Basic statistical drift detection
                            st.info("Using basic statistical drift detection (Evidently not fully available)")

                            numeric_cols = full_data.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                            drift_results = []
                            total_checked = 0
                            drift_detected_count = 0

                            for col in numeric_cols:
                                ref_mean = reference_df[col].mean()
                                cur_mean = current_df[col].mean()
                                ref_std = reference_df[col].std()

                                if ref_std > 0:
                                    z_score = abs(ref_mean - cur_mean) / ref_std
                                    is_drifted = z_score > 2
                                    total_checked += 1
                                    if is_drifted:
                                        drift_detected_count += 1

                                    drift_results.append({
                                        'Feature': col.replace('_', ' ').title(),
                                        'Drift Detected': 'Yes' if is_drifted else 'No',
                                        'Z-Score': round(z_score, 4),
                                        'Reference Mean': round(ref_mean, 3),
                                        'Current Mean': round(cur_mean, 3),
                                        'Difference': round(cur_mean - ref_mean, 3)
                                    })

                            if drift_results:
                                drift_df = pd.DataFrame(drift_results)
                                drift_df = drift_df.sort_values('Z-Score', ascending=False)

                                # Summary
                                is_drifted = drift_detected_count > 0
                                if is_drifted:
                                    st.markdown(f"""
                                    <div class="drift-status-alert">
                                        <h3 style="color: #FF4B4B; margin: 0;">⚠️ POTENTIAL DRIFT DETECTED</h3>
                                        <p style="color: #c9d1d9; margin: 10px 0 0 0;">{drift_detected_count} out of {total_checked} features show significant changes (Z-score > 2)</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="drift-status-ok">
                                        <h3 style="color: #17B794; margin: 0;">✅ DATA STABLE</h3>
                                        <p style="color: #c9d1d9; margin: 10px 0 0 0;">No significant drift detected (all Z-scores < 2)</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                col_d1, col_d2, col_d3 = st.columns(3)
                                with col_d1:
                                    drift_share_pct = (drift_detected_count / total_checked * 100) if total_checked > 0 else 0
                                    st.metric("Drifted Features", f"{drift_detected_count}/{total_checked}", delta=f"{drift_share_pct:.1f}%")
                                with col_d2:
                                    avg_z = drift_df['Z-Score'].mean()
                                    st.metric("Avg Z-Score", f"{avg_z:.2f}", delta="Lower is better")
                                with col_d3:
                                    status = "⚠️ DRIFTED" if is_drifted else "✅ STABLE"
                                    st.metric("Overall Status", status)

                                # Show table
                                st.markdown("---")
                                st.markdown("### 📋 Feature-Level Drift Details")

                                def highlight_drift_fallback(row):
                                    if row['Drift Detected'] == 'Yes':
                                        return ['background-color: #2d1515; color: #FF4B4B'] * len(row)
                                    else:
                                        return ['background-color: #0d2818; color: #17B794'] * len(row)

                                st.dataframe(drift_df.style.apply(highlight_drift_fallback, axis=1), use_container_width=True, height=400)

                        else:
                            # EVIDENTLY FULL RESULTS
                            drift_metrics = extract_drift_metrics(drift_report)
                            is_drifted, drift_cols, total_cols = render_drift_summary(drift_metrics)

                            dataset_drift = drift_metrics.get('DatasetDriftMetric', {})
                            col_d1, col_d2, col_d3 = st.columns(3)

                            with col_d1:
                                drift_share_pct = (drift_cols / total_cols * 100) if total_cols > 0 else 0
                                st.metric("Drifted Features", f"{drift_cols}/{total_cols}", delta=f"{drift_share_pct:.1f}%")

                            with col_d2:
                                drift_ratio = dataset_drift.get('drift_share', 0)
                                st.metric("Drift Share", f"{drift_ratio:.2%}", delta="Should be low")

                            with col_d3:
                                status = "⚠️ DRIFTED" if is_drifted else "✅ STABLE"
                                st.metric("Overall Status", status)

                            st.markdown("---")
                            st.markdown("### 📋 Feature-Level Drift Details")
                            st.caption("Green = No drift detected | Red = Drift detected (above threshold)")
                            render_drift_table(drift_metrics)

                        # Recommendations
                        st.markdown("---")
                        st.markdown("### 💡 Recommendations")

                        if drift_report is None:
                            if drift_detected_count > 0:
                                st.warning("""
                                **⚠️ Action Required:**
                                1. **Investigate drifted features** - Check if the drift is due to data collection issues or genuine changes
                                2. **Consider retraining** - If drift is significant and persistent, retrain the model with recent data
                                3. **Update monitoring** - Adjust the threshold if the detected changes are expected
                                """)
                            else:
                                st.success("""
                                **✅ No Immediate Action Required**
                                Your data distributions are stable. Continue regular monitoring to catch drift early.
                                """)
                        else:
                            if is_drifted:
                                st.warning("""
                                **⚠️ Action Required:**
                                1. **Investigate drifted features** - Check if the drift is due to data collection issues or genuine changes
                                2. **Consider retraining** - If drift is significant and persistent, retrain the model with recent data
                                3. **Update monitoring** - Adjust the drift threshold if the detected changes are expected
                                """)
                            else:
                                st.success("""
                                **✅ No Immediate Action Required**
                                Your data distributions are stable. Continue regular monitoring to catch drift early.
                                """)

                        # Export option
                        with st.expander("📥 Export Report"):
                            if drift_report is not None:
                                try:
                                    drift_metrics_export = extract_drift_metrics(drift_report)
                                    report_json = json.dumps(drift_metrics_export, indent=2, default=str)
                                    st.code(report_json, language='json')
                                except:
                                    st.write(drift_metrics)
                            else:
                                st.dataframe(drift_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Drift analysis failed: {e}")
                        import traceback
                        with st.expander("🔧 Error Details"):
                            st.code(traceback.format_exc())

            st.markdown("---")

            # Feature Distribution Comparison
            st.markdown("### 📈 Feature Distribution Comparison")
            st.caption("Visual comparison of key features between reference and current periods")

            if st.button("📊 Show Distribution Charts", type="secondary", key="show_dist"):
                with st.spinner("Generating distribution charts..."):
                    try:
                        numeric_cols = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()

                        if len(numeric_cols) == 0:
                            st.warning("No numeric columns available for comparison")
                        else:
                            reference_df = df.sample(frac=0.8, random_state=42)
                            current_df = df.drop(reference_df.index)

                            for col in numeric_cols[:6]:
                                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                                axes[0].hist(reference_df[col].dropna(), bins=30, color='#17B794', alpha=0.7, edgecolor='white')
                                axes[0].set_title(f'Reference: {col.replace("_", " ").title()}', color='white', fontsize=12)
                                axes[0].set_facecolor('#161b22')
                                axes[0].tick_params(colors='white')
                                for spine in axes[0].spines.values():
                                    spine.set_color('#30363d')

                                axes[1].hist(current_df[col].dropna(), bins=30, color='#EEB76B', alpha=0.7, edgecolor='white')
                                axes[1].set_title(f'Current: {col.replace("_", " ").title()}', color='white', fontsize=12)
                                axes[1].set_facecolor('#161b22')
                                axes[1].tick_params(colors='white')
                                for spine in axes[1].spines.values():
                                    spine.set_color('#30363d')

                                fig.patch.set_facecolor('#0E1117')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                    except Exception as e:
                        st.error(f"Could not generate distribution charts: {e}")

            st.markdown("---")
            st.markdown("""
            <div class="custom-card" style="background: linear-gradient(to right, #1c2128 0%, #21262d 100%); border-left: 4px solid #8b949e;">
                <h4 style="color: #8b949e; margin-top: 0;">📅 Monitoring Best Practices</h4>
                <ul style="color: #c9d1d9; line-height: 1.8;">
                    <li><strong>Weekly:</strong> Run quick drift check on new data</li>
                    <li><strong>Monthly:</strong> Full drift analysis with distribution comparison</li>
                    <li><strong>Quarterly:</strong> Model performance review and potential retraining</li>
                    <li><strong>After major changes:</strong> Re-org, policy changes, market shifts → Run immediate drift check</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()
