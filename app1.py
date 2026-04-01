# ====================================================================
# All Necessary Imports
# ====================================================================
import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from time import sleep
from scipy.sparse import issparse
from scipy.special import expit, logit
from scipy.optimize import milp, LinearConstraint

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel

# --- Imports for Evaluation 2 (Intelligent Interface: Groq) ---
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
    
    .result-leave { color: #EEB76B; text-shadow: 0 0 20px rgba(238, 183, 107, 0.3); }
    .result-stay { color: #17B794; text-shadow: 0 0 20px rgba(23, 183, 148, 0.3); }
    .card-percentage { font-size: 2rem; font-weight: 700; margin-top: 8px; }
    .percentage-stay { color: #17B794; }
    .percentage-leave { color: #EEB76B; }
    .card-indicator { width: 8px; height: 8px; border-radius: 50%; margin-top: 12px; }
    .indicator-stay { background-color: #17B794; box-shadow: 0 0 10px rgba(23, 183, 148, 0.5); }
    .indicator-leave { background-color: #EEB76B; box-shadow: 0 0 10px rgba(238, 183, 107, 0.5); }
    .card-section-first { background: linear-gradient(135deg, #1c2128 0%, #21262d 100%); }
    .highlight-leave .card-section-first { background: linear-gradient(135deg, #2d2515 0%, #1c2128 100%); }
    .highlight-stay .card-section-first { background: linear-gradient(135deg, #0d2818 0%, #1c2128 100%); }
    
    @media (max-width: 768px) {
        .prediction-card { flex-direction: column; }
        .card-section:not(:last-child)::after {
            right: 20%; top: auto; bottom: 0; height: 1px; width: 60%;
            background: linear-gradient(to right, transparent, #30363d, transparent);
        }
        .card-result { font-size: 1.8rem; }
        .card-percentage { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================
def format_money(amount):
    """Formats money based on user-selected currency."""
    mult = st.session_state.get('currency_mult', 1.0)
    sym = st.session_state.get('currency_symbol', '₹')
    val = amount * mult
    if val >= 10000000: return f"{sym}{val/10000000:.2f} Cr"
    elif val >= 100000: return f"{sym}{val/100000:.2f} L"
    else: return f"{sym}{val:,.0f}"

def get_salary_map():
    """Returns base INR salary map which gets multiplied by currency multiplier later if needed, 
    but for simplicity we map to fixed INR values and format_money handles conversion."""
    return {'low': 400000, 'medium': 600000, 'high': 900000}

def calibrate_probability(prob, temperature=0.55):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    scaled_logit = logit(prob) * temperature
    return float(expit(scaled_logit))

def calibrate_probability_array(probs, temperature=0.55):
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    scaled_logit = logit(probs) * temperature
    return expit(scaled_logit)

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
    fig = px.box(data_frame=the_df, x=column, title=f'{column.title().replace("_", " ")} Distribution', template="plotly_dark", labels={column: column.title().replace("_", " ")}, height=500, color_discrete_sequence=['#17B794'])
    custome_layout(fig, showlegend=False)
    return fig

def bar_plot(the_df, column, orientation="v"):
    dep = the_df[column].value_counts().nlargest(10)
    fig = px.bar(data_frame=dep, x=dep.index, y=dep.values, orientation=orientation, color=dep.index.astype(str), title=f'Distribution Via {column.title().replace("_", " ")}', color_discrete_sequence=["#17B794"], labels={"x": column.title().replace("_", " "), "y": "Count"}, template="plotly_dark", text_auto=True, height=500)
    custome_layout(fig)
    return fig

def pie_chart(the_df, column):
    counts = the_df[column].value_counts()
    fig = px.pie(data_frame=counts, names=counts.index, values=counts.values, title=f'Popularity of {column.title().replace("_", " ")}', color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"], template="plotly_dark", height=500)
    custome_layout(fig, showlegend=True)
    fig.update_traces(textfont={"size": 14, "color": "#fff"}, pull=[0, 0.1])
    return fig

def create_heat_map(the_df):
    numeric_df = the_df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()
    fig = px.imshow(correlation, template="plotly_dark", text_auto="0.2f", aspect=1, color_continuous_scale="greens", title="Correlation Heatmap", height=500)
    custome_layout(fig)
    return fig

def create_vizualization(the_df):
    with st.expander("📊 Data Visualizations (Click to Expand)", expanded=False):
        num_cols = list(the_df.select_dtypes(include=np.number).columns)
        obj_cols = list(the_df.select_dtypes(include=['object', 'category']).columns)
        
        tabs = st.tabs(["Numeric Distributions", "Categorical Counts", "Ratios"])
        with tabs[0]:
            for col in num_cols:
                if the_df[col].nunique() > 5: st.plotly_chart(box_plot(the_df, col), use_container_width=True)
        with tabs[1]:
            for col in obj_cols:
                st.plotly_chart(bar_plot(the_df, col), use_container_width=True)
        with tabs[2]:
            for col in the_df.columns:
                if 1 < the_df[col].nunique() <= 4: st.plotly_chart(pie_chart(the_df, col), use_container_width=True)
        st.plotly_chart(create_heat_map(the_df), use_container_width=True)

# ====================================================================
# Logic Engine & AI Functions
# ====================================================================
def analyze_why_people_leave(df):
    st.markdown("### 🔍 Root Cause Analysis (Causal AI)")
    required_cols = ['salary', 'satisfaction_level', 'average_montly_hours', 'number_project']
    if all(col in df.columns for col in required_cols):
        df_causal = df.copy()
        salary_map = {'low': 1, 'medium': 2, 'high': 3}
        df_causal['salary_num'] = df_causal['salary'].map(salary_map)
        causal_graph = """digraph { salary_num -> satisfaction_level; satisfaction_level -> left; average_montly_hours -> left; number_project -> average_montly_hours; }"""
        df_model = df_causal[['salary_num', 'satisfaction_level', 'average_montly_hours', 'number_project', 'left']]
        
        st.graphviz_chart(causal_graph)
        effects = {}
        try:
            for treat in ['salary_num', 'satisfaction_level', 'average_montly_hours']:
                model = CausalModel(data=df_model, treatment=treat, outcome='left', graph=causal_graph.replace('\n', ' '))
                est = model.estimate_effect(model.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
                effects[treat.replace('_num', '').title()] = abs(est.value) * (10 if 'hours' in treat else 1)
            
            sorted_effects = sorted(effects.items(), key=lambda item: item[1], reverse=True)
            c1, c2, c3 = st.columns(3)
            colors = ["#FF4B4B", "#FFA500", "#FFD700"]
            for idx, (col, val) in enumerate(sorted_effects):
                with [c1, c2, c3][idx]:
                    st.markdown(f"<div style='background-color: {colors[idx]}20; border: 1px solid {colors[idx]}; border-radius: 12px; padding: 20px; text-align: center;'><h2 style='color: {colors[idx]}; margin: 0;'>#{idx+1} {col}</h2></div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Causal analysis could not run: {e}")
    else:
        st.info("Advanced Causal Graph requires default dataset columns. Using SHAP analysis instead.")

def get_shap_explanations(_pipeline, _df):
    try:
        model = _pipeline.named_steps['classifier']
        preprocessor = _pipeline.named_steps['preprocessor']
        X = _df.drop('left', axis=1).drop_duplicates()
        X_processed = preprocessor.transform(X)
        if issparse(X_processed): X_processed = X_processed.toarray()
        
        clean_names = [name.split('__')[-1].replace('_', ' ') for name in preprocessor.get_feature_names_out()]
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
    if 'satisfaction_level' in employee_data.index and employee_data['satisfaction_level'] <= 0.45: strategies.append("🗣️ Conduct 1-on-1 meeting to address dissatisfaction.")
    if 'number_project' in employee_data.index:
        if employee_data['number_project'] <= 2: strategies.append("📈 Discuss career aspirations and assign challenging work.")
        if employee_data['number_project'] >= 6: strategies.append("⚠️ Assess workload immediately to prevent burnout.")
    if 'time_spend_company' in employee_data.index and 'promotion_last_5years' in employee_data.index:
        if employee_data['time_spend_company'] >= 4 and employee_data['promotion_last_5years'] == 0: strategies.append("📊 Develop clear career path and promotion timeline.")
    if not strategies: strategies.append("✅ Employee seems stable. Monitor standard engagement metrics.")
    return strategies

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Initialize Session States
    if 'currency_symbol' not in st.session_state: st.session_state['currency_symbol'] = '₹'
    if 'currency_mult' not in st.session_state: st.session_state['currency_mult'] = 1.0

    # ====================================================================
    # CROSS-PAGE NAVIGATION HANDLER
    # ====================================================================
    menu_options = ['⚙️ Global Setup', '📊 Executive Dashboard', '🔮 Predict & Prevent', '💰 Business Case', '🧠 AI Tools']
    
    if 'nav_to' in st.session_state:
        page = st.session_state.pop('nav_to')
        default_idx = menu_options.index(page) if page in menu_options else 1
    else:
        default_idx = 1 # FIXED: Default to Dashboard instead of Setup

    # ====================================================================
    # DATA LOADING & TRAINING
    # ====================================================================
    if 'is_global' in st.session_state and st.session_state['is_global']:
        pipeline = st.session_state['global_pipeline']
        df = st.session_state['global_df']
        X_train_ref = st.session_state['global_X_train']
        X_test_cur = st.session_state['global_X_test']
        y_train = st.session_state.get('global_y_train', pd.Series([0]))
        y_test = st.session_state.get('global_y_test', pd.Series([0]))
        preprocessor = pipeline.named_steps['preprocessor']
        cat_feat = st.session_state.get('global_cat_feat', [])
        num_feat = st.session_state.get('global_num_feat', [])
    else:
        @st.cache_data
        def load_data_and_train_model():
            if not os.path.exists('HR_comma_sep.csv'):
                st.error("❌ Default dataset 'HR_comma_sep.csv' not found. Please upload your data in ⚙️ Global Setup.")
                st.stop()
            
            df = pd.read_csv('HR_comma_sep.csv')
            df_train = df.drop_duplicates().reset_index(drop=True)
            X = df_train.drop('left', axis=1); y = df_train['left']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
            
            best_params = {'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 12, 'max_depth': 4, 'min_child_samples': 40, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'scale_pos_weight': 1.5}
            
            final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgb.LGBMClassifier(**best_params))])
            final_pipeline.fit(X_train, y_train)
            
            return final_pipeline, df, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model()
        st.session_state['cat_feat'] = cat_feat
        st.session_state['num_feat'] = num_feat

    # ====================================================================
    # SIDEBAR
    # ====================================================================
    with st.sidebar:
        st.markdown("""
        <div style='padding: 20px; text-align: center;'>
            <h1 style='font-size: 1.8rem; color: #17B794; margin-bottom: 0;'>RetainAI</h1>
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px; letter-spacing: 1px;'>ENTERPRISE WORKFORCE INTELLIGENCE</p>
        </div>
        <hr style='border-color: #30363d; margin: 20px 0;'>
        """, unsafe_allow_html=True)
        
        page = option_menu(
            menu_title=None, options=menu_options,
            icons=['gear', 'speedometer2', "cpu", 'cash-stack', 'robot'], 
            menu_icon="cast", default_index=default_idx, 
            styles={"container": {"padding": "0!important", "background-color": 'transparent'}, "icon": {"color": "#17B794", "font-size": "18px"}, "nav-link": {"color": "#c9d1d9", "font-size": "15px", "text-align": "left", "margin": "0px", "margin-bottom": "8px"}, "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"}}
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding:20px; border-top:1px solid #2d333b;'><div style='font-size:0.85rem; color:#8b949e;'>Built by</div><div style='font-size:1.4rem; font-weight:600; color:#00E5A8; margin-bottom:10px;'>Nisarg Rathod</div></div>", unsafe_allow_html=True)

    # ====================================================================
    # PAGE 1: GLOBAL SETUP
    # ====================================================================
    if page == "⚙️ Global Setup":
        st.header("⚙️ Global Setup & Configuration")
        st.markdown("<p style='color: #9ca3af;'>Turn this AI into your company's dedicated assistant.</p>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🌍 Localization")
            currency_opt = st.selectbox("Select Currency", ["₹ INR (Rupee)", "$ USD (Dollar)", "€ EUR (Euro)", "£ GBP (Pound)"])
            curr_map = {"₹ INR (Rupee)": ("₹", 1.0), "$ USD (Dollar)": ("$", 0.012), "€ EUR (Euro)": ("€", 0.011), "£ GBP (Pound)": ("£", 0.0095)}
            st.session_state['currency_symbol'], st.session_state['currency_mult'] = curr_map[currency_opt]
            st.success(f"Currency set to {currency_opt}")
            
        with c2:
            st.markdown("### 📂 Data Upload")
            uploaded_file = st.file_uploader("Upload HR Dataset (CSV)", type=["csv"])
            
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.dataframe(new_df.head())
                target_col = st.selectbox("Which column indicates 'Left'?", new_df.columns)
                unique_vals = new_df[target_col].unique()
                left_value = st.selectbox(f"Which value means 'Left'?", unique_vals)
                
                if st.button("🚀 Train Custom AI Model", type="primary"):
                    with st.spinner("🤖 Training on your data..."):
                        y = new_df[target_col].apply(lambda x: 1 if x == left_value else 0)
                        X = new_df.drop(target_col, axis=1)
                        valid_idx = X.dropna().index; X_clean = X.loc[valid_idx]; y_clean = y.loc[valid_idx]
                        
                        categorical_auto = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_auto = X_clean.select_dtypes(include=np.number).columns.tolist()
                        
                        preprocessor_global = ColumnTransformer(transformers=[
                            ('num', 'passthrough', numerical_auto), 
                            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_auto)])
                        
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        spw = min((y_train_g == 0).sum() / (y_train_g == 1).sum(), 2.0) if (y_train_g == 1).sum() > 0 else 1.0
                        
                        global_pipeline = Pipeline(steps=[('preprocessor', preprocessor_global), ('classifier', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=12, max_depth=4, random_state=42, verbose=-1, scale_pos_weight=spw))])
                        global_pipeline.fit(X_train_g, y_train_g)
                        acc = accuracy_score(y_test_g, global_pipeline.predict(X_test_g))
                        
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g
                        st.session_state['global_cat_feat'] = categorical_auto; st.session_state['global_num_feat'] = numerical_auto
                        st.session_state['is_global'] = True
                        st.success(f"🎉 Training Complete! Accuracy: {acc:.1%}.")
                        st.session_state['nav_to'] = '📊 Executive Dashboard'
                        st.rerun()
            except Exception as e: st.error(f"Error: {e}")
            
        if st.button("🔄 Reset to Default Demo Data"):
            if 'is_global' in st.session_state: del st.session_state['is_global']
            st.rerun()

    # ====================================================================
    # PAGE 2: EXECUTIVE DASHBOARD
    # ====================================================================
    if page == "📊 Executive Dashboard":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Executive Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af;'>Real-time pulse of your organization.</p>", unsafe_allow_html=True)
        
        # --- ALERT SYSTEM ---
        alerts = []
        attrition_rate = (df['left'].sum() / len(df)) * 100
        if attrition_rate > 20: alerts.append(("🚨 CRITICAL", f"Attrition rate is {attrition_rate:.1f}% (Threshold: 20%)"))
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.5: alerts.append(("⚠️ WARNING", "Average satisfaction below 50%"))
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: alerts.append(("⚠️ WARNING", "High average working hours (Burnout risk)"))
        
        for level, msg in alerts:
            if "CRITICAL" in level: st.error(msg)
            else: st.warning(msg)
        
        # --- METRICS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Workforce", f"{len(df):,}")
        c2.metric("Overall Attrition", f"{attrition_rate:.1f}%", delta="Current Period")
        c3.metric("Avg Satisfaction", f"{df['satisfaction_level'].mean():.2f}" if 'satisfaction_level' in df.columns else "N/A")
        
        # --- QUICK ACTIONS ---
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1: if st.button("🎯 Predict Employee", use_container_width=True): st.session_state['nav_to'] = '🔮 Predict & Prevent'; st.rerun()
        with c2: if st.button("💰 Budget Planner", use_container_width=True): st.session_state['nav_to'] = '💰 Business Case'; st.rerun()
        with c3: if st.button("📧 Draft Email", use_container_width=True): st.session_state['nav_to'] = '🧠 AI Tools'; st.rerun()
        with c4: if st.button("🚀 6-Month Plan", use_container_width=True): st.session_state['nav_to'] = '💰 Business Case'; st.rerun()
        
        # --- AT RISK EMPLOYEES ---
        st.markdown("---")
        st.markdown("### 🔴 Top 10 At-Risk Employees (Current)")
        st.caption("Predicted by AI as most likely to leave. Click 'Predict & Prevent' to see how to save them.")
        
        current_df = df[df['left'] == 0].copy()
        if len(current_df) > 0:
            features = [c for c in df.columns if c != 'left']
            risks = pipeline.predict_proba(current_df[features])[:, 1]
            current_df['Risk_Score'] = calibrate_probability_array(risks)
            top_risk = current_df.nlargest(10, 'Risk_Score')
            
            for idx, row in top_risk.iterrows():
                risk_pct = row['Risk_Score'] * 100
                dept = row.get('Department', 'N/A')
                salary = row.get('salary', 'N/A')
                color = "#FF4B4B" if risk_pct > 70 else "#EEB76B" if risk_pct > 50 else "#17B794"
                st.markdown(f"<div style='background:{color}15; border-left:4px solid {color}; padding:10px 15px; margin:5px 0; border-radius:0 8px 8px 0; display:flex; justify-content:space-between;'>"
                           f"<span><strong>{dept}</strong> | {salary.title()} Salary</span>"
                           f"<span style='color:{color}; font-weight:700;'>Risk: {risk_pct:.1f}%</span></div>", unsafe_allow_html=True)
        else:
            st.info("No current employees in dataset.")
            
        # --- VISUALIZATIONS ---
        create_vizualization(df)

    # ====================================================================
    # PAGE 3: PREDICT & PREVENT
    # ====================================================================
    if page == "🔮 Predict & Prevent":
        tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📦 Batch Upload", "🧠 Why They Leave"])
        
        with tab1:
            st.markdown("<h1 style='margin-bottom: 5px;'>Individual Attrition Predictor</h1>", unsafe_allow_html=True)
            feature_columns = [c for c in df.columns if c != 'left']
            is_default_data = 'satisfaction_level' in feature_columns 

            with st.form("Predict_value_form"):
                input_data = {}
                if is_default_data:
                    satisfaction_map = {'Very Dissatisfied': 0.1, 'Dissatisfied': 0.3, 'Neutral': 0.5, 'Satisfied': 0.7, 'Very Satisfied': 0.9}
                    evaluation_map = {'Needs Improvement': 0.4, 'Meets Expectations': 0.7, 'Exceeds Expectations': 0.9}
                    c1, c2 = st.columns(2)
                    with c1:
                        input_data['satisfaction_level'] = satisfaction_map[st.select_slider('Satisfaction Level', options=satisfaction_map.keys())]
                        input_data['last_evaluation'] = evaluation_map[st.select_slider('Last Evaluation', options=evaluation_map.keys())]
                        input_data['number_project'] = st.slider('Number of Projects', 2, 7, 4)
                        input_data['average_montly_hours'] = st.slider('Avg. Monthly Hours', 90, 310, 200)
                        input_data['time_spend_company'] = st.slider('Years at Company', 2, 10, 3)
                    with c2:
                        input_data['Work_accident'] = 1 if st.selectbox('Work Accident', ('No', 'Yes')) == 'Yes' else 0
                        input_data['promotion_last_5years'] = 1 if st.selectbox('Promotion in Last 5 Years', ('No', 'Yes')) == 'Yes' else 0
                        input_data['Department'] = st.selectbox('Department', df['Department'].unique())
                        input_data['salary'] = st.selectbox('Salary', df['salary'].unique())
                else:
                    cols = st.columns(2)
                    for i, col in enumerate(feature_columns):
                        with cols[i%2]:
                            if df[col].dtype == 'object': input_data[col] = st.selectbox(col.title(), df[col].unique())
                            else: input_data[col] = st.slider(col.title(), float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                predict_button = st.form_submit_button('🔮 Analyze Employee', type='primary')

            if predict_button:
                input_df = pd.DataFrame([input_data])[feature_columns]
                raw_probas = pipeline.predict_proba(input_df)[0]
                calibrated_leave = 1 - calibrate_probability(raw_probas[0])
                prediction = 1 if calibrated_leave >= 0.5 else 0
                
                stay_percent = int(round((1 - calibrated_leave) * 100))
                leave_percent = int(round(calibrated_leave * 100))
                
                result_text = "LEAVE" if prediction == 1 else "STAY"
                result_class = "result-leave" if prediction == 1 else "result-stay"
                highlight_class = "highlight-leave" if prediction == 1 else "highlight-stay"
                indicator_class = "indicator-leave" if prediction == 1 else "indicator-stay"
                
                st.markdown(f"""<div class="prediction-card {highlight_class}">
                    <div class="card-section card-section-first"><div class="card-label">Prediction</div><div class="card-result {result_class}">{result_text}</div></div>
                    <div class="card-section"><div class="card-label">Stay Prob.</div><div class="card-percentage percentage-stay">{stay_percent}%</div><div class="card-indicator indicator-stay"></div></div>
                    <div class="card-section"><div class="card-label">Leave Prob.</div><div class="card-percentage percentage-leave">{leave_percent}%</div><div class="card-indicator {indicator_class}"></div></div>
                </div>""", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown("### 💡 Immediate Actions")
                    for rec in get_retention_strategies(input_df): st.info(rec)
                    
                    st.markdown("### 🔮 AI Retention Strategies (What-If)")
                    if st.button("💡 Show Me How to Keep Them", type="primary"):
                        with st.spinner("Simulating strategies..."):
                            try:
                                continuous_features = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                                if len(continuous_features) < 2: st.warning("Not enough features for simulation.")
                                else:
                                    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left')
                                    m = dice_ml.Model(model=pipeline, backend='sklearn')
                                    exp = Dice(d, m, method='random')
                                    cf = exp.generate_counterfactuals(input_df, total_CFs=3, desired_class="opposite")
                                    cf_df = cf.cf_examples_list[0].final_cfs_df
                                    original = input_df.iloc[0]
                                    
                                    cols_s = st.columns(3)
                                    for i in range(len(cf_df)):
                                        changes = []
                                        for col in original.index:
                                            orig_val, new_val = original[col], cf_df.iloc[i][col]
                                            if isinstance(orig_val, float) and abs(orig_val - new_val) > 0.05:
                                                changes.append(f"• Adjust <strong>{col.title()}</strong> from {orig_val:.2f} to {new_val:.2f}")
                                            elif orig_val != new_val:
                                                changes.append(f"• Change <strong>{col.title()}</strong> to {new_val}")
                                        if not changes: changes.append("• Minor supervision required")
                                        html_str = "".join([f"<div class='action-item'>{c}</div>" for c in changes])
                                        with cols_s[i]:
                                            st.markdown(f"<div class='custom-card' style='border-color: #17B794;'><h4 style='color: #17B794; margin-top:0;'>Strategy {i+1}</h4>{html_str}</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Could not generate strategies: {e}")
                                
                    st.markdown("---")
                    if st.button("📧 Draft Retention Email for This Person", use_container_width=True):
                        st.session_state['nav_to'] = '🧠 AI Tools'; st.rerun()

        with tab2:
            st.markdown("### 📦 Batch Prediction")
            st.write("Upload a CSV of multiple employees to get predictions instantly.")
            batch_file = st.file_uploader("Upload Employee CSV", type=["csv"], key="batch")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                req_cols = [c for c in feature_columns if c in batch_df.columns]
                if len(req_cols) == len(feature_columns):
                    probs = pipeline.predict_proba(batch_df[feature_columns])[:, 1]
                    batch_df['Risk Score'] = calibrate_probability_array(probs)
                    batch_df['Prediction'] = batch_df['Risk Score'].apply(lambda x: "LEAVE" if x > 0.5 else "STAY")
                    batch_df = batch_df.sort_values('Risk Score', ascending=False)
                    st.dataframe(batch_df, use_container_width=True)
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results", csv, "batch_predictions.csv", "text/csv")
                else:
                    st.error(f"Missing columns. Required: {feature_columns}")

        with tab3:
            analyze_why_people_leave(df)
            st.markdown("---")
            st.markdown("### 🔧 Technical Deep Dive (SHAP Insights)")
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
            if shap_values is not None:
                if isinstance(shap_values, list): vals = np.abs(shap_values[1]).mean(0)
                else: vals = np.abs(shap_values).mean(0)
                
                feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance']).sort_values('Importance', ascending=False)
                top_3 = feature_importance.head(3)
                c1, c2, c3 = st.columns(3)
                for idx, (_, row) in enumerate(top_3.iterrows()):
                    with [c1, c2, c3][idx]:
                        st.markdown(f"<div class='custom-card' style='text-align:center;'><h3 style='color: #17B794; margin-top:0;'>{row['Feature'].title()}</h3><p style='color: #c9d1d9;'>Top Driver #{idx+1}</p></div>", unsafe_allow_html=True)
                
                with st.expander("View SHAP Plots"):
                    fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type="bar", show=False); st.pyplot(fig1); plt.close(fig1)
                    fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False); st.pyplot(fig2); plt.close(fig2)

    # ====================================================================
    # PAGE 4: BUSINESS CASE
    # ====================================================================
    if page == "💰 Business Case":
        tab_budget, tab_roadmap = st.tabs(["💰 Budget Planner", "🚀 Strategic Roadmap"])
        
        with tab_budget:
            st.markdown("<h1 style='margin-bottom: 5px;'>Budget Planner</h1>", unsafe_allow_html=True)
            
            # PHASE 1: BURN RATE
            st.markdown("""<div class="custom-card" style="border-left: 5px solid #FF4B4B;"><h3 style="color: #FF4B4B; margin-top: 0;">💸 Phase 1: The Burn Rate</h3><p style='color: #e6edf3;'>Translating attrition into hard numbers.</p></div>""", unsafe_allow_html=True)
            
            if 'Department' in df.columns and 'salary' in df.columns:
                salary_map = get_salary_map()
                df_cost = df.copy()
                df_cost['annual_salary'] = df_cost['salary'].map(salary_map)
                df_left = df_cost[df_cost['left'] == 1].copy()
                
                if 'time_spend_company' in df_left.columns:
                    df_left['total_cost'] = df_left['annual_salary'] * 0.5 * (1 + (df_left['time_spend_company'] * 0.10)).clip(upper=2.0)
                else:
                    df_left['total_cost'] = df_left['annual_salary'] * 0.75
                    
                dept_costs = df_left.groupby('Department').agg(employees_left=('left', 'count'), total_cost=('total_cost', 'sum')).reset_index().sort_values('total_cost', ascending=False)
                grand_total = dept_costs['total_cost'].sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Money Lost", format_money(grand_total), delta=f"{dept_costs['employees_left'].sum()} Exits", delta_color="inverse")
                c2.metric("Avg Cost Per Exit", format_money(grand_total/len(df_left)))
                c3.metric("Most Expensive Dept", dept_costs.iloc[0]['Department'], delta=format_money(dept_costs.iloc[0]['total_cost'])+" Lost", delta_color="inverse")
                
                fig_cost = px.bar(dept_costs, x='Department', y='total_cost', title="Cost by Department", template="plotly_dark", color='total_cost', color_continuous_scale=['#17B794', '#FF4B4B'])
                fig_cost.update_layout(yaxis_title="Loss", showlegend=False)
                custome_layout(fig_cost)
                st.plotly_chart(fig_cost, use_container_width=True)

            # PHASE 2: OPTIMIZER
            st.markdown("""<div class="custom-card" style="border-left: 5px solid #17B794;"><h3 style="color: #17B794; margin-top: 0;">🛡️ Phase 2: ROI Optimizer</h3><p style='color: #e6edf3;'>Who should we save with limited budget?</p></div>""", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1: budget = st.number_input("Enter Retention Budget", min_value=100000, max_value=100000000, value=1000000, step=50000, format="%d")
            with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Run Optimizer", type="primary")
            
            if optimize_btn:
                with st.spinner("🧮 Running Optimization..."):
                    X = df.drop('left', axis=1)
                    probas = calibrate_probability_array(pipeline.predict_proba(X)[:, 1])
                    opt_df = df.copy(); opt_df['risk'] = probas
                    high_risk_df = opt_df[opt_df['risk'] > 0.5].copy()
                    
                    if len(high_risk_df) > 0:
                        high_risk_df['salary_val'] = high_risk_df['salary'].map(salary_map) if 'salary' in high_risk_df.columns else 500000
                        high_risk_df['cost_to_retain'] = high_risk_df['salary_val'] * 0.10
                        high_risk_df['expected_loss'] = high_risk_df['risk'] * (high_risk_df['salary_val'] * 0.50)
                        high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['cost_to_retain']
                        candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
                        
                        if len(candidates) > 0:
                            n = len(candidates)
                            res = milp(c=-candidates['net_savings'].values, constraints=LinearConstraint(np.array([candidates['cost_to_retain'].values]), lb=-np.inf, ub=budget), integrality=np.ones(n))
                            if res.success:
                                selected = candidates.iloc[np.where(res.x == 1)[0]]
                                st.success("✅ Optimization Complete!")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Investment", format_money(selected['cost_to_retain'].sum()))
                                m2.metric("Projected Savings", format_money(selected['net_savings'].sum()))
                                m3.metric("Lives Saved", f"{len(selected)} People")
                                st.dataframe(selected[['Department', 'salary', 'risk', 'cost_to_retain', 'net_savings']])
                            else: st.error("❌ Budget too low to save anyone.")
                        else: st.warning("No high-ROI candidates found.")
                    else: st.success("🎉 Workforce is highly stable.")

        with tab_roadmap:
            st.header("🚀 Strategic Roadmap & Projections")
            issues = []
            if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.6: issues.append("Low Employee Satisfaction")
            if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: issues.append("Employee Burnout")
            if not issues: issues.append("Standard Workforce Stabilization")
            
            if st.button("✍️ Draft 6-Month HR Action Plan", type="primary"):
                with st.spinner("Drafting strategy..."):
                    try:
                        api_key = st.secrets.get("GROQ_API_KEY")
                        if api_key:
                            llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                            chain = PromptTemplate.from_template("You are an HR Strategist. Issues: {issues}. Create a 6-month roadmap with phases, actions, and metrics. Plain English.") | llm | StrOutputParser()
                            st.markdown(f"<div class='llm-response'>{chain.invoke({'issues': ', '.join(issues)})}</div>", unsafe_allow_html=True)
                        else: st.warning("API Key missing.")
                    except Exception as e: st.error(f"Error: {e}")
            
            st.markdown("---")
            st.markdown("### 📈 12-Month Projection")
            c1, c2 = st.columns(2)
            with c1: eff = st.slider("Intervention Efficacy (%)", 10, 50, 20, 5)
            with c2: nat = st.slider("Natural Attrition (%)", 0.5, 2.0, 1.0, 0.1)
            
            if st.button("📈 Show Projection"):
                months = list(range(1, 13)); curr = len(df)
                total_risk = calibrate_probability_array(pipeline.predict_proba(df.drop('left', axis=1))[:, 1]).sum()
                m_no = total_risk / 12.0; m_yes = m_no * (1 - (eff/100.0))
                f_bau, f_int, t_bau, t_int = [], [], float(curr), float(curr)
                
                for m in months:
                    t_bau -= m_no + t_bau*(nat/100); t_int -= m_yes + t_int*(nat/100)
                    f_bau.append(t_bau); f_int.append(t_int)
                    
                plot_df = pd.DataFrame({'Month': months, 'Do Nothing': f_bau, 'Follow Plan': f_int}).melt(id_vars='Month', var_name='Scenario', value_name='Headcount')
                fig = px.line(plot_df, x='Month', y='Headcount', color='Scenario', template="plotly_dark", markers=True, color_discrete_map={'Do Nothing': "#EEB76B", 'Follow Plan': "#17B794"})
                st.plotly_chart(fig, use_container_width=True)
                
                saved = int(f_int[-1] - f_bau[-1])
                st.metric("Employees Saved", f"{saved} People")

    # ====================================================================
    # PAGE 5: AI TOOLS
    # ====================================================================
    if page == "🧠 AI Tools":
        tab1, tab2, tab3, tab4 = st.tabs(["📧 Communication", "⚙️ Benchmarking", "🔬 Dept. Strategy", "🛡️ AI Defense"])
        
        with tab1:
            st.header("✍️ AI Communication Assistant")
            with st.form("llm_form"):
                c1, c2 = st.columns(2)
                with c1:
                    emp_name = st.text_input("Employee Name", "Rahul Sharma")
                    emp_dept = st.selectbox("Department", df['Department'].unique()) if 'Department' in df.columns else st.text_input("Department", "Sales")
                with c2:
                    situation = st.selectbox("Situation", ["Overworked", "Seeking Salary", "Low Morale", "No Growth"])
                    solution = st.selectbox("Solution", ["Flexible Hours", "Salary Review", "Role Change", "Wellness Session"])
                if st.form_submit_button("🚀 Generate Email"):
                    with st.spinner("Drafting..."):
                        try:
                            api_key = st.secrets.get("GROQ_API_KEY")
                            if api_key:
                                llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
                                chain = PromptTemplate.from_template("Write a polite HR email to {emp_name} in {emp_dept}. Situation: {situation}. Solution: {solution}. Tone: Professional.") | llm | StrOutputParser()
                                st.markdown(f"<div class='llm-response'>{chain.invoke({'emp_name': emp_name, 'emp_dept': emp_dept, 'situation': situation, 'solution': solution})}</div>", unsafe_allow_html=True)
                            else: st.warning("API Key missing.")
                        except Exception as e: st.error(f"Error: {e}")

        with tab2:
            st.header("⚙️ Model Benchmarking")
            if st.button("Run Benchmark"):
                with st.spinner("Training models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf = Pipeline([('p', preprocessor), ('c', RandomForestClassifier(random_state=42))]); rf.fit(X_train_ref, y_train)
                    lr = Pipeline([('p', preprocessor), ('c', LogisticRegression(max_iter=1000))]); lr.fit(X_train_ref, y_train)
                    
                    metrics = pd.DataFrame({
                        'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'],
                        'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, rf.predict(X_test_cur)), accuracy_score(y_test, lr.predict(X_test_cur))],
                        'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, rf.predict_proba(X_test_cur)[:,1]), roc_auc_score(y_test, lr.predict_proba(X_test_cur)[:,1])]
                    })
                    st.dataframe(metrics.style.highlight_max(axis=0, color='#17B794'))
                    st.success("🏆 LightGBM selected for production.")

        with tab3:
            st.header("🔬 Departmental Strategy Deep Dive")
            if 'Department' not in df.columns: st.warning("Department column missing.")
            else:
                dept = st.selectbox("Select Dept", sorted(df['Department'].unique()))
                if st.button("Generate Strategy"):
                    shap_vals, X_proc_df = get_shap_explanations(pipeline, df)
                    if shap_vals is not None:
                        target_col = next((c for c in X_proc_df.columns if c.lower().replace('_',' ') == dept.lower().replace('_',' ')), None)
                        if target_col:
                            mask = X_proc_df[target_col] == 1
                            if mask.sum() > 0:
                                dept_shap = shap_vals[1][mask] if isinstance(shap_vals, list) else shap_vals[mask]
                                imp = pd.DataFrame({'Feature': X_proc_df.columns, 'Impact': np.abs(dept_shap).mean(0)}).sort_values('Impact', ascending=False)
                                imp = imp[~imp['Feature'].str.contains('Department')]
                                st.plotly_chart(px.bar(imp.head(5).iloc[::-1], x='Impact', y='Feature', orientation='h', template="plotly_dark", color_discrete_sequence=['#17B794']), use_container_width=True)
                                
                                for i, (_, row) in enumerate(imp.head(3).iterrows()):
                                    st.info(f"Driver #{i+1}: {row['Feature'].title()}")
                            else: st.warning("Not enough data.")
                        else: st.error(f"Could not map {dept}.")

        with tab4:
            st.header("🛡️ AI Disruption Defense & Fairness")
            st.markdown("""<div class="custom-card" style="border-left: 5px solid #9ca3ca;"><h4 style="color: #9ca3ca; margin-top: 0;">🧠 The 'Proof of Work' Defense</h4><p style='color: #c9d1d9;'>"AI replaces tasks, not jobs. Reskilling 50 people is cheaper than laying off 200."</p></div>""", unsafe_allow_html=True)
            
            # Bias Check
            st.markdown("### ⚖️ Fairness & Bias Check")
            if 'salary' in df.columns:
                current_df = df[df['left']==0].copy()
                risks = pipeline.predict_proba(current_df.drop('left', axis=1))[:, 1]
                current_df['risk'] = risks
                low_risk = current_df[current_df['salary']=='low']['risk'].mean()
                high_risk = current_df[current_df['salary']=='high']['risk'].mean()
                ratio = low_risk / high_risk if high_risk > 0 else 0
                if ratio > 1.5: st.warning(f"⚠️ Potential Bias: Low-salary employees are {ratio:.1f}x more likely to be flagged.")
                else: st.success("✅ No significant bias detected across salary bands.")
                
            # Reskill Calculator
            st.markdown("### 🧮 Reskill vs Layoff Calculator")
            c1, c2 = st.columns(2)
            num_emp = c1.number_input("At-Risk Employees", 5, 500, 50)
            avg_sal = c2.number_input("Avg Salary (Base INR)", 200000, 3000000, 600000, step=50000)
            
            layoff_cost = (avg_sal/12)*3*num_emp + (avg_sal*2)*(num_emp*0.1)
            reskill_cost = (avg_sal/12)*1.5*num_emp
            savings = layoff_cost - reskill_cost
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Layoff Cost", format_money(layoff_cost))
            c2.metric("Reskill Cost", format_money(reskill_cost))
            c3.metric("Savings", format_money(savings), delta="Reskill wins" if savings > 0 else "Layoff cheaper")

if __name__ == "__main__":
    main()
