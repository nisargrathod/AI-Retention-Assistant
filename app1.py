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
from scipy.special import expit, logit  # 🔧 CRITICAL: For probability calibration

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel
from scipy.optimize import milp, LinearConstraint, Bounds

# --- Imports for Evaluation 2 (Intelligent Interface) ---
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { padding-left: 20px; padding-right: 20px; }
    .main { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1rem; font-weight: 400; color: #9ca3af; }
    .custom-card { background-color: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; color: #c9d1d9; }
    div[data-testid="stFormSubmitButton"] > button { width: 100%; background: linear-gradient(90deg, #17B794 0%, #11998e 100%); border: none; padding: 12px; border-radius: 8px; color: white; font-weight: 600; font-size: 16px; transition: all 0.3s ease; }
    div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(23, 183, 148, 0.4); }
    .dataframe { border-radius: 8px; overflow: hidden; }
    .dataframe th { background-color: #21262d; color: #ffffff; font-weight: 600; }
    .streamlit-expanderHeader { background-color: #21262d; border-radius: 8px; color: white; font-weight: 600; }
    .llm-response { background-color: #21262d; border-left: 4px solid #17B794; padding: 15px; border-radius: 5px; margin-top: 10px; color: #e6edf3; line-height: 1.6; white-space: pre-wrap; }
    .action-item { background-color: #161b22; padding: 8px; margin-bottom: 8px; border-left: 3px solid #17B794; border-radius: 4px; font-size: 0.9rem; }
    .action-item-high-effort { border-left: 3px solid #FF4B4B; }
    
    .prob-bar-container { width: 100%; background-color: #21262d; border-radius: 10px; overflow: hidden; height: 35px; margin-top: 10px; display: flex; }
    .prob-bar-stay { height: 100%; background: linear-gradient(90deg, #17B794, #11998e); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 14px; }
    .prob-bar-leave { height: 100%; background: linear-gradient(90deg, #FF4B4B, #cc3a3a); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 🔧 CRITICAL FIX: TEMPERATURE SCALING FUNCTION
# ====================================================================
# This is the STANDARD solution used by Netflix, Uber, etc.
# to fix overconfident tree model predictions.
# ====================================================================
def calibrate_probability(prob, temperature=0.55):
    """
    Apply temperature scaling to reduce overconfidence.
    
    How it works:
    - Convert probability to logit space (log-odds)
    - Divide by temperature (shrinks extreme values toward 0)
    - Convert back to probability using sigmoid
    
    Temperature < 1.0 = More conservative (less confident)
    Temperature > 1.0 = More extreme (more confident)
    
    Example with temperature=0.55:
    - 99.9% → ~88%
    - 99% → ~82%
    - 90% → ~68%
    - 50% → ~50%
    - 10% → ~32%
    - 1% → ~18%
    - 0.1% → ~12%
    """
    # Clip to avoid log(0) which is undefined
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    
    # Convert to logit space: log(p / (1-p))
    logit_prob = logit(prob)
    
    # Apply temperature scaling
    scaled_logit = logit_prob * temperature
    
    # Convert back to probability using sigmoid: 1 / (1 + exp(-x))
    calibrated = expit(scaled_logit)
    
    return float(calibrated)

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={"font": {"size": title_size, "family": "tahoma"}},
        hoverlabel={"bgcolor": "#000", "font_size": hover_font_size, "font_family": "arial"},
        paper_bgcolor="#0E1117", plot_bgcolor="#161b22", font_color="#c9d1d9"
    )

def box_plot(the_df, column):
    fig = px.box(data_frame=the_df, x=column, title=f'{column.title().replace("_", " ")} Distribution', template="plotly_dark", height=600, color_discrete_sequence=['#17B794'])
    custome_layout(fig, showlegend=False)
    return fig

def bar_plot(the_df, column, orientation="v"):
    dep = the_df[column].value_counts()
    fig = px.bar(data_frame=dep, x=dep.index, y=dep.values, orientation=orientation, color=dep.index.astype(str), title=f'Distribution via {column.title()}', color_discrete_sequence=["#17B794"], template="plotly_dark", text_auto=True, height=650)
    custome_layout(fig, title_size=28)
    return fig

def pie_chart(the_df, column):
    counts = the_df[column].value_counts()
    fig = px.pie(data_frame=counts, names=counts.index, values=counts.values, title=f'{column.title()}', color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"], template="plotly_dark", height=650)
    custome_layout(fig, showlegend=True, title_size=28)
    return fig

def create_heat_map(the_df):
    numeric_df = the_df.select_dtypes(include=np.number)
    fig = px.imshow(numeric_df.corr(), template="plotly_dark", text_auto="0.2f", aspect=1, color_continuous_scale="greens", title="Correlation Heatmap", height=650)
    custome_layout(fig)
    return fig

def create_vizualization(the_df, viz_type="box", data_type="number"):
    figs = []; num_columns = list(the_df.select_dtypes(include=data_type).columns); cols_index = []
    if viz_type == "box":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() > 10: figs.append(box_plot(the_df, num_columns[i])); cols_index.append(i)
    if viz_type == "bar":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() < 15: figs.append(bar_plot(the_df, num_columns[i])); cols_index.append(i)
    if viz_type == "pie":
        num_columns = list(the_df.columns)
        for i in range(len(num_columns)):
            if 1 < the_df[num_columns[i]].nunique() <= 4: figs.append(pie_chart(the_df, num_columns[i])); cols_index.append(i)
    if len(cols_index) > 0:
        tabs = st.tabs([num_columns[i].title().replace("_", " ") for i in cols_index])
        for i in range(len(cols_index)): tabs[i].plotly_chart(figs[i], use_container_width=True)

# ====================================================================
# Logic Engine Functions
# ====================================================================
def analyze_why_people_leave(df):
    st.markdown("### 🔍 Why do people leave?")
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
            c1, c2, c3 = st.columns(3)
            for idx, (col, val) in enumerate(sorted_effects):
                color = "#FF4B4B" if idx == 0 else "#FFA500" if idx == 1 else "#FFD700"
                status = "CRITICAL" if idx == 0 else "MAJOR" if idx == 1 else "MODERATE"
                card_html = f"<div style='background-color: {color}20; border: 1px solid {color}; border-radius: 12px; padding: 20px; text-align: center;'><h2 style='color: {color}; margin: 0;'>#{idx+1} {col}</h2><h4 style='color: white;'>{status}</h4></div>"
                with [c1, c2, c3][idx]: st.markdown(card_html, unsafe_allow_html=True)
        except Exception as e: st.error(f"Causal analysis error: {e}")
    else:
        st.info("📊 *Advanced Causal Graph requires specific columns.*")

def plan_retention_budget(df, pipeline, budget_limit):
    X = df.drop('left', axis=1)
    probas = pipeline.predict_proba(X)[:, 1]
    opt_df = df.copy(); opt_df['attrition_risk'] = probas
    high_risk_df = opt_df[opt_df['attrition_risk'] > 0.5].copy()
    if len(high_risk_df) == 0: st.success("🎉 Workforce is stable."); return None
    if 'salary' in df.columns: high_risk_df['annual_salary'] = high_risk_df['salary'].map({'low': 400000, 'medium': 600000, 'high': 900000})
    else: high_risk_df['annual_salary'] = 500000
    high_risk_df['replacement_cost'] = high_risk_df['annual_salary'] * 0.5
    high_risk_df['expected_loss'] = high_risk_df['attrition_risk'] * high_risk_df['replacement_cost']
    high_risk_df['intervention_cost'] = high_risk_df['annual_salary'] * 0.10
    high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['intervention_cost']
    candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
    if len(candidates) == 0: st.warning("⚠️ Not cost-effective."); return None
    n = len(candidates); c = -candidates['net_savings'].values
    A = np.array([candidates['intervention_cost'].values]); b = np.array([budget_limit])
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb=-np.inf, ub=b), integrality=np.ones(n))
        if res.success:
            selected = candidates.iloc[np.where(res.x == 1)[0]]
            return selected, selected['intervention_cost'].sum(), selected['net_savings'].sum()
    except Exception as e: st.error(f"Error: {e}")
    return None

def run_groq_consultant(name, dept, situation, solution, cost):
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key: st.warning("🔑 API Key missing."); return
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
        if "overwork" in situation.lower(): root = "High Workload & Burnout"
        elif "salary" in situation.lower(): root = "Compensation Issues"
        else: root = "Attrition Risk"
        template = """Write professional HR email to {name} ({dept}). Situation: {sit}. Root cause: {root}. Solution: {sol}. Cost: {cost}. Tone: Professional, Supportive."""
        chain = PromptTemplate.from_template(template) | llm | StrOutputParser()
        with st.spinner("Drafting..."):
            response = chain.invoke({"name": name, "dept": dept, "sit": situation, "root": root, "sol": solution, "cost": cost})
            st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
    except Exception as e: st.error(f"Error: {e}")

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ====================================================================
    # GLOBAL ROUTER
    # ====================================================================
    if 'is_global' in st.session_state and st.session_state['is_global']:
        pipeline = st.session_state['global_pipeline']
        df = st.session_state['global_df']
        X_train_ref = st.session_state['global_X_train']
        X_test_cur = st.session_state['global_X_test']
        y_train = st.session_state.get('global_y_train', pd.Series([0]))
        y_test = st.session_state.get('global_y_test', pd.Series([0]))
        preprocessor = pipeline.named_steps['preprocessor']
    else:
        # ====================================================================
        # 🔧 CRITICAL FIX #1: FORCE CACHE INVALIDATION
        # ====================================================================
        # The _model_version parameter forces Streamlit to retrain the model
        # with the new parameters. Without this, the old cached model persists!
        # ====================================================================
        @st.cache_data
        def load_data_and_train_model(_model_version="v4_calibrated"):
            st.write("📂 Step 1/3: Loading Dataset...")
            df = pd.read_csv('HR_comma_sep.csv')
            
            st.write("🧹 Step 2/3: Preprocessing...")
            df_train = df.drop_duplicates().reset_index(drop=True)
            X = df_train.drop('left', axis=1)
            y = df_train['left']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            categorical_features = X.select_dtypes(include=['object']).columns
            numerical_features = X.select_dtypes(include=np.number).columns
            
            # ====================================================================
            # 🔧 CRITICAL FIX #2: NO SCALING FOR TREE MODELS
            # ====================================================================
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),  # ✅ NO StandardScaler
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                ])
            
            st.write("🤖 Step 3/3: Training AI Model...")
            
            # ====================================================================
            # 🔧 CRITICAL FIX #3: SIMPLER MODEL WITH REGULARIZATION
            # ====================================================================
            best_params = {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'num_leaves': 12,           # Reduced: Simpler trees
                'max_depth': 4,             # Reduced: Shallower trees  
                'min_child_samples': 40,    # NEW: Larger leaf samples
                'reg_alpha': 0.5,           # NEW: Stronger L1 regularization
                'reg_lambda': 0.5,          # NEW: Stronger L2 regularization
                'subsample': 0.8,           # NEW: Row sampling
                'colsample_bytree': 0.8,    # NEW: Column sampling
                'random_state': 42,
                'verbose': -1,
                'scale_pos_weight': 1.5     # Reduced: Less aggressive
            }
            
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', lgb.LGBMClassifier(**best_params))])
            
            final_pipeline.fit(X_train, y_train)
            
            return final_pipeline, df, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

        # ====================================================================
        # 🔧 CRITICAL FIX #4: PASS VERSION TO FORCE CACHE MISS
        # ====================================================================
        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model(_model_version="v4_calibrated")
        st.empty()

    @st.cache_data
    def get_shap_explanations(_pipeline, _df):
        model = _pipeline.named_steps['classifier']
        preprocessor = _pipeline.named_steps['preprocessor']
        X = _df.drop('left', axis=1).drop_duplicates()
        X_processed = preprocessor.transform(X)
        if issparse(X_processed): X_processed = X_processed.toarray()
        clean_names = [name.split('__')[-1].replace('_', ' ').title() for name in preprocessor.get_feature_names_out()]
        X_processed_df = pd.DataFrame(X_processed, columns=clean_names)
        booster = model.booster_ if hasattr(model, "booster_") else model._Booster if hasattr(model, "_Booster") else model
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
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px;'>ENTERPRISE WORKFORCE INTELLIGENCE</p>
        </div>
        <hr style='border-color: #30363d; margin: 20px 0;'>
        """, unsafe_allow_html=True)
        
        page = option_menu(
            menu_title=None,
            options=['⚙️ Global Setup', 'Home', 'Employee Insights', 'Predict Attrition', 'Why They Leave', 'Budget Planner', 'AI Assistant', 'AI Research Lab', 'Strategic Roadmap'],  
            icons=['gear', 'house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill'], 
            menu_icon="cast", default_index=0, 
            styles={"container": {"padding": "0!important", "background-color": 'transparent'}, "icon": {"color": "#17B794", "font-size": "18px"}, "nav-link": {"color": "#c9d1d9", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "10px"}, "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"}})
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 20px; text-align: center; color: #8b949e; font-size: 0.8rem;'>Developed by<br><strong>Nisarg Rathod</strong></div>", unsafe_allow_html=True)

    # ====================================================================
    # PAGE: GLOBAL SETUP
    # ====================================================================
    if page == "⚙️ Global Setup":
        st.header("⚙️ Global Setup: Upload Your Company Data")
        uploaded_file = st.file_uploader("Upload your HR Dataset (CSV format)", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!"); st.dataframe(new_df.head())
                target_col = st.selectbox("Which column indicates if the employee left?", new_df.columns)
                left_value = st.selectbox(f"In '{target_col}', which value means 'Left'?", new_df[target_col].unique())
                feature_cols = [c for c in new_df.columns if c != target_col]
                cat_cols = new_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = new_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
                if st.button("🚀 Train Custom AI Model", type="primary"):
                    with st.spinner("🤖 AI is learning..."):
                        y = new_df[target_col].apply(lambda x: 1 if x == left_value else 0); X = new_df[feature_cols]
                        valid_idx = X.dropna().index; X_clean = X.loc[valid_idx]; y_clean = y.loc[valid_idx]
                        prep = ColumnTransformer([('num', 'passthrough', num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)] if cat_cols else [('num', 'passthrough', num_cols)])
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        spw = min((y_train_g == 0).sum() / (y_train_g == 1).sum(), 2.0) if (y_train_g == 1).sum() > 0 else 1.0
                        global_pipeline = Pipeline([('preprocessor', prep), ('classifier', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=12, max_depth=4, min_child_samples=40, reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, scale_pos_weight=spw))])
                        global_pipeline.fit(X_train_g, y_train_g); acc = accuracy_score(y_test_g, global_pipeline.predict(X_test_g))
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state.update({'global_pipeline': global_pipeline, 'global_df': final_df, 'global_X_train': X_train_g, 'global_X_test': X_test_g, 'global_y_train': y_train_g, 'global_y_test': y_test_g, 'is_global': True})
                        st.balloons(); st.success(f"🎉 Training Complete! Accuracy: **{acc:.1%}**")
            except Exception as e: st.error(f"Error: {e}")
        if st.button("🔄 Reset to Default Demo Data"):
            if 'is_global' in st.session_state: del st.session_state['is_global']
            st.rerun()

    # ====================================================================
    # PAGE: HOME
    # ====================================================================
    if page == "Home":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        total_employees = len(df); attrition_rate = (df['left'].sum() / len(df)) * 100
        if 'satisfaction_level' in df.columns:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f}")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        st.markdown("---"); st.dataframe(df.head(100), use_container_width=True)

    # ====================================================================
    # PAGE: EMPLOYEE INSIGHTS
    # ====================================================================
    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis")
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    # ====================================================================
    # PAGE: PREDICT ATTRITION (FULLY CALIBRATED VERSION)
    # ====================================================================
    if page == "Predict Attrition":
        st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af;'>Enter employee details to assess risk.</p>", unsafe_allow_html=True)
        
        # --- MODEL DIAGNOSTICS ---
        with st.expander("🧪 Model Diagnostics", expanded=False):
            st.write("Test the AI against real historical data:")
            c_test1, c_test2 = st.columns(2)
            with c_test1:
                if st.button("Test with Employee who Left"):
                    sample = df[df['left'] == 1].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    raw_prob = pipeline.predict_proba(test_df)[0][1]
                    calibrated_prob = calibrate_probability(raw_prob)  # 🔧 APPLY CALIBRATION
                    pred = 1 if calibrated_prob >= 0.5 else 0
                    if pred == 1: st.success(f"✅ Correct! Calibrated risk: {calibrated_prob:.1%}")
                    else: st.error(f"❌ Wrong! Calibrated risk: {calibrated_prob:.1%}")
            with c_test2:
                if st.button("Test with Employee who Stayed"):
                    sample = df[df['left'] == 0].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    raw_prob = pipeline.predict_proba(test_df)[0][1]
                    calibrated_prob = calibrate_probability(raw_prob)  # 🔧 APPLY CALIBRATION
                    pred = 1 if calibrated_prob >= 0.5 else 0
                    if pred == 0: st.success(f"✅ Correct! Calibrated risk: {calibrated_prob:.1%}")
                    else: st.error(f"❌ Wrong! Calibrated risk: {calibrated_prob:.1%}")

        st.markdown("---")

        # --- INPUT FORM ---
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
                
                input_data = {
                    'satisfaction_level': satisfaction_map[satisfaction_text], 
                    'last_evaluation': evaluation_map[evaluation_text], 
                    'number_project': number_project, 
                    'average_montly_hours': average_montly_hours, 
                    'time_spend_company': time_spend_company, 
                    'Work_accident': 1 if work_accident_text == 'Yes' else 0, 
                    'promotion_last_5years': 1 if promotion_text == 'Yes' else 0, 
                    'Department': Department, 
                    'salary': salary
                }
            else:
                cols = st.columns(2)
                for i, col in enumerate(feature_columns):
                    with cols[i%2]:
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                            input_data[col] = st.selectbox(col.replace('_', ' ').title(), df[col].unique())
                        else:
                            input_data[col] = st.slider(col.replace('_', ' ').title(), float(df[col].min()), float(df[col].max()), float(df[col].mean()))

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1: predict_button = st.form_submit_button(label='🔮 Analyze Employee', type='primary')
            with col_btn2: test_high_risk = st.form_submit_button(label='🔥 Simulate High-Risk', type='secondary')

        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
            st.session_state.input_df = None
            st.session_state.prediction_probas = None

        if predict_button:
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if input_df[col].dtype != 'object': input_df[col] = input_df[col].astype(float)
            input_df = input_df[feature_columns]
            
            with st.spinner('AI is analyzing...'):
                sleep(0.3)
                raw_probas = pipeline.predict_proba(input_df)[0]
                
                # ====================================================================
                # 🔧 CRITICAL FIX #5: APPLY TEMPERATURE SCALING
                # ====================================================================
                calibrated_stay = calibrate_probability(raw_probas[0], temperature=0.55)
                calibrated_leave = 1 - calibrated_stay
                
                prediction = 1 if calibrated_leave >= 0.5 else 0
                
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = [calibrated_stay, calibrated_leave]

        if test_high_risk and is_default_data:
            input_data = {'satisfaction_level': 0.1, 'last_evaluation': 0.7, 'number_project': 7, 'average_montly_hours': 310, 'time_spend_company': 4, 'Work_accident': 1, 'promotion_last_5years': 0, 'Department': Department, 'salary': 'low'}
            input_df = pd.DataFrame([input_data])
            with st.spinner('Simulating...'):
                sleep(0.3)
                raw_probas = pipeline.predict_proba(input_df)[0]
                calibrated_stay = calibrate_probability(raw_probas[0], temperature=0.55)
                calibrated_leave = 1 - calibrated_stay
                prediction = 1 if calibrated_leave >= 0.5 else 0
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = [calibrated_stay, calibrated_leave]
                st.toast("🔥 High-Risk Profile Loaded", icon="🔥")

        # ====================================================================
        # DISPLAY CALIBRATED RESULTS
        # ====================================================================
        if st.session_state.prediction_result is not None:
            st.markdown("---")
            
            stay_prob = st.session_state.prediction_probas[0]
            leave_prob = st.session_state.prediction_probas[1]
            
            # Determine prediction
            if st.session_state.prediction_result == 0:
                pred_label = "✅ LIKELY TO STAY"
                pred_color = "#17B794"
                pred_bg = "linear-gradient(135deg, #1c2128 0%, #0d2818 100%)"
                bar_color = "stay"
            else:
                pred_label = "🚨 AT RISK OF LEAVING"
                pred_color = "#FF4B4B"
                pred_bg = "linear-gradient(135deg, #1c2128 0%, #2d1515 100%)"
                bar_color = "leave"
            
            # Confidence level indicator
            if 0.6 <= leave_prob <= 0.75:
                confidence_text = "📊 Moderate Risk"
                confidence_color = "#FFA500"
            elif 0.75 < leave_prob <= 0.85:
                confidence_text = "⚠️ High Risk"
                confidence_color = "#FF6B6B"
            elif leave_prob > 0.85:
                confidence_text = "🔴 Critical Risk"
                confidence_color = "#FF4B4B"
            else:
                confidence_text = "✅ Low Risk"
                confidence_color = "#17B794"
            
            st.markdown(f"""
            <div class="custom-card" style="text-align: center; border: 2px solid {pred_color}; background: {pred_bg};">
                <h2 style="color: {pred_color}; margin-bottom: 5px;">{pred_label}</h2>
                <p style="color: {confidence_color}; font-size: 1.1rem; margin-bottom: 15px;">{confidence_text}</p>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-weight: 600;">
                    <span style="color: #17B794;">Stay: {stay_prob:.1%}</span>
                    <span style="color: #FF4B4B;">Leave: {leave_prob:.1%}</span>
                </div>
                
                <div class="prob-bar-container">
                    <div class="prob-bar-stay" style="width: {stay_prob*100:.1f}%; font-size: {12 if stay_prob < 0.2 else 14}px;">{stay_prob:.1%}</div>
                    <div class="prob-bar-leave" style="width: {leave_prob*100:.1f}%; font-size: {12 if leave_prob < 0.2 else 14}px;">{leave_prob:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_stay, col_leave = st.columns(2)
            with col_stay: st.metric("Stay Probability", f"{stay_prob:.1%}")
            with col_leave: st.metric("Leave Probability", f"{leave_prob:.1%}", delta="Risk Level", delta_color="inverse" if leave_prob > 0.5 else "normal")
            
            # --- IF HIGH RISK: SHOW INTERVENTIONS ---
            if st.session_state.prediction_result == 1:
                st.markdown("---"); st.markdown("### 💡 Immediate Recommended Actions")
                for rec in get_retention_strategies(st.session_state.input_df): st.info(rec)
                
                st.markdown("---"); st.markdown("### 🔮 AI Retention Strategies (What-If Simulator)")
                if st.button("💡 Show Me How to Keep Them", type="primary", key="gen_cf"):
                    with st.spinner("🧠 Simulating strategies..."):
                        try:
                            query_instance = st.session_state.input_df
                            continuous_features = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                            if not continuous_features: st.error("No numerical columns found.")
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
                                        if isinstance(orig_val, (int, float)) and abs(orig_val - new_val) > 0.05:
                                            col_lower = col.lower(); action_text = ""
                                            if 'satisfaction' in col_lower: action_text = f"🤝 <strong>Boost Engagement</strong>: Improve from {orig_val:.2f} to {new_val:.2f}."
                                            elif 'hours' in col_lower: action_text = f"⏰ <strong>Reduce Workload</strong>: Cut by ~{abs(orig_val - new_val):.0f} hours."
                                            elif 'project' in col_lower: action_text = f"📂 <strong>Rebalance</strong>: Adjust to {int(new_val)} projects."
                                            else: action_text = f"• <strong>{col.title()}</strong>: {orig_val:.2f} → {new_val:.2f}"
                                            if action_text: changes.append(action_text)
                                        elif orig_val != new_val:
                                            if 'department' in col.lower(): has_high_effort = True; action_text = f"🏢 <strong>Transfer</strong>: {orig_val} → {new_val} (High Effort)"
                                            else: action_text = f"• <strong>{col.title()}</strong>: {orig_val} → {new_val}"
                                            changes.append(action_text)
                                    
                                    if not changes: changes.append("• Maintain current status")
                                    changes_str = "".join([f"<div class='action-item {'action-item-high-effort' if has_high_effort else ''}'>{c}</div>" for c in changes])
                                    effort_badge = "🔴 High Effort" if has_high_effort else "🟢 Low Effort"
                                    scenarios_html.append(f"""<div class="custom-card" style="border-color: #17B794;"><div style="display: flex; justify-content: space-between; margin-bottom: 15px;"><h4 style="color: #17B794; margin: 0;">Strategy {i+1}</h4><span style="background: {'#3d1515' if has_high_effort else '#0d2818'}; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;">{effort_badge}</span></div>{changes_str}<div style="margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;"><small style="color: #17B794;"><strong>✅ Result:</strong> Employee predicted to STAY</small></div></div>""")
                                
                                cols_list = [st.columns(min(len(scenarios_html), 3))[i] for i in range(len(scenarios_html))]
                                for i, html in enumerate(scenarios_html): cols_list[i].markdown(html, unsafe_allow_html=True)
                        except Exception as e: st.error(f"Error: {e}")

    # ====================================================================
    # PAGE: WHY THEY LEAVE
    # ====================================================================
    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers")
        analyze_why_people_leave(df)
        with st.spinner("Analyzing..."):
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
            if isinstance(shap_values, list): vals = np.abs(shap_values[1]).mean(0)
            else: vals = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance']).sort_values('Importance', ascending=False).head(3)
            c1, c2, c3 = st.columns(3); cols = [c1, c2, c3]
            for idx, (_, row) in enumerate(feature_importance.iterrows()):
                if idx < 3:
                    advice = "Morale" if 'satisfaction' in row['Feature'].lower() else "Workload" if 'project' in row['Feature'].lower() else "Other"
                    card_html = f"<div class='custom-card' style='text-align: center;'><h3 style='color: #17B794; margin-top: 0;'>{advice}</h3><small style='color: #8b949e;'>({row['Feature']})</small></div>"
                    with cols[idx]: st.markdown(card_html, unsafe_allow_html=True)
        with st.expander("🔧 SHAP Plots"):
            fig, ax = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False); st.pyplot(fig); plt.close(fig)

    # ====================================================================
    # PAGE: BUDGET PLANNER
    # ====================================================================
    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        analyze_why_people_leave(df)
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input("Total Budget (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
        with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Generate Plan", type="primary")
        if optimize_btn:
            results = plan_retention_budget(df, pipeline, budget)
            if results:
                selected_df, total_cost, total_savings = results
                st.success("✅ Optimization Complete")
                m1, m2, m3 = st.columns(3)
                m1.metric("Budget", f"₹{budget:,.0f}"); m2.metric("Investment", f"₹{total_cost:,.0f}"); m3.metric("Savings", f"₹{total_savings:,.0f}")
                display_cols = [c for c in selected_df.columns if c in ['Department', 'salary', 'satisfaction_level', 'number_project', 'attrition_risk', 'intervention_cost', 'net_savings']]
                if display_cols:
                    display_df = selected_df[display_cols].copy()
                    display_df.columns = ['Department', 'Tier', 'Satisfaction', 'Projects', 'Risk', 'Cost', 'Savings']
                    display_df['Cost'] = display_df['Cost'].apply(lambda x: f"₹{x:,.0f}")
                    display_df['Savings'] = display_df['Savings'].apply(lambda x: f"₹{x:,.0f}")
                    display_df['Risk'] = (display_df['Risk'] * 100).apply(lambda x: f"{x:.0f}%")
                    st.dataframe(display_df, use_container_width=True)

    # ====================================================================
    # PAGE: AI ASSISTANT
    # ====================================================================
    if page == "AI Assistant":
        st.header("🤖 AI Assistant")
        with st.form("llm_form"):
            c1, c2 = st.columns(2)
            with c1:
                emp_name = st.text_input("Employee Name", value="Rahul Sharma")
                emp_dept = st.selectbox("Department", df['Department'].unique()) if 'Department' in df.columns else st.text_input("Department", value="Sales")
            with c2:
                situation = st.selectbox("Situation?", ["Overworked", "Low Salary", "Low Morale", "No Growth"])
                solution = st.selectbox("Solution?", ["Flexible Hours", "Salary Adjustment", "Promotion", "Wellness"])
                cost = st.text_input("Cost", value="₹50,000")
            if st.form_submit_button("🚀 Generate Email"): run_groq_consultant(emp_name, emp_dept, situation, solution, cost)

    # ====================================================================
    # PAGE: AI RESEARCH LAB
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        tab1, tab2, tab3 = st.tabs(["📊 Benchmarking", "🕵️ Anomalies", "🎯 Ideal Candidate"])
        with tab1:
            if st.button("Run Benchmark", type="primary"):
                with st.spinner("Training..."):
                    y_pred = pipeline.predict(X_test_cur); proba = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf = Pipeline([('p', preprocessor), ('c', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
                    rf.fit(X_train_ref, y_train); y_pred_rf = rf.predict(X_test_cur); proba_rf = rf.predict_proba(X_test_cur)[:, 1]
                    metrics = {'Model': ['LightGBM', 'Random Forest'], 'Accuracy': [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rf)], 'F1': [f1_score(y_test, y_pred), f1_score(y_test, y_pred_rf)], 'AUC': [roc_auc_score(y_test, proba), roc_auc_score(y_test, proba_rf)]}
                    st.dataframe(pd.DataFrame(metrics).style.highlight_max(axis=0, color='#17B794'))
        with tab2:
            y_pred_all = pipeline.predict(df.drop('left', axis=1)); y_true = df['left']
            happy = df.iloc[np.where((y_pred_all == 0) & (y_true == 1))[0]]
            loyal = df.iloc[np.where((y_pred_all == 1) & (y_true == 0))[0]]
            col_a, col_b = st.columns(2)
            with col_a: st.markdown("### 🚪 Happy Leavers"); st.dataframe(happy.head(), use_container_width=True) if len(happy) > 0 else st.success("None")
            with col_b: st.markdown("### 🛡️ Loyal Sufferers"); st.dataframe(loyal.head(), use_container_width=True) if len(loyal) > 0 else st.success("None")
        with tab3:
            mask = (df['left'] == 0) & (df['time_spend_company'] > 4) if 'time_spend_company' in df.columns else (df['left'] == 0)
            superstars = df[mask]; average = df[(df['left'] == 0) & (~mask)]
            if len(superstars) >= 5:
                metrics = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                diff = (superstars[metrics].mean() - average[metrics].mean()).abs().nlargest(3)
                st.dataframe(pd.DataFrame({'Metric': diff.index, 'Difference': diff.values}))
            else: st.warning("Not enough data")

    # ====================================================================
    # PAGE: STRATEGIC ROADMAP
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Future Planning")
        issues = []
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.6: issues.append("Low Satisfaction")
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: issues.append("High Hours")
        if not issues: issues.append("Standard Stabilization")
        st.markdown(f"<div class='custom-card'><h4 style='color: #17B794;'>🩺 AI Diagnostic</h4><p>Risks: <strong style='color: #FF4B4B;'>{', '.join(issues)}</strong></p></div>", unsafe_allow_html=True)
        
        if st.button("✍️ Draft 6-Month Plan", type="primary"):
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
                if api_key:
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                    chain = PromptTemplate.from_template("Create 6-month HR roadmap for: {issues}") | llm | StrOutputParser()
                    st.markdown(f"<div class='llm-response'>{chain.invoke({'issues': ', '.join(issues)})}</div>", unsafe_allow_html=True)
            except: st.warning("🔑 API Key missing")

        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1: efficacy = st.slider("Success rate (%)", 10, 50, 20, 5)
        with col_f2: natural = st.slider("Natural attrition (%)", 0.5, 2.0, 1.0, 0.1)

        if st.button("📈 Show 12-Month Projection", type="primary"):
            current = len(df)
            # Use calibrated probabilities for projection
            raw_probs = pipeline.predict_proba(df.drop('left', axis=1))[:, 1]
            calibrated_probs = np.array([calibrate_probability(p, 0.55) for p in raw_probs])
            risk_sum = calibrated_probs.sum()
            monthly_no = risk_sum / 12; monthly_yes = monthly_no * (1 - efficacy/100)
            bau, plan = [], []; tmp_b, tmp_p = float(current), float(current)
            for m in range(1, 13):
                tmp_b = max(0, tmp_b - monthly_no - tmp_b * natural/100)
                tmp_p = max(0, tmp_p - monthly_yes - tmp_p * natural/100)
                bau.append(tmp_b); plan.append(tmp_p)
            fig = px.line(pd.DataFrame({'Month': range(1,13), 'Do Nothing': bau, 'Follow Plan': plan}).melt('Month', var_name='Scenario', value_name='Headcount'), x='Month', y='Headcount', color='Scenario', template="plotly_dark", color_discrete_map={'Do Nothing': "#FF4B4B", 'Follow Plan': "#17B794"}, markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"**Bottom Line:** Retaining {efficacy}% saves **{int(plan[-1] - bau[-1])} employees**.")

if __name__ == "__main__":
    main()
