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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
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
        border-left: 3px solid #FF4B4B; 
    }
    
    /* Probability bar styling */
    .prob-bar-container {
        width: 100%;
        background-color: #21262d;
        border-radius: 10px;
        overflow: hidden;
        height: 30px;
        margin-top: 10px;
    }
    .prob-bar-stay {
        height: 100%;
        background: linear-gradient(90deg, #17B794, #11998e);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 14px;
        transition: width 0.5s ease;
    }
    .prob-bar-leave {
        height: 100%;
        background: linear-gradient(90deg, #FF4B4B, #cc3a3a);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 14px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

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
# Logic Engine Functions (Evaluation 1)
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
        st.info("📊 *Advanced Causal Graph requires specific columns (satisfaction_level, salary, etc.) which are not in this uploaded dataset.*")


def plan_retention_budget(df, pipeline, budget_limit):
    st.markdown("### 💰 Retention Budget Planner")
    st.markdown("<p style='color: #9ca3af;'>Optimize your spend to save on replacement costs.</p>", unsafe_allow_html=True)
    X = df.drop('left', axis=1)
    probas = pipeline.predict_proba(X)[:, 1]
    opt_df = df.copy()
    opt_df['attrition_risk'] = probas
    high_risk_df = opt_df[opt_df['attrition_risk'] > 0.5].copy()
    
    if len(high_risk_df) == 0:
        st.success("🎉 Great news! The workforce is stable.")
        return None

    if 'salary' in df.columns:
        salary_val_map = {'low': 400000, 'medium': 600000, 'high': 900000}
        high_risk_df['annual_salary'] = high_risk_df['salary'].map(salary_val_map)
    else:
        high_risk_df['annual_salary'] = 500000 

    high_risk_df['replacement_cost'] = high_risk_df['annual_salary'] * 0.5
    high_risk_df['expected_loss'] = high_risk_df['attrition_risk'] * high_risk_df['replacement_cost']
    high_risk_df['intervention_cost'] = high_risk_df['annual_salary'] * 0.10
    high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['intervention_cost']

    candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
    if len(candidates) == 0:
        st.warning("⚠️ It is currently not cost-effective to offer raises.")
        return None

    n = len(candidates)
    c = -candidates['net_savings'].values 
    A = np.array([candidates['intervention_cost'].values])
    b = np.array([budget_limit])
    integrality = np.ones(n)
    
    with st.spinner("🧮 Calculating optimal resource allocation..."):
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb=-np.inf, ub=b), integrality=integrality)
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            return None

    if res.success:
        selected_indices = np.where(res.x == 1)[0]
        selected_employees = candidates.iloc[selected_indices]
        total_cost = selected_employees['intervention_cost'].sum()
        total_savings = selected_employees['net_savings'].sum()
        return selected_employees, total_cost, total_savings
    else:
        st.error("❌ Budget is too low.")
        return None

# ====================================================================
# Evaluation 2: Intelligent Interface Functions
# ====================================================================
def run_groq_consultant(employee_name, department, situation, solution, budget):
    st.subheader("✍️ AI Communication Assistant")
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key: st.warning("🔑 System Error: API Key missing."); return
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
    except Exception as e: st.error(f"Connection Error: {e}"); return

    if "overwork" in situation.lower(): root_cause = "High Workload & Potential Burnout"
    elif "salary" in situation.lower(): root_cause = "Compensation & Salary Competitiveness"
    elif "morale" in situation.lower(): root_cause = "Low Job Satisfaction & Morale"
    else: root_cause = "Attrition Risk Factors"

    template = """You are an expert HR Consultant.
**Employee:** {employee_name} ({department})
**Situation:** {situation}
**Root Cause:** {root_cause}
**Solution:** {action_description}
**Cost:** {cost_str}
**Task:** Write a professional email draft. Acknowledge value, address situation, propose solution.
**Tone:** Professional, Supportive."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    with st.spinner("Drafting your message..."):
        try:
            response = chain.invoke({"employee_name": employee_name, "department": department, "situation": situation, "root_cause": root_cause, "action_description": solution, "cost_str": budget})
            st.markdown("#### 📧 Generated Email Draft")
            st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
        except Exception as e: st.error(f"Error: {e}")


# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="RetainAI | Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

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
        def load_data_and_train_model():
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
            
            # ====================================================================
            # 🔧 CRITICAL FIX #1: REMOVED StandardScaler
            # ====================================================================
            # Tree models (LightGBM, Random Forest) do NOT need feature scaling.
            # StandardScaler was COMPRESSING values, causing the sigmoid function
            # in LightGBM to output extreme probabilities (0.1% or 99.9%).
            # ====================================================================
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),  # ✅ NO SCALING
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # ✅ dense output
                ])
            
            st.write("🤖 Step 3/3: Training AI Model (LightGBM)...")
            
            # ====================================================================
            # 🔧 CRITICAL FIX #2: CALIBRATED MODEL PARAMETERS
            # ====================================================================
            # PROBLEM: Previous settings created overconfident, extreme predictions
            # SOLUTION: Reduce complexity + Add regularization + Lower class weight
            # ====================================================================
            best_params = {
                'n_estimators': 200,          # Reduced: Less overfitting
                'learning_rate': 0.05,        # Keep: Stable learning
                'num_leaves': 15,             # REDUCED from 31: Simpler trees
                'max_depth': 5,               # REDUCED from 10: Shallower trees
                'min_child_samples': 30,      # NEW: Prevents tiny leaf nodes
                'reg_alpha': 0.1,             # NEW: L1 regularization
                'reg_lambda': 0.1,            # NEW: L2 regularization
                'random_state': 42,
                'verbose': -1,
                'scale_pos_weight': 2.0       # REDUCED from 5: Less aggressive
            }
            
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', lgb.LGBMClassifier(**best_params))])
            
            final_pipeline.fit(X_train, y_train)
            
            return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

        pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model()
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
        
        page = option_menu(
            menu_title=None,
            options=['⚙️ Global Setup', 'Home', 'Employee Insights', 'Predict Attrition', 'Why They Leave', 'Budget Planner', 'AI Assistant', 'AI Research Lab', 'Strategic Roadmap'],  
            icons=['gear', 'house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill'], 
            menu_icon="cast", default_index=0, 
            styles={
                "container": {"padding": "0!important", "background-color": 'transparent'},
                "icon": {"color": "#17B794", "font-size": "18px"},
                "nav-link": {"color": "#c9d1d9", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "10px"},
                "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"},
            }
        )
        
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
                unique_vals = new_df[target_col].unique(); left_value = st.selectbox(f"In '{target_col}', which value means 'Left'?", unique_vals)
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
                        if len(categorical_auto) == 0: 
                            preprocessor_global = ColumnTransformer(transformers=[('num', 'passthrough', numerical_auto)])
                        else: 
                            preprocessor_global = ColumnTransformer(transformers=[
                                ('num', 'passthrough', numerical_auto),  # ✅ NO SCALING
                                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_auto)
                            ])
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        
                        neg_count = (y_train_g == 0).sum()
                        pos_count = (y_train_g == 1).sum()
                        spw = min(neg_count / pos_count, 3.0) if pos_count > 0 else 1.0  # Cap at 3.0
                        
                        global_pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor_global), 
                            ('classifier', lgb.LGBMClassifier(
                                n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1,
                                num_leaves=15, max_depth=5, min_child_samples=30,
                                reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=spw
                            ))
                        ])
                        global_pipeline.fit(X_train_g, y_train_g)
                        y_pred_g = global_pipeline.predict(X_test_g); acc = accuracy_score(y_test_g, y_pred_g)
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_train'] = y_train_g; st.session_state['global_y_test'] = y_test_g
                        st.session_state['is_global'] = True
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
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        total_employees = len(df); attrition_rate = (df['left'].sum() / len(df)) * 100
        
        if 'satisfaction_level' in df.columns:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        
        st.markdown("---")
        st.dataframe(df.head(100), use_container_width=True)

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
    # PAGE: PREDICT ATTRITION (FULLY FIXED VERSION)
    # ====================================================================
    if page == "Predict Attrition":
        st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af;'>Enter employee details to assess risk.</p>", unsafe_allow_html=True)
        
        # --- MODEL DIAGNOSTICS ---
        with st.expander("🧪 Model Diagnostics (Verification)", expanded=False):
            st.write("Test the AI against real historical data:")
            c_test1, c_test2 = st.columns(2)
            
            with c_test1:
                if st.button("Test with Employee who Left"):
                    sample = df[df['left'] == 1].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    pred = pipeline.predict(test_df)[0]
                    prob = pipeline.predict_proba(test_df)[0][1]
                    if pred == 1: st.success(f"✅ Correct! Predicted LEAVE ({prob:.1%} risk)")
                    else: st.error(f"❌ Wrong! Predicted STAY ({prob:.1%} risk)")
                    with st.expander("See data"): st.json(sample.to_dict())

            with c_test2:
                if st.button("Test with Employee who Stayed"):
                    sample = df[df['left'] == 0].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    pred = pipeline.predict(test_df)[0]
                    prob = pipeline.predict_proba(test_df)[0][1]
                    if pred == 0: st.success(f"✅ Correct! Predicted STAY ({prob:.1%} risk)")
                    else: st.error(f"❌ Wrong! Predicted LEAVE ({prob:.1%} risk)")
                    with st.expander("See data"): st.json(sample.to_dict())

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
                            min_val = float(df[col].min()); max_val = float(df[col].max())
                            input_data[col] = st.slider(col.replace('_', ' ').title(), min_value=min_val, max_value=max_val, value=float(df[col].mean()))

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1: predict_button = st.form_submit_button(label='🔮 Analyze Employee', type='primary')
            with col_btn2: test_high_risk = st.form_submit_button(label='🔥 Simulate High-Risk Employee', type='secondary')

        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
            st.session_state.input_df = None
            st.session_state.prediction_probas = None

        if predict_button:
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if input_df[col].dtype != 'object':
                    input_df[col] = input_df[col].astype(float)
            input_df = input_df[feature_columns]
            
            with st.spinner('AI is analyzing...'):
                sleep(0.5)
                prediction = pipeline.predict(input_df)[0]
                prediction_probas = pipeline.predict_proba(input_df)[0]
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = prediction_probas

        if test_high_risk and is_default_data:
            input_data = {
                'satisfaction_level': 0.1, 'last_evaluation': 0.7, 'number_project': 7, 
                'average_montly_hours': 310, 'time_spend_company': 4, 'Work_accident': 1, 
                'promotion_last_5years': 0, 'Department': Department, 'salary': 'low'
            }
            input_df = pd.DataFrame([input_data])
            with st.spinner('Simulating high-risk scenario...'):
                sleep(0.5)
                prediction = pipeline.predict(input_df)[0]
                prediction_probas = pipeline.predict_proba(input_df)[0]
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = prediction_probas
                st.toast("🔥 High-Risk Profile Loaded", icon="🔥")
        
        if test_high_risk and not is_default_data:
            st.warning("⚠️ **High-Risk Simulation** only available with default demo dataset.")

        # ====================================================================
        # 🔧 CRITICAL FIX #3: VISUAL PROBABILITY DISPLAY
        # ====================================================================
        # Instead of just showing numbers, show a visual bar so users can
        # see the actual confidence level without being misled by 99.9%
        # ====================================================================
        if st.session_state.prediction_result is not None:
            st.markdown("---")
            
            stay_prob = st.session_state.prediction_probas[0]
            leave_prob = st.session_state.prediction_probas[1]
            
            # Determine prediction label and color
            if st.session_state.prediction_result == 0:
                pred_label = "✅ LIKELY TO STAY"
                pred_color = "#17B794"
                pred_bg = "linear-gradient(135deg, #1c2128 0%, #0d2818 100%)"
            else:
                pred_label = "🚨 AT RISK OF LEAVING"
                pred_color = "#FF4B4B"
                pred_bg = "linear-gradient(135deg, #1c2128 0%, #2d1515 100%)"
            
            # Prediction Card
            st.markdown(f"""
            <div class="custom-card" style="text-align: center; border: 2px solid {pred_color}; background: {pred_bg};">
                <h2 style="color: {pred_color}; margin-bottom: 15px;">{pred_label}</h2>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #17B794; font-weight: 600;">Stay: {stay_prob:.1%}</span>
                    <span style="color: #FF4B4B; font-weight: 600;">Leave: {leave_prob:.1%}</span>
                </div>
                
                <div class="prob-bar-container">
                    <div class="prob-bar-stay" style="width: {stay_prob*100:.1f}%;">{stay_prob:.1%}</div>
                    <div class="prob-bar-leave" style="width: {leave_prob*100:.1f}%;">{leave_prob:.1%}</div>
                </div>
                
                <p style="color: #8b949e; font-size: 0.85rem; margin-top: 15px; margin-bottom: 0;">
                    {'⚠️ High confidence prediction' if (stay_prob > 0.8 or leave_prob > 0.8) else '📊 Moderate confidence - consider additional factors'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            col_stay, col_leave = st.columns(2)
            with col_stay:
                st.metric("Stay Probability", f"{stay_prob:.1%}")
            with col_leave:
                delta_color = "inverse" if leave_prob > 0.5 else "normal"
                st.metric("Leave Probability", f"{leave_prob:.1%}", delta="Risk Level", delta_color=delta_color)
            
            # --- IF HIGH RISK: SHOW INTERVENTIONS ---
            if st.session_state.prediction_result == 1:
                st.markdown("---"); 
                st.markdown("### 💡 Immediate Recommended Actions")
                for rec in get_retention_strategies(st.session_state.input_df): 
                    st.info(rec)
                
                # --- COUNTERFACTUAL STRATEGIES ---
                st.markdown("---"); 
                st.markdown("### 🔮 AI Retention Strategies (What-If Simulator)")
                st.write("""
                <p style='color: #9ca3af; margin-bottom: 15px;'>
                    Here are <strong>3 different ways</strong> to prevent this employee from leaving.<br>
                    <span style='color: #17B794;'>🟢 Green border</span> = Easy to implement &nbsp;|&nbsp; 
                    <span style='color: #FF4B4B;'>🔴 Red border</span> = High effort required
                </p>
                """, unsafe_allow_html=True)
                
                if st.button("💡 Show Me How to Keep Them", type="primary", key="gen_cf"):
                    with st.spinner("🧠 Simulating retention strategies..."):
                        try:
                            query_instance = st.session_state.input_df
                            continuous_features = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                            
                            if not continuous_features: 
                                st.error("❌ No numerical columns found for simulation.")
                            else:
                                d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left')
                                m = dice_ml.Model(model=pipeline, backend='sklearn')
                                exp = Dice(d, m, method='random')
                                cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
                                cf_df = cf.cf_examples_list[0].final_cfs_df
                                original = query_instance.iloc[0]
                                
                                scenarios_html = []
                                for i in range(len(cf_df)):
                                    changes = []; 
                                    cf_row = cf_df.iloc[i]; 
                                    has_high_effort = False
                                    
                                    for col in original.index:
                                        orig_val = original[col]; 
                                        new_val = cf_row[col]
                                        
                                        if isinstance(orig_val, (int, float)):
                                            if abs(orig_val - new_val) > 0.05:
                                                col_lower = col.lower(); 
                                                action_text = ""
                                                if 'satisfaction' in col_lower: 
                                                    action_text = f"🤝 <strong>Boost Engagement</strong>: Improve satisfaction from <strong>{orig_val:.2f}</strong> to <strong>{new_val:.2f}</strong>."
                                                elif 'hours' in col_lower:
                                                    diff = orig_val - new_val
                                                    if diff > 0: 
                                                        action_text = f"⏰ <strong>Reduce Workload</strong>: Cut monthly hours by ~<strong>{abs(diff):.0f}</strong>."
                                                    else: 
                                                        action_text = f"⏰ <strong>Adjust Hours</strong>: Change to ~<strong>{new_val:.0f}</strong>."
                                                elif 'project' in col_lower: 
                                                    action_text = f"📂 <strong>Rebalance Projects</strong>: Adjust to <strong>{int(new_val)}</strong> projects."
                                                elif 'evaluation' in col_lower: 
                                                    action_text = f"📊 <strong>Performance Coaching</strong>: Guide score to <strong>{new_val:.2f}</strong>."
                                                else: 
                                                    action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val:.2f} to {new_val:.2f}."
                                                if action_text: 
                                                    changes.append(action_text)
                                        else:
                                            if orig_val != new_val:
                                                if 'department' in col.lower():
                                                    has_high_effort = True
                                                    action_text = f"🏢 <strong>Department Transfer</strong>: Move from <strong>{orig_val}</strong> to <strong>{new_val}</strong>. <span style='color:#FF4B4B;'>(High Effort)</span>"
                                                elif 'salary' in col.lower():
                                                    action_text = f"💰 <strong>Salary Adjustment</strong>: Change from <strong>{orig_val}</strong> to <strong>{new_val}</strong>."
                                                else: 
                                                    action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val} to {new_val}."
                                                changes.append(action_text)
                                    
                                    if not changes: 
                                        changes.append("• (AI suggests maintaining current status)")
                                    
                                    changes_str = "".join([f"<div class='action-item {'action-item-high-effort' if has_high_effort else ''}'>{c}</div>" for c in changes])
                                    effort_badge = "🔴 High Effort" if has_high_effort else "🟢 Low Effort"
                                    
                                    scenarios_html.append(f"""
                                        <div class="custom-card" style="border-color: #17B794; height: 100%;">
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                                <h4 style="color: #17B794; margin: 0;">Strategy {i+1}</h4>
                                                <span style="background: {'#3d1515' if has_high_effort else '#0d2818'}; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;">
                                                    {effort_badge}
                                                </span>
                                            </div>
                                            <p style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                                                {changes_str}
                                            </p>
                                            <div style="margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;">
                                                <small style="color: #17B794;">
                                                    <strong>✅ Predicted Outcome:</strong> If implemented, the AI predicts this employee will <strong>STAY</strong>.
                                                </small>
                                            </div>
                                        </div>
                                    """)

                                if len(scenarios_html) >= 3:
                                    col_s1, col_s2, col_s3 = st.columns(3); 
                                    cols_list = [col_s1, col_s2, col_s3]
                                else:
                                    cols_list = [st.columns(len(scenarios_html))[i] for i in range(len(scenarios_html))]
                                
                                for i, html in enumerate(scenarios_html):
                                    with cols_list[i]: 
                                        st.markdown(html, unsafe_allow_html=True)
                                        
                        except Exception as e: 
                            st.error(f"❌ Error generating strategies: {e}")

    # ====================================================================
    # PAGE: WHY THEY LEAVE
    # ====================================================================
    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers")
        analyze_why_people_leave(df)
        with st.spinner("Analyzing model insights..."):
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
            if isinstance(shap_values, list): vals = np.abs(shap_values[1]).mean(0)
            else: vals = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance'])
            feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
            top_3 = feature_importance.head(3)
            def get_feature_advice(feature_name):
                if 'satisfaction' in feature_name.lower(): return "Employee Morale", "Conduct engagement surveys."
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
            fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False)
            st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)

    # ====================================================================
    # PAGE: BUDGET PLANNER
    # ====================================================================
    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        analyze_why_people_leave(df)
        st.markdown("---"); st.markdown("### 💰 Budget Optimization Tool")
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input("Total Retention Budget (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
        with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Generate Plan", type="primary")
        if optimize_btn:
            results = plan_retention_budget(df, pipeline, budget)
            if results:
                selected_df, total_cost, total_savings = results
                st.success("✅ **Optimization Complete.**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Budget Allocated", f"₹{budget:,.0f}")
                m2.metric("Investment Needed", f"₹{total_cost:,.0f}")
                m3.metric("Projected Savings", f"₹{total_savings:,.0f}")
                display_cols = [c for c in selected_df.columns if c in ['Department', 'salary', 'satisfaction_level', 'number_project', 'attrition_risk', 'intervention_cost', 'net_savings']]
                if len(display_cols) > 0:
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
                if 'Department' in df.columns: emp_dept = st.selectbox("Department", df['Department'].unique())
                else: emp_dept = st.text_input("Department", value="Sales")
            with c2:
                situation_input = st.selectbox("Situation?", ["Overworked & Burned out", "Seeking Higher Salary", "Low Morale", "Lack of Growth"])
                solution_input = st.selectbox("Solution?", ["Flexible Hours", "Salary Adjustment", "Promotion", "Wellness Session"])
                cost_input = st.text_input("Cost (Optional)", value="₹50,000")
            if st.form_submit_button("🚀 Generate Email"): run_groq_consultant(emp_name, emp_dept, situation_input, solution_input, cost_input)

    # ====================================================================
    # PAGE: AI RESEARCH LAB
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        tab1, tab2, tab3 = st.tabs(["📊 Benchmarking", "🕵️ Anomaly Detection", "🎯 Ideal Candidate"])
        
        with tab1:
            if st.button("Run Benchmark", type="primary"):
                with st.spinner("Training models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
                    rf.fit(X_train_ref, y_train); y_pred_rf = rf.predict(X_test_cur); proba_rf = rf.predict_proba(X_test_cur)[:, 1]
                    lr = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
                    lr.fit(X_train_ref, y_train); y_pred_lr = lr.predict(X_test_cur); proba_lr = lr.predict_proba(X_test_cur)[:, 1]
                    metrics = {'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'], 
                               'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)], 
                               'F1 Score': [f1_score(y_test, y_pred_lgbm), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_lr)], 
                               'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_lr)]}
                    st.dataframe(pd.DataFrame(metrics).style.highlight_max(axis=0, color='#17B794'))

        with tab2:
            X_all = df.drop('left', axis=1); y_true = df['left']; y_pred = pipeline.predict(X_all)
            happy = df.iloc[np.where((y_pred == 0) & (y_true == 1))[0]]
            loyal = df.iloc[np.where((y_pred == 1) & (y_true == 0))[0]]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🚪 Happy Leavers"); st.caption(f"Count: {len(happy)}")
                if len(happy) > 0: st.dataframe(happy.head(), use_container_width=True)
                else: st.success("✅ None found")
            with col_b:
                st.markdown("### 🛡️ Loyal Sufferers"); st.caption(f"Count: {len(loyal)}")
                if len(loyal) > 0: st.dataframe(loyal.head(), use_container_width=True)
                else: st.success("✅ None found")

        with tab3:
            if 'time_spend_company' in df.columns: mask = (df['left'] == 0) & (df['time_spend_company'] > 4)
            else: mask = (df['left'] == 0)
            superstars = df[mask]; average = df[(df['left'] == 0) & (~mask)]
            if len(superstars) < 5: st.warning("Not enough data")
            else:
                metrics = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                diff = (superstars[metrics].mean() - average[metrics].mean()).abs().nlargest(3)
                st.dataframe(pd.DataFrame({'Metric': diff.index, 'Difference': diff.values}))

    # ====================================================================
    # PAGE: STRATEGIC ROADMAP
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Future Planning")
        issues = []
        if 'satisfaction_level' in df.columns and df['satisfaction_level'].mean() < 0.6: issues.append("Low Satisfaction")
        if 'average_montly_hours' in df.columns and df['average_montly_hours'].mean() > 200: issues.append("High Working Hours")
        if not issues: issues.append("Standard Stabilization")
        
        st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: #17B794; margin-top: 0;">🩺 AI Diagnostic</h4>
            <p style="color: #c9d1d9;">Risks: <strong style="color: #FF4B4B;">{', '.join(issues)}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
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
        with col_f1: efficacy = st.slider("Retention success rate (%)", 10, 50, 20, 5)
        with col_f2: natural = st.slider("Natural attrition (%)", 0.5, 2.0, 1.0, 0.1)

        if st.button("📈 Show 12-Month Projection", type="primary"):
            current = len(df)
            risk_sum = pipeline.predict_proba(df.drop('left', axis=1))[:, 1].sum()
            monthly_no = risk_sum / 12; monthly_yes = monthly_no * (1 - efficacy/100)
            bau, plan = [], []; tmp_b, tmp_p = float(current), float(current)
            for m in range(1, 13):
                tmp_b = max(0, tmp_b - monthly_no - tmp_b * natural/100)
                tmp_p = max(0, tmp_p - monthly_yes - tmp_p * natural/100)
                bau.append(tmp_b); plan.append(tmp_p)
            fig = px.line(pd.DataFrame({'Month': range(1,13), 'Do Nothing': bau, 'Follow Plan': plan}).melt('Month', var_name='Scenario', value_name='Headcount'), 
                         x='Month', y='Headcount', color='Scenario', template="plotly_dark",
                         color_discrete_map={'Do Nothing': "#FF4B4B", 'Follow Plan': "#17B794"}, markers=True)
            st.plotly_chart(fig, use_container_width=True)
            saved = int(plan[-1] - bau[-1])
            st.success(f"**Bottom Line:** Retaining {efficacy}% of at-risk staff saves **{saved} employees**.")

if __name__ == "__main__":
    main()
