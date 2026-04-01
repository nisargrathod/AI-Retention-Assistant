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


def plan_retention_budget(df, pipeline, budget_limit):
    st.markdown("### 💰 Retention Budget Planner")
    st.markdown("<p style='color: #9ca3af;'>Optimize your spend to save on replacement costs.</p>", unsafe_allow_html=True)
    X = df.drop('left', axis=1)
    
    raw_probs = pipeline.predict_proba(X)[:, 1]
    probas = calibrate_probability_array(raw_probs, temperature=0.55)
    
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
            temperature=0.7
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
            st.error(f"Error generating draft: {e}")


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
                        st.balloons(); st.success(f"🎉 Training Complete! Accuracy: **{acc:.1%}**.")
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
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
            col3.metric("Avg. Satisfaction", f"{df['satisfaction_level'].mean():.2f} / 1.0")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Workforce", f"{total_employees:,}")
            col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        
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
                    else: st.error(f"❌ **Incorrect.** Prediction: Stay")
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

    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; margin-bottom: 30px;'>Data-driven decision support for HR.</p>", unsafe_allow_html=True)
        analyze_why_people_leave(df)
        st.markdown("---"); st.markdown("### 💰 Budget Optimization Tool")
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input("Total Retention Budget (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
        with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Generate Plan", type="primary")
        if optimize_btn:
            results = plan_retention_budget(df, pipeline, budget)
            if results:
                selected_df, total_cost, total_savings = results
                st.markdown("<br>", unsafe_allow_html=True); st.success("✅ **Optimization Complete.** Here is your strategic plan.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Budget Allocated", f"₹{budget:,.0f}")
                m2.metric("Investment Needed", f"₹{total_cost:,.0f}", delta=f"{(total_cost/budget)*100:.1f}% Used")
                m3.metric("Projected Savings", f"₹{total_savings:,.0f}", delta="ROI Positive")
                st.markdown("<br>", unsafe_allow_html=True); st.markdown("### 📋 Actionable Retention List")
                st.caption("Target these employees with a 10% retention bonus. It is cheaper to retain than to replace.")
                display_cols = [c for c in selected_df.columns if c in ['Department', 'department', 'salary', 'satisfaction_level', 'number_project', 'attrition_risk', 'intervention_cost', 'net_savings']]
                if len(display_cols) > 0:
                    display_df = selected_df[display_cols].copy()
                    if 'Department' in display_df.columns: display_df.rename(columns={'Department': 'Department'}, inplace=True)
                    elif 'department' in display_df.columns: display_df.rename(columns={'department': 'Department'}, inplace=True)
                    if 'salary' in display_df.columns: display_df.rename(columns={'salary': 'Tier'}, inplace=True)
                    if 'Cost to Retain' not in display_df.columns and 'intervention_cost' in display_df.columns:
                        display_df['Cost to Retain'] = display_df['intervention_cost'].apply(lambda x: f"₹{x:,.0f}")
                    if 'Savings' not in display_df.columns and 'net_savings' in display_df.columns:
                        display_df['Savings'] = display_df['net_savings'].apply(lambda x: f"₹{x:,.0f}")
                    if 'Risk' not in display_df.columns and 'attrition_risk' in display_df.columns:
                        display_df['Risk'] = (display_df['attrition_risk'] * 100).apply(lambda x: f"{x:.0f}%")
                    st.dataframe(display_df, use_container_width=True)

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
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Advanced modules for Strategy, Anomalies, and Recruitment.</p>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Benchmarking", "🔬 Departmental Strategy Deep Dive", "⚠️ Blind Spots", "📊 Retention Priority Matrix", "🎯 The 'Ideal Candidate' Profiler"])
        
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
                                        card_html = f"<div class='custom-card' style='border-top: 4px solid #17B794;'><div style='display: flex; align-items: center; margin-bottom: 10px;'><span style='font-size: 1.5rem; margin-right: 10px;'>{icon}</span><h4 style='margin: 0; color: #fff;'>{title}</h4></div><p style='color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;'>{advice}</p><small style='color: #8b949e;'>Driver: {feature_name.replace('_', ' ').title()}</small></div>"
                                        with col: st.markdown(card_html, unsafe_allow_html=True)

        # ====================================================================
        # COMPLETELY REBUILT: "Blind Spots" — Zero confusion, 100% actionable
        # ====================================================================
        with tab3:
            st.subheader("⚠️ Where Our AI Gets It Wrong")
            st.caption("No AI is perfect. This page shows you the people it misjudged — so you can catch what the data misses.")
            st.write("")
            
            X_all = df.drop('left', axis=1)
            y_true = df['left']
            
            raw_probs_anomaly = pipeline.predict_proba(X_all)[:, 1]
            cal_probs_anomaly = calibrate_probability_array(raw_probs_anomaly, temperature=0.55)
            y_pred = (cal_probs_anomaly >= 0.5).astype(int)
            
            missed_indices = np.where((y_pred == 0) & (y_true == 1))[0]
            surprise_indices = np.where((y_pred == 1) & (y_true == 0))[0]
            df_missed = df.iloc[missed_indices]
            df_surprise = df.iloc[surprise_indices]
            
            # ---- SECTION 1: PEOPLE WE MISSED ----
            st.markdown("---")
            st.markdown("### 🚨 People We Missed")
            
            if len(df_missed) > 0:
                st.error(f"**{len(df_missed)} people left, and our system thought they would stay.**")
                st.markdown("**Why does this happen?**")
                st.markdown("These people looked fine on paper — good satisfaction, normal workload. But they left anyway. This almost always means:")
                
                st.info("🎯 **A competitor offered them a better job.** Our data can't see external offers. If multiple people went to the same competitor, that's a pattern you need to address.")
                st.info("🏠 **Personal reasons.** Spouse relocation, health issues, or family changes don't show up in HR data.")
                st.info("😐 **Quiet quitting turned into actual quitting.** They seemed satisfied in surveys but were already mentally checked out.")
                
                st.markdown("**What you should do:**")
                st.success("1. **Pull the exit interviews** for these {len(df_missed)} people. Look for recurring reasons that our system can't detect.")
                st.success("2. **Check if they joined a specific competitor.** If yes, you have a poaching problem, not a retention problem.")
                st.success("3. **Don't blame the AI.** If 15%+ of leavers were missed, it means your data is missing something important — consider adding new survey questions.")
                
                with st.expander("📋 See who we missed"):
                    st.dataframe(df_missed.head(10), use_container_width=True)
            else:
                st.success("✅ **Nobody was missed.** The AI correctly predicted every person who left.")

            # ---- SECTION 2: WALKING TIME BOMBS ----
            st.markdown("---")
            st.markdown("### ⏰ Walking Time Bombs")
            
            if len(df_surprise) > 0:
                st.warning(f"**{len(df_surprise)} people look like they should leave any day now — but they haven't.**")
                st.markdown("**Why haven't they left yet?**")
                st.markdown("Their profile screams 'high risk' — low satisfaction, overworked, no promotion. But they're still here. The most common reasons:")
                
                st.error("⛓️ **Golden handcuffs.** Their salary, stock options, or benefits are too good to walk away from — even though they're unhappy. They'll leave the moment someone matches the pay.")
                st.error("🔍 **No better option right now.** They want to leave but can't find another job in this market. The moment the job market improves, they're gone.")
                st.error("🔇 **They've already given up.** They're doing the bare minimum work (quiet quitting). They're not a flight risk — they're already gone mentally. They just haven't updated their resume yet.")
                
                st.markdown("**What you should do THIS WEEK:**")
                st.success("1. **Do NOT ignore them.** The fact that they haven't left yet is a gift — you still have time to act.")
                st.success("2. **Have a real conversation.** Not a survey — an actual 1-on-1. Ask: 'What would make you want to stay here for another 2 years?'")
                st.success("3. **Check their manager.** In most cases, people don't leave companies — they leave bad managers. If multiple time bombs report to the same manager, that's your real problem.")
                st.success("4. **Fix the easiest thing first.** If it's a pay issue, a market adjustment is cheaper than replacement. If it's a workload issue, redistribute projects. Don't overthink it.")
                
                with st.expander("📋 See who's at risk"):
                    st.dataframe(df_surprise.head(10), use_container_width=True)
            else:
                st.success("✅ **No hidden risks.** Everyone the AI flagged as high-risk actually left. The model is well-calibrated.")

            # ---- SECTION 3: THE BOTTOM LINE ----
            if len(df_missed) > 0 or len(df_surprise) > 0:
                st.markdown("---")
                st.markdown("### 💡 The Bottom Line")
                
                missed_pct = (len(df_missed) / max(y_true.sum(), 1)) * 100
                surprise_pct = (len(df_surprise) / max(len(y_true) - y_true.sum(), 1)) * 100
                
                col_bl1, col_bl2 = st.columns(2)
                col_bl1.metric("Missed Leavers", f"{missed_pct:.1f}%", help="Of all people who left, what % did the AI not predict?")
                col_bl2.metric("Hidden Risks", f"{surprise_pct:.1f}%", help="Of all people who stayed, what % are actually high-risk?")
                
                if missed_pct > 20:
                    st.error("**Your data has a blind spot.** Too many people are leaving for reasons the AI can't see. Start doing stay interviews to understand what's missing from your data.")
                elif missed_pct > 10:
                    st.warning("**Some blind spots exist.** A few people slipped through. Review their exit interviews to find the pattern.")
                else:
                    st.success("**Your data captures the real reasons well.** The AI catches most leavers. Keep it up.")
                
                if surprise_pct > 15:
                    st.error("**You have a large silent risk group.** Many unhappy people are hiding in plain sight. Prioritize 1-on-1s with these employees before the market improves.")
                elif surprise_pct > 5:
                    st.warning("**A small group needs attention.** Not urgent, but worth having conversations with these people sooner rather than later.")
                else:
                    st.success("**No silent risk group.** People who look risky actually leave. The model is honest.")
            else:
                st.markdown("---")
                st.balloons()
                st.success("🎯 **Perfect score!** The AI predicted every single person correctly. No blind spots found.")

        with tab4:
            st.subheader("📊 Retention Priority Matrix")
            st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>A strategic view to prioritize your HR efforts. <br>We map <strong>Attrition Risk</strong> against <strong>Replacement Cost</strong> to identify who needs immediate attention vs. who is safe to let go.</p>", unsafe_allow_html=True)
            X_all = df.drop('left', axis=1)
            
            raw_risk_probs = pipeline.predict_proba(X_all)[:, 1]
            risk_probs = calibrate_probability_array(raw_risk_probs, temperature=0.55)
            
            if 'salary' in df.columns:
                salary_cost_map = {'low': 400000, 'medium': 600000, 'high': 900000} 
                replacement_costs = df['salary'].map(salary_cost_map) * 0.5 
            else:
                replacement_costs = pd.Series([500000]*len(df))
            plot_data = pd.DataFrame({'Employee_ID': df.index, 'Risk_Probability': risk_probs, 'Replacement_Cost': replacement_costs})
            if 'Department' in df.columns: plot_data['Department'] = df['Department']
            risk_threshold = 0.5; cost_threshold = replacement_costs.median()
            def get_zone(row):
                if row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] >= cost_threshold: return "🔴 Critical Zone (Save Now)"
                elif row['Risk_Probability'] < risk_threshold and row['Replacement_Cost'] >= cost_threshold: return "🟡 Retain Zone (Keep Happy)"
                elif row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] < cost_threshold: return "🟢 Outplacement Zone (Let Go)"
                else: return "⚪ Monitor Zone"
            plot_data['Zone'] = plot_data.apply(get_zone, axis=1)
            hover_data = ['Department'] if 'Department' in plot_data.columns else None
            fig = px.scatter(plot_data, x='Risk_Probability', y='Replacement_Cost', color='Zone', color_discrete_map={"🔴 Critical Zone (Save Now)": "#FF4B4B", "🟡 Retain Zone (Keep Happy)": "#F59E0B", "🟢 Outplacement Zone (Let Go)": "#17B794", "⚪ Monitor Zone": "#9ca3af"}, hover_data=hover_data, title="Employee Prioritization Map", template="plotly_dark", labels={'Risk_Probability': 'Predicted Attrition Risk', 'Replacement_Cost': 'Est. Replacement Cost (₹)'}, height=600)
            fig.add_hline(y=cost_threshold, line_dash="dash", line_color="white", opacity=0.3); fig.add_vline(x=risk_threshold, line_dash="dash", line_color="white", opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---"); col_z1, col_z2, col_z3 = st.columns(3)
            critical_count = len(plot_data[plot_data['Zone'] == "🔴 Critical Zone (Save Now)"]); outplace_count = len(plot_data[plot_data['Zone'] == "🟢 Outplacement Zone (Let Go)"])
            col_z1.metric("Critical Interventions Needed", critical_count, delta="High Priority", delta_color="inverse"); col_z2.metric("Potential Efficiency Savings", outplace_count, delta="Safe to Exit", delta_color="normal"); col_z3.metric("Average Replacement Cost", f"₹{int(cost_threshold):,}")

        with tab5:
            st.subheader("🎯 The 'Ideal Candidate' Profiler")
            st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Shift from Retention to <strong>Acquisition</strong>. <br>We analyze your 'Superstars' (Loyal + High Performers) to build a clear hiring checklist.</p>", unsafe_allow_html=True)
            if 'time_spend_company' in df.columns and 'last_evaluation' in df.columns:
                superstar_mask = (df['left'] == 0) & (df['time_spend_company'] > 4) & (df['last_evaluation'] > 0.8)
            else:
                st.info("Dataset missing specific tenure/evaluation columns. Defining Superstars simply as top retained performers.")
                superstar_mask = (df['left'] == 0)
                
            df_superstars = df[superstar_mask]; df_average = df[(df['left'] == 0) & (~superstar_mask)]
            if len(df_superstars) < 5: st.warning("Not enough 'Superstar' data in this dataset to generate a reliable profile.")
            else:
                st.success(f"Analyzed {len(df_superstars)} Superstars vs {len(df_average)} Average Employees.")
                metrics_to_compare = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                super_mean = df_superstars[metrics_to_compare].mean(); avg_mean = df_average[metrics_to_compare].mean()
                comparison_long = pd.DataFrame({'Metric': metrics_to_compare, 'Superstar': super_mean.values, 'Average Employee': avg_mean.values}).melt(id_vars='Metric', var_name='Group', value_name='Average Value')
                fig_compare = px.bar(comparison_long, x='Metric', y='Average Value', color='Group', barmode='group', title="Superstars vs. Average Employees (Head-to-Head)", template="plotly_dark", color_discrete_map={'Superstar': '#17B794', 'Average Employee': '#9ca3af'}, text_auto=True, height=500)
                fig_compare.update_xaxes(title="", tickangle=45); fig_compare.update_layout(yaxis_title="Average Score / Value"); st.plotly_chart(fig_compare, use_container_width=True)
                st.markdown("### 🧬 The DNA of a Top Performer"); st.write("Here are the 3 biggest differentiators between your best employees and the rest.")
                diff_df = pd.DataFrame({'Metric': metrics_to_compare, 'Difference': (super_mean - avg_mean).values})
                diff_df['Abs_Diff'] = diff_df['Difference'].abs(); top_3_diff = diff_df.nlargest(3, 'Abs_Diff')
                col_dna1, col_dna2, col_dna3 = st.columns(3)
                def get_dna_insight(metric, diff_val):
                    metric_name = metric.replace('_', ' ').title()
                    if 'satisfaction' in metric.lower():
                        if diff_val > 0: return "🟢 High Culture Fit", f"Superstars are {diff_val:.2f} points happier on average."
                        else: return "🔴 Low Satisfaction", f"Unexpected: Superstars seem less satisfied."
                    elif 'hours' in metric.lower():
                        if diff_val < 0: return "🟢 Work-Life Balance", f"Superstars work {abs(diff_val):.0f} hrs LESS."
                        else: return "🔴 High Workload", f"Superstars work harder."
                    elif 'evaluation' in metric.lower():
                        if diff_val > 0: return "🟢 High Performance", f"Superstars score {diff_val:.2f} points higher."
                        else: return "🔴 Low Performance", f"Superstars score lower."
                    else: return "📊 " + metric_name, f"Difference of {diff_val:.2f}."
                cols = [col_dna1, col_dna2, col_dna3]
                for i, col in enumerate(cols):
                    if i < len(top_3_diff):
                        row = top_3_diff.iloc[i]; title, text = get_dna_insight(row['Metric'], row['Difference'])
                        st.markdown(f"<div class='custom-card' style='text-align: center; border-top: 4px solid #17B794;'><h3 style='margin-top: 0;'>{title}</h3><p style='color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;'>{text}</p></div>", unsafe_allow_html=True)
                st.markdown("---"); st.markdown("### 📝 Hiring Checklist (Do's & Don'ts)"); st.write("Based on the data, apply these filters to your next job opening:")
                checklist = []
                
                top_3_features = top_3_diff['Metric'].tolist()
                
                for _, row in top_3_diff.iterrows():
                    feature = row['Metric']
                    diff_val = row['Difference']
                    feature_lower = feature.lower()
                    
                    if 'satisfaction' in feature_lower:
                        if diff_val > 0.05:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"Candidates who value 'Culture', 'Team', and 'Purpose' (Superstars are {diff_val:.2f} pts more satisfied)"})
                        elif diff_val < -0.05:
                            checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who seem disconnected from company culture or overly focused only on perks."})
                        else:
                            checklist.append({"Type": "ℹ️ NOTE", "Rule": "Satisfaction is not a major differentiator here. Don't over-prioritize 'culture fit' questions."})
                    
                    elif 'hour' in feature_lower or 'time_spend' in feature_lower:
                        if diff_val < -5:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"Candidates who demonstrate 'Work-Life Balance' and set healthy boundaries (Superstars work {abs(diff_val):.0f} hrs less)"})
                            checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who brag about 'sleeping at the office' or working 80-hour weeks consistently."})
                        elif diff_val > 5:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"High-energy candidates willing to put in extra hours when the project demands it."})
                            checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who strictly clock out at 5 PM regardless of deadlines or team needs."})
                    
                    elif 'project' in feature_lower:
                        if diff_val > 0.3:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"Candidates who can comfortably handle {int(diff_val)}+ concurrent projects without burning out."})
                        elif diff_val < -0.3:
                            checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who prefer focused, deep work on fewer projects rather than juggling many."})
                    
                    elif 'evaluation' in feature_lower:
                        if diff_val > 0.05:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"Candidates with a proven track record of exceeding targets (Superstars score {diff_val:.2f} pts higher)"})
                        elif diff_val < -0.05:
                            checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who seem overconfident but lack measurable achievements to back it up."})
                    
                    elif 'accident' in feature_lower:
                        if diff_val < -0.02:
                            checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who emphasize safety protocols, process compliance, and risk awareness."})
                    
                    elif 'promotion' in feature_lower:
                        if diff_val > 0.02:
                            checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who show a growth mindset, ambition, and interest in long-term career progression."})
                        elif diff_val < -0.02:
                            checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who seem entitled or expect rapid promotions without putting in the effort."})
                    
                    elif 'salary' in feature_lower or 'wage' in feature_lower:
                        if diff_val > 0:
                            checklist.append({"Type": "⚠️ NOTE", "Rule": "Superstars tend to be in higher salary bands — ensure your offer is competitive for top talent."})
                        else:
                            checklist.append({"Type": "ℹ️ NOTE", "Rule": "Pay parity exists among top performers. Focus on non-monetary benefits in your pitch."})
                    
                    elif 'tenure' in feature_lower or 'spend_company' in feature_lower:
                        if diff_val > 0.5:
                            checklist.append({"Type": "✅ DO Look For", "Rule": f"Candidates with longer tenures at previous companies ({diff_val:.1f}+ years avg) — shows loyalty and resilience."})
                        elif diff_val < -0.5:
                            checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who bring fresh perspectives from diverse experiences, even if tenures are shorter."})
                    
                    else:
                        metric_clean = feature.replace('_', ' ').title()
                        if diff_val > 0:
                            checklist.append({"Type": "✅ CONSIDER", "Rule": f"Higher '{metric_clean}' scores may correlate with top performance. Probe this area in interviews."})
                        else:
                            checklist.append({"Type": "✅ CONSIDER", "Rule": f"Lower '{metric_clean}' scores may correlate with top performance. Don't assume higher is always better."})
                
                if len(checklist) == 0:
                    checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who demonstrate stability in previous roles (2+ years per company on average)."})
                    checklist.append({"Type": "✅ DO Look For", "Rule": "Candidates who ask thoughtful questions about company culture, team dynamics, and growth opportunities."})
                    checklist.append({"Type": "🚫 AVOID", "Rule": "Candidates who focus exclusively on salary and title, with no interest in the role itself or the team."})
                
                for item in checklist:
                    if "DO" in item['Type']: 
                        st.success(f"**{item['Type']}**: {item['Rule']}")
                    elif "AVOID" in item['Type']: 
                        st.error(f"**{item['Type']}**: {item['Rule']}")
                    elif "NOTE" in item['Type']: 
                        st.warning(f"**{item['Type']}**: {item['Rule']}")
                    elif "CONSIDER" in item['Type']: 
                        st.info(f"**{item['Type']}**: {item['Rule']}")
                    else: 
                        st.info(f"**{item['Type']}**: {item['Rule']}")

    # ====================================================================
    # Page: STRATEGIC ROADMAP
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Future Planning & Projections")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>A simple tool to show leadership exactly what happens if we take action vs. if we do nothing.</p>", unsafe_allow_html=True)
        
        st.markdown("### 📋 Step 1: Get Your 6-Month Action Plan")
        avg_sat = df['satisfaction_level'].mean() if 'satisfaction_level' in df.columns else 0.5
        avg_hours = df['average_montly_hours'].mean() if 'average_montly_hours' in df.columns else 0
        avg_projects = df['number_project'].mean() if 'number_project' in df.columns else 0
        issues = []
        if avg_sat < 0.6: issues.append("Low Employee Satisfaction")
        if avg_hours > 200: issues.append("Employee Burnout (High Working Hours)")
        if avg_projects > 4: issues.append("Unbalanced Workload (Too Many Projects)")
        if len(issues) == 0: issues.append("Standard Workforce Stabilization")
        issues_str = ", ".join(issues)
        
        st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: #17B794; margin-top: 0;">🩺 AI Diagnostic Summary</h4>
            <p style="color: #c9d1d9; line-height: 1.6;">
                Before making a plan, here is what the AI flagged as your biggest risks:<br>
                <strong style="color: #EEB76B;">➤ {issues_str}</strong>
            </p>
            <small style="color: #8b949e;">Click the button below to generate a custom 6-month HR strategy to fix these exact issues.</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✍️ Draft My 6-Month HR Action Plan", type="primary"):
            with st.spinner("Drafting your 6-month strategy..."):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if api_key:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                        template = """You are an expert HR Strategist speaking to an HR Manager.\n**Company Context:**\nOur AI identified these attrition drivers: {issues}.\nWe need to stabilize the workforce over 6 months.\n**Task:** Create a 6-month execution roadmap. Break it into phases. For each month give: 1. Phase Name, 2. Actionable Steps (2-3 bullets), 3. Success Metrics.\n**Tone:** Plain English. Practical HR actions (e.g. "Run stay interviews"). Avoid technical AI jargon."""
                        prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({"issues": issues_str})
                        st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("🔑 API Key missing. Showing generic template.")
                        st.markdown("**Month 1: Diagnosis & Audit**\n* Conduct stay interviews with top 10% at-risk employees.\n* *Metric:* Complete 100% of risk interviews.\n\n**Month 3: Pilot Launch**\n* Launch intervention in one high-risk department.\n* *Metric:* Pilot participation rate > 80%.\n\n**Month 6: Review**\n* Measure impact on satisfaction.\n* *Metric:* 5% reduction in projected attrition.")
                except Exception as e: st.error(f"Error: {e}")

        st.markdown("---")
        
        st.markdown("### 📈 Step 2: See the Future Impact (12-Month Projection)")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Adjust the sliders to match your realistic expectations. This calculates the exact headcount and money saved.</p>", unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            intervention_efficacy = st.slider("If we take action, how many at-risk people will we actually save? (%)", 10, 50, 20, 5, help="If 100 people are at risk, and you set this to 20%, you expect to successfully retain 20 people.")
        with col_f2:
            natural_attrition_rate = st.slider("People who leave for personal reasons (Spouse relocation, etc.) (%)", 0.5, 2.0, 1.0, 0.1, help="People leave for reasons no HR plan can fix. This is that baseline %.")

        if st.button("📈 Show Me the 12-Month Projection", type="primary"):
            months = list(range(1, 13)); current_workforce = len(df)
            
            raw_risk_scores = pipeline.predict_proba(df.drop('left', axis=1))[:, 1]
            total_risk_score = calibrate_probability_array(raw_risk_scores, temperature=0.55).sum()
            
            monthly_leavers_no_action = total_risk_score / 12.0
            monthly_leavers_with_action = monthly_leavers_no_action * (1 - (intervention_efficacy / 100.0))
            forecast_bau = []; forecast_intervention = []; temp_bau = float(current_workforce); temp_int = float(current_workforce)
            for m in months:
                natural_leavers_bau = temp_bau * (natural_attrition_rate / 100.0); natural_leavers_int = temp_int * (natural_attrition_rate / 100.0)
                total_leavers_bau = monthly_leavers_no_action + natural_leavers_bau; total_leavers_int = monthly_leavers_with_action + natural_leavers_int
                temp_bau = max(0, temp_bau - total_leavers_bau); temp_int = max(0, temp_int - total_leavers_int)
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
            st.success(f"**The Bottom Line:** If we execute our 6-month plan and successfully retain just **{intervention_efficacy}%** of our at-risk staff, we will finish the year with **{int(forecast_intervention[-1])} employees** instead of **{int(forecast_bau[-1])}**. This prevents approximately **₹{total_money_saved:,.0f}** in recruitment, onboarding, and lost productivity costs.")

if __name__ == "__main__":
    main()
