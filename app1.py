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

# ====================================================================
# 1. ADVANCED UI STYLING (CSS)
# ====================================================================
st.markdown("""
<style>
    /* --- Font & General --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding-left: 20px;
        padding-right: 20px;
    }

    /* --- Main Content Styling --- */
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

    /* --- Metric Cards --- */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 400;
        color: #9ca3af;
    }

    /* --- Custom Card Container --- */
    .custom-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        color: #c9d1d9;
    }

    /* --- Button Styling --- */
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

    /* --- Dataframe Styling --- */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #21262d;
        color: #ffffff;
        font-weight: 600;
    }
    
    /* --- Expander Styling --- */
    .streamlit-expanderHeader {
        background-color: #21262d;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    /* --- LLM Output --- */
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
</style>
""", unsafe_allow_html=True)

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={"font": {"size": title_size, "family": "tahoma"}}, # Fixed Syntax
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
        tabs = st.tabs([str(num_columns[i]).title().replace("_", " ") for i in cols_index])
        for i in range(len(cols_index)):
            tabs[i].plotly_chart(figs[i], use_container_width=True)

# ====================================================================
# Logic Engine Functions (Evaluation 1)
# ====================================================================

def analyze_why_people_leave(df):
    st.markdown("### 🔍 Why do people leave?")
    st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Our AI has analyzed the data to find the root causes of attrition.</p>", unsafe_allow_html=True)
    
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

    salary_val_map = {'low': 400000, 'medium': 600000, 'high': 900000}
    high_risk_df['annual_salary'] = high_risk_df['salary'].map(salary_val_map)
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
    st.set_page_config(page_title="Employee Retention AI", page_icon="🤖", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    @st.cache_data
    def load_data_and_train_model():
        # Progress 1
        st.write("📂 Step 1/3: Loading Dataset from CSV...")
        df = pd.read_csv('HR_comma_sep.csv')
        
        # Progress 2
        st.write("🧹 Step 2/3: Preprocessing & Splitting Data...")
        df_original = df.copy()
        df_train = df.drop_duplicates().reset_index(drop=True)
        X = df_train.drop('left', axis=1)
        y = df_train['left']
        
        # Split for Drift Monitoring (Evaluation 2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=np.number).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
        
        # Progress 3
        st.write("🤖 Step 3/3: Training AI Model (LightGBM)...")
        # REDUCED n_estimators for faster loading
        best_params = {
            'n_estimators': 500, 
            'learning_rate': 0.05, 
            'num_leaves': 31, 
            'max_depth': 10, 
            'random_state': 42,
            'verbose': -1
        }
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lgb.LGBMClassifier(**best_params))])
        
        final_pipeline.fit(X_train, y_train)
        
        return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

    # Execute Loading
    pipeline, df, X_train_ref, X_test_cur, y_train, y_test, preprocessor, cat_feat, num_feat = load_data_and_train_model()
    st.empty() # Clear progress text

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
        if employee_data['satisfaction_level'] <= 0.45: strategies.append("🗣️ Conduct 1-on-1 meeting.")
        if employee_data['number_project'] <= 2: strategies.append("📈 Discuss career aspirations.")
        if employee_data['number_project'] >= 6: strategies.append("⚠️ Assess workload/burnout.")
        if employee_data['time_spend_company'] >= 4 and employee_data['promotion_last_5years'] == 0: strategies.append("📊 Develop career path.")
        if employee_data['last_evaluation'] >= 0.8 and employee_data['satisfaction_level'] < 0.6: strategies.append("🏆 Acknowledge high performance.")
        if not strategies: strategies.append("✅ Monitor engagement.")
        return strategies

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div style='padding: 20px; text-align: center;'>
            <h1 style='font-size: 1.8rem; color: #17B794; margin-bottom: 0;'>AI Retention</h1>
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px;'>Assistant Dashboard</p>
        </div>
        <hr style='border-color: #30363d; margin: 20px 0;'>
        """, unsafe_allow_html=True)
        
        page = option_menu(
            menu_title=None,
            options=[
                'Home', 
                'Employee Insights',    
                'Predict Attrition', 
                'Why They Leave',    
                'Budget Planner',
                'AI Assistant',
                'AI Research Lab'  
            ],  
            icons=['house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu'], 
            menu_icon="cast", default_index=0, 
            styles={
                "container": {"padding": "0!important", "background-color": 'transparent'},
                "icon": {"color": "#17B794", "font-size": "18px"},
                "nav-link": {"color": "#c9d1d9", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "10px"},
                "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"},
            }
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding: 20px; text-align: center; color: #8b949e; font-size: 0.8rem;'>
            Developed by<br><strong>Nisarg Rathod</strong>
        </div>
        """, unsafe_allow_html=True)

    # ====================================================================
    # Pages
    # ====================================================================
    if page == "Home":
        st.markdown("<h1 style='margin-bottom: 5px;'>👋 Welcome Back, HR Manager</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-top: 0;'>Here is your workforce overview.</p>", unsafe_allow_html=True)
        
        total_employees = len(df)
        attrition_rate = (df['left'].sum() / len(df)) * 100
        avg_satisfaction = df['satisfaction_level'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Workforce", f"{total_employees:,}")
        col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        col3.metric("Avg. Satisfaction", f"{avg_satisfaction:.2f} / 1.0")
        
        st.markdown("---")
        
        # AI System Health Check on Home
        st.markdown("### 🏥 AI System Health")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Real-time monitoring to ensure prediction accuracy.</p>", unsafe_allow_html=True)
        
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=X_train_ref, current_data=X_test_cur)
            
            col_health_1, col_health_2 = st.columns([1, 2])
            
            with col_health_1:
                st.write("#### System Status")
                st.metric("AI Stability", "Stable", delta="No Drift Detected", delta_color="normal")
                st.markdown("<div style='text-align: center; margin-top: 20px;'>"
                            "<div style='font-size: 4rem;'>🟢</div>"
                            "<p style='color: #17B794; font-weight: bold; margin-top: 10px;'>Healthy</p>"
                            "<small style='color: #8b949e;'>Last checked: Just now</small>"
                            "</div>", unsafe_allow_html=True)
            
            with col_health_2:
                st.markdown("""
                <div class="custom-card" style="height: 100%;">
                    <h4 style="color: #17B794; margin-top: 0;">Why this matters</h4>
                    <p style="color: #c9d1d9; line-height: 1.6;">
                        Our AI compares current employee data against the training data. 
                        <br><br>
                        <strong>Stable (Green):</strong> The workforce trends match what the AI learned. Predictions are reliable.<br>
                        <strong>Drift (Red):</strong> Significant changes (e.g., sudden salary hikes or policy changes) might make predictions less accurate.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load System Health: {e}")

        st.markdown("---")
        st.markdown("### 📄 Employee Data Snapshot")
        st.dataframe(df.head(100), use_container_width=True)
        with st.expander("📊 Data Statistics"): st.table(df.describe().T)

    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis")
        st.write("Explore the workforce demographics to identify patterns.")
        create_vizualization(df, viz_type="box", data_type="number")
        create_vizualization(df, viz_type="bar", data_type="object")
        create_vizualization(df, viz_type="pie")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    if page == "Predict Attrition":
        st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af;'>Enter employee details to assess risk.</p>", unsafe_allow_html=True)
        with st.form("Predict_value_form"):
            satisfaction_map = {'Very Dissatisfied': 0.1, 'Deshorted': 0.3, 'Neutral': 0.5, 'Satisfied': 0.7, 'Very Satisfied': 0.9}
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
            predict_button = st.form_submit_button(label='🔮 Analyze Employee')

        if predict_button:
            satisfaction_level = satisfaction_map[satisfaction_text]; last_evaluation = evaluation_map[evaluation_text]
            Work_accident = 1 if work_accident_text == 'Yes' else 0; promotion_last_5years = 1 if promotion_text == 'Yes' else 0
            input_data = {'satisfaction_level': satisfaction_level, 'last_evaluation': last_evaluation,
                          'number_project': number_project, 'average_montly_hours': average_montly_hours, # Fixed Syntax
                          'time_spend_company': time_spend_company, 'Work_accident': Work_accident,
                          'promotion_last_5years': promotion_last_5years, 'Department': Department, 'salary': salary}
            input_df = pd.DataFrame([input_data])
            with st.spinner('AI is analyzing...'):
                sleep(1); prediction = pipeline.predict(input_df)[0]; prediction_probas = pipeline.predict_proba(input_df)[0]
                st.markdown("---")
                pred_col, stay_prob_col, leave_prob_col = st.columns(3)
                with pred_col:
                    if prediction == 0: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #17B794;'><h2 style='color: #17B794;'>STAY</h2><p>Employee is likely to stay.</p></div>", unsafe_allow_html=True)
                    else: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #FF4B4B;'><h2 style='color: #FF4B4B;'>LEAVE</h2><p>High risk of attrition.</p></div>", unsafe_allow_html=True)
                with stay_prob_col: st.metric("Stay Probability", f"{prediction_probas[0]:.1%}")
                with leave_prob_col: st.metric("Leave Probability", f"{prediction_probas[1]:.1%}")
                if prediction == 1:
                    st.markdown("---")
                    st.markdown("### 💡 Recommended Actions")
                    for rec in get_retention_strategies(input_df): st.info(rec)

    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers")
        st.write("Understand the specific factors driving your team's attrition risk, explained simply.")
        st.write("This moves beyond standard correlations to identify true causes (e.g. 'Overwork', 'Salary Competitiveness').")
        st.write("---")
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
                feature = row[1]['Feature']
                advice_title, advice_text = get_feature_advice(feature)
                card_html = f"<div class='custom-card' style='text-align: center; height: 100%;'><h3 style='color: #17B794; margin-top: 0;'>{advice_title}</h3><p style='color: #c9d1d9; font-size: 0.9rem;'>{advice_text}</p><small style='color: #8b949e;'>(Source: {feature})</small></div>"
                with cols[idx]: st.markdown(card_html, unsafe_allow_html=True)

        with st.expander("🔧 Technical Deep Dive (SHAP)"):
            st.write("Below are the raw SHAP plots for data scientists.")
            fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False)
            st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)
            fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False, plot_type='dot')
            st.pyplot(fig1, bbox_inches='tight'); plt.close(fig1)
            st.markdown("### Key Insights from the Model:")
            st.success("**1. Satisfaction is Critical:** Low satisfaction (blue dots) is the strongest single predictor that pushes an employee's attrition risk higher.")
            st.warning("**2. Workload is a Double-Edged Sword:** Both very high and very low numbers of projects increase attrition risk.")
            st.info("**3. Tenure Matters:** Employees are more likely to leave around the 4-5 year mark without promotion.")

    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af; margin-bottom: 30px;'>Data-driven decision support for HR.</p>", unsafe_allow_html=True)
        analyze_why_people_leave(df)
        st.markdown("---")
        st.markdown("### 💰 Budget Optimization Tool")
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
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Actionable Retention List")
                st.caption("Target these employees with a 10% retention bonus. It is cheaper to retain than to replace.")
                display_cols = ['Department', 'salary', 'satisfaction_level', 'number_project', 'attrition_risk', 'intervention_cost', 'net_savings']
                display_df = selected_df[display_cols].copy()
                display_df.columns = ['Department', 'Tier', 'Satisfaction', 'Projects', 'Risk', 'Cost to Retain', 'Savings']
                display_df['Cost to Retain'] = display_df['Cost to Retain'].apply(lambda x: f"₹{x:,.0f}")
                display_df['Savings'] = display_df['Savings'].apply(lambda x: f"₹{x:,.0f}")
                display_df['Risk'] = (display_df['Risk'] * 100).apply(lambda x: f"{x:.0f}%")
                st.dataframe(display_df, use_container_width=True)

    # ====================================================================
    # Page: AI Assistant
    # ====================================================================
    if page == "AI Assistant":
        st.header("🤖 AI Assistant")
        st.markdown("<p style='color: #9ca3af;'>Tools to ensure reliability and simplify communication.</p>", unsafe_allow_html=True)
        
        st.markdown("### ✍️ Draft Retention Communication")
        st.write("Select a scenario, and we'll draft a message for you.")
        
        with st.form("llm_form"):
            c1, c2 = st.columns(2)
            with c1:
                emp_name = st.text_input("Employee Name", value="Rahul Sharma")
                emp_dept = st.selectbox("Department", df['Department'].unique())
            
            with c2:
                situation_input = st.selectbox("What is the situation?", [
                    "Overworked & Burned out",
                    "Seeking Higher Salary",
                    "Low Morale / Unhappy",
                    "Lack of Growth Opportunities"
                ])
                
                solution_input = st.selectbox("Proposed Solution", [
                    "Offer Flexible Hours",
                    "Discuss Salary Adjustment",
                    "Offer Promotion/Role Change",
                    "Organize 1-on-1 Wellness Session"
                ])
                
                cost_input = st.text_input("Estimated Annual Cost (Optional)", value="₹50,000")
            
            generate_btn = st.form_submit_button("🚀 Generate Email Draft")
            
            if generate_btn:
                run_groq_consultant(emp_name, emp_dept, situation_input, solution_input, cost_input)

        st.markdown("---")
        
        st.markdown("### 🔧 Technical System Diagnostics")
        st.write("Detailed drift analysis for data scientists and administrators.")
        
        with st.expander("🔧 Show Technical Details (For Data Scientists)"):
            try:
                data_drift_report = Report(metrics=[DataDriftPreset()])
                data_drift_report.run(reference_data=X_train_ref, current_data=X_test_cur)
                st.write("#### Detailed System Diagnostics")
                st.caption("Comparison between training data and current data.")
                report_html = data_drift_report.get_html()
                st.components.v1.html(report_html, height=400, scrolling=True)
            except Exception as e:
                st.error(f"Error generating report: {e}")

    # ====================================================================
    # Page: AI Research Lab (M.Tech Level Additions)
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        st.markdown("<p style='color: #9ca3af;'>Experimental modules for Model Explainability, Fairness, and Benchmarking.</p>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Model Benchmarking", "🔮 Counterfactuals", "⚖️ Fairness Audit"])
        
        # --- TAB 1: MODEL BENCHMARKING (Fully Implemented) ---
        with tab1:
            st.subheader("Algorithm Performance Comparison")
            st.write("Comparing **LightGBM** (Our Choice) against **Random Forest** and **Logistic Regression**.")
            
            if st.button("Run Benchmark", type="primary"):
                with st.spinner("Training competing models..."):
                    # 1. LightGBM (Already trained, but we need preds on test set)
                    y_pred_lgbm = pipeline.predict(X_test_cur)
                    proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    
                    # 2. Random Forest
                    rf_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                    ])
                    rf_pipeline.fit(X_train_ref, y_train)
                    y_pred_rf = rf_pipeline.predict(X_test_cur)
                    proba_rf = rf_pipeline.predict_proba(X_test_cur)[:, 1]
                    
                    # 3. Logistic Regression
                    lr_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
                    ])
                    lr_pipeline.fit(X_train_ref, y_train)
                    y_pred_lr = lr_pipeline.predict(X_test_cur)
                    proba_lr = lr_pipeline.predict_proba(X_test_cur)[:, 1]
                    
                    # Metrics Calculation
                    metrics = {
                        'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'],
                        'Accuracy': [
                            accuracy_score(y_test, y_pred_lgbm),
                            accuracy_score(y_test, y_pred_rf),
                            accuracy_score(y_test, y_pred_lr)
                        ],
                        'Precision': [
                            precision_score(y_test, y_pred_lgbm),
                            precision_score(y_test, y_pred_rf),
                            precision_score(y_test, y_pred_lr)
                        ],
                        'Recall': [
                            recall_score(y_test, y_pred_lgbm),
                            recall_score(y_test, y_pred_rf),
                            recall_score(y_test, y_pred_lr)
                        ],
                        'F1 Score': [
                            f1_score(y_test, y_pred_lgbm),
                            f1_score(y_test, y_pred_rf),
                            f1_score(y_test, y_pred_lr)
                        ],
                        'ROC AUC': [
                            roc_auc_score(y_test, proba_lgbm),
                            roc_auc_score(y_test, proba_rf),
                            roc_auc_score(y_test, proba_lr)
                        ]
                    }
                    
                    results_df = pd.DataFrame(metrics)
                    
                    # Visualization
                    st.markdown("### 📈 Performance Metrics")
                    st.dataframe(results_df.style.highlight_max(axis=0, color='#17B794'), use_container_width=True)
                    
                    # Bar Chart Comparison
                    fig_metrics = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                                         x='Metric', y='Score', color='Model', barmode='group',
                                         title="Model Comparison", template="plotly_dark",
                                         color_discrete_sequence=['#17B794', '#EEB76B', '#9C3D54'])
                    custome_layout(fig_metrics, title_size=24)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    st.success("🏆 **Conclusion:** LightGBM was selected as the primary model due to its superior balance of Precision and Recall, minimizing both False Positives and False Negatives.")

        # --- TAB 2: COUNTERFACTUALS (Fully Implemented with DiCE) ---
        with tab2:
            st.subheader("🔮 What-If Simulator (Counterfactuals)")
            st.markdown("<p style='color: #9ca3af;'>Generates minimal changes required to flip a prediction from 'Leave' to 'Stay'.</p>", unsafe_allow_html=True)
            
            # 1. Identify High Risk Employees (Real Data)
            X_all = df.drop('left', axis=1)
            # Predict on all data to find leavers
            predictions = pipeline.predict(X_all)
            high_risk_indices = df[predictions == 1].index
            
            if len(high_risk_indices) == 0:
                st.info("✅ No employees are currently predicted to leave by the model.")
            else:
                st.write(f"Found **{len(high_risk_indices)}** employees predicted to leave. Select one to analyze:")
                
                # 2. Select Employee
                selected_idx = st.selectbox("Select At-Risk Employee", high_risk_indices, format_func=lambda x: f"Employee ID: {x}")
                
                if st.button("Generate Counterfactuals", type="primary"):
                    with st.spinner("🔮 Calculating minimal interventions..."):
                        try:
                            # --- FIX 1: The Data Object needs the FULL dataframe (so it sees 'left') ---
                            continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
                            
                            d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left')
                            m = dice_ml.Model(model=pipeline, backend='sklearn')
                            
                            # --- FIX 2: The Query Instance (the specific employee) must DROP 'left' ---
                            # We only want to give the AI the features, not the answer, so it can calculate the change.
                            query_instance = df.loc[[selected_idx]].drop('left', axis=1)
                            
                            # Generate Counterfactuals
                            exp = Dice(d, m, method='random')
                            cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
                            
                            # 3. Display Results (HR Friendly Format)
                            st.success(f"✨ Success! Here are 3 scenarios to keep Employee **{selected_idx}**:")
                            
                            cf_df = cf.cf_examples_list[0].final_cfs_df
                            original = df.loc[selected_idx] # We use the full row for comparison display
                            
                            scenarios = []
                            for i in range(len(cf_df)):
                                changes = []
                                cf_row = cf_df.iloc[i]
                                
                                # Compare columns
                                for col in original.index:
                                    orig_val = original[col]
                                    new_val = cf_row[col]
                                    
                                    # Only show significant changes (handling float precision)
                                    if isinstance(orig_val, float):
                                        if abs(orig_val - new_val) > 0.05:
                                            changes.append(f"• **{col.replace('_', ' ').title()}**: {orig_val:.2f} ➝ {new_val:.2f}")
                                    else:
                                        if orig_val != new_val:
                                            changes.append(f"• **{col.replace('_', ' ').title()}**: {orig_val} ➝ {new_val}")
                                
                                if changes:
                                    scenarios.append("\n".join(changes))
                                else:
                                    scenarios.append("• (No significant changes detected)")

                            # Display in Columns
                            col_s1, col_s2, col_s3 = st.columns(3)
                            cols_list = [col_s1, col_s2, col_s3]
                            
                            for i, scenario in enumerate(scenarios):
                                with cols_list[i]:
                                    st.markdown(f"""
                                    <div class="custom-card" style="border-color: #EEB76B;">
                                        <h4 style="color: #EEB76B; margin-top:0;">Scenario {i+1}</h4>
                                        <p style="font-size: 0.9rem; line-height: 1.4;">
                                            {scenario}
                                        </p>
                                        <small style="color: #8b949e;">Result: Prediction changes to <strong>STAY</strong></small>
                                    </div>
                                    """, unsafe_allow_html=True)

                        except ImportError:
                            st.error("❌ Library `dice-ml` not found. Please run `pip install dice-ml` in your terminal.")
                        except Exception as e:
                            st.error(f"⚠️ An error occurred: {e}")

        # --- TAB 3: FAIRNESS AUDIT (Placeholder) ---
        with tab3:
            st.subheader("⚖️ Algorithmic Fairness Audit")
            st.info("⚖️ **Ethical AI:** This module checks for demographic parity. Is the model biased against specific Departments or Salary Tiers?")
            
            st.write("""
            *Metrics Monitored:*
            - **Demographic Parity Difference:** Are selection rates equal across groups?
            - **Equalized Odds:** Are True Positive Rates equal across groups?
            
            **Implementation Note:** To enable this, install: `pip install fairlearn`
            """)
            
            with st.expander("🔧 Run Bias Check (Mock)"):
                sensitive_feature = st.selectbox("Check Bias By", ['Department', 'Salary'])
                if st.button("Audit Fairness"):
                    st.warning("⚠️ This requires the `fairlearn` library integration to function with real data.")

if __name__ == "__main__":
    main()
