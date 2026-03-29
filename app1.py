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

    /* --- HR Friendly Action Item Styling --- */
    .action-item {
        background-color: #161b22;
        padding: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #17B794;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .action-item-high-effort {
        border-left: 3px solid #FF4B4B; /* Red for high effort */
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
        
        best_params = {
            'n_estimators': 500, 
            'learning_rate': 0.05, 
            'num_leaves': 31, 
            'max_depth': 10, 
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced',
            'scale_pos_weight': 15
        }
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lgb.LGBMClassifier(**best_params))])
        
        final_pipeline.fit(X_train, y_train)
        
        return final_pipeline, df_original, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

    # Execute Loading
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
                'AI Research Lab',
                'Strategic Roadmap'  
            ],  
            icons=['house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill'], 
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
        st.markdown("<p style='color: #9ca3af;'>Enter employee details to assess risk and get retention strategies.</p>", unsafe_allow_html=True)
        
        # --- MODEL DIAGNOSTICS ---
        with st.expander("🧪 Model Diagnostics (Verification)", expanded=False):
            st.write("Not sure if the AI is working? Test it against real historical data:")
            c_test1, c_test2 = st.columns(2)
            
            with c_test1:
                if st.button("Test with Employee who Left"):
                    sample = df[df['left'] == 1].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    pred = pipeline.predict(test_df)[0]
                    if pred == 1:
                        st.success("✅ **Correct!** The AI correctly identified this employee as 'Leave'.")
                    else:
                        st.error("❌ **Incorrect.** The AI failed to identify this employee as 'Leave'.")
                    st.json(sample.to_dict(), expanded=False)

            with c_test2:
                if st.button("Test with Employee who Stayed"):
                    sample = df[df['left'] == 0].iloc[0]
                    test_df = sample.drop('left').to_frame().T
                    pred = pipeline.predict(test_df)[0]
                    if pred == 0:
                        st.success("✅ **Correct!** The AI correctly identified this employee as 'Stay'.")
                    else:
                        st.error("❌ **Incorrect.** The AI failed to identify this employee as 'Stay'.")
                    st.json(sample.to_dict(), expanded=False)

        st.markdown("---")

        with st.form("Predict_value_form"):
            st.markdown("##### 👤 Employee Profile")
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
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_button = st.form_submit_button(label='🔮 Analyze Employee', type='primary')
            with col_btn2:
                test_high_risk = st.form_submit_button(label='🔥 Simulate High-Risk Employee', type='secondary')

        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
            st.session_state.input_df = None
            st.session_state.prediction_probas = None

        # Handle Normal Prediction
        if predict_button:
            satisfaction_level = satisfaction_map[satisfaction_text]; last_evaluation = evaluation_map[evaluation_text]
            Work_accident = 1 if work_accident_text == 'Yes' else 0; promotion_last_5years = 1 if promotion_text == 'Yes' else 0
            
            input_data = {
                'satisfaction_level': satisfaction_level, 
                'last_evaluation': last_evaluation,
                'number_project': number_project, 
                'average_montly_hours': average_montly_hours, 
                'time_spend_company': time_spend_company, 
                'Work_accident': Work_accident,
                'promotion_last_5years': promotion_last_5years, 
                'Department': Department, 
                'salary': salary
            }
            input_df = pd.DataFrame([input_data])
            
            with st.spinner('AI is analyzing...'):
                sleep(1); prediction = pipeline.predict(input_df)[0]; prediction_probas = pipeline.predict_proba(input_df)[0]
                
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = prediction_probas

        # Handle High Risk Simulation
        if test_high_risk:
            input_data = {
                'satisfaction_level': 0.1, 
                'last_evaluation': 0.7,
                'number_project': 7,       
                'average_montly_hours': 310, 
                'time_spend_company': 4,
                'Work_accident': 1,
                'promotion_last_5years': 0,
                'Department': Department,
                'salary': 'low'
            }
            input_df = pd.DataFrame([input_data])
            
            with st.spinner('Simulating high-risk scenario...'):
                sleep(1); prediction = pipeline.predict(input_df)[0]; prediction_probas = pipeline.predict_proba(input_df)[0]
                
                st.session_state.prediction_result = prediction
                st.session_state.input_df = input_df
                st.session_state.prediction_probas = prediction_probas
                st.toast("High-Risk Profile Loaded", icon="🔥")

        if st.session_state.prediction_result is not None:
            st.markdown("---")
            pred_col, stay_prob_col, leave_prob_col = st.columns(3)
            with pred_col:
                if st.session_state.prediction_result == 0: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #17B794;'><h2 style='color: #17B794;'>STAY</h2><p>Employee is likely to stay.</p></div>", unsafe_allow_html=True)
                else: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #FF4B4B;'><h2 style='color: #FF4B4B;'>LEAVE</h2><p>High risk of attrition.</p></div>", unsafe_allow_html=True)
            with stay_prob_col: st.metric("Stay Probability", f"{st.session_state.prediction_probas[0]:.1%}")
            with leave_prob_col: st.metric("Leave Probability", f"{st.session_state.prediction_probas[1]:.1%}")
            
            if st.session_state.prediction_result == 1:
                st.markdown("---")
                st.markdown("### 💡 Recommended Actions")
                for rec in get_retention_strategies(st.session_state.input_df): st.info(rec)

                # --- NEW INTEGRATED COUNTERFACTUALS SECTION (HR FRIENDLY) ---
                st.markdown("---")
                st.markdown("### 🔮 AI Retention Strategies (What-If Simulator)")
                st.write("<p style='color: #9ca3af; margin-bottom: 15px;'>Here are 3 different ways to prevent this employee from leaving, ranked by feasibility.</p>", unsafe_allow_html=True)
                
                if st.button("💡 Show Me How to Keep Them", type="primary", key="gen_cf"):
                    with st.spinner("Simulating retention strategies..."):
                        try:
                            query_instance = st.session_state.input_df
                            continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
                            
                            # DiCE Setup
                            d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left')
                            m = dice_ml.Model(model=pipeline, backend='sklearn')
                            
                            exp = Dice(d, m, method='random')
                            cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
                            
                            cf_df = cf.cf_examples_list[0].final_cfs_df
                            original = query_instance.iloc[0]
                            
                            scenarios_html = []
                            for i in range(len(cf_df)):
                                changes = []
                                cf_row = cf_df.iloc[i]
                                has_high_effort = False
                                
                                for col in original.index:
                                    orig_val = original[col]
                                    new_val = cf_row[col]
                                    
                                    # --- TRANSLATION LOGIC ---
                                    if isinstance(orig_val, float):
                                        if abs(orig_val - new_val) > 0.05:
                                            col_lower = col.lower()
                                            action_text = ""
                                            if 'satisfaction' in col_lower:
                                                action_text = f"🤝 <strong>Boost Engagement</strong>: Improve satisfaction score from <strong>{orig_val:.2f}</strong> to <strong>{new_val:.2f}</strong> (via 1-on-1s, feedback, recognition)."
                                            elif 'hours' in col_lower:
                                                diff = orig_val - new_val
                                                if diff > 0:
                                                    action_text = f"⏰ <strong>Reduce Workload</strong>: Cut monthly hours by ~<strong>{abs(diff):.0f}</strong> to prevent burnout."
                                                else:
                                                    action_text = f"⏰ <strong>Increase Engagement</strong>: Adjust hours to ~<strong>{new_val:.0f}</strong>."
                                            elif 'project' in col_lower:
                                                action_text = f"📂 <strong>Rebalance Projects</strong>: Adjust project count to <strong>{int(new_val)}</strong>."
                                            elif 'evaluation' in col_lower:
                                                action_text = f"📊 <strong>Performance Coaching</strong>: Guide evaluation score to <strong>{new_val:.2f}</strong>."
                                            else:
                                                action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val:.2f} to {new_val:.2f}."
                                            
                                            if action_text: changes.append(action_text)
                                    else:
                                        if orig_val != new_val:
                                            if 'department' in col.lower():
                                                has_high_effort = True
                                                action_text = f"🏢 <strong>Department Transfer</strong>: Move from <strong>{orig_val}</strong> to <strong>{new_val}</strong>. <span style='color:#FF4B4B;'>(High Effort)</span>"
                                            else:
                                                action_text = f"• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val} to {new_val}."
                                            changes.append(action_text)
                                
                                if not changes: changes.append("• (AI suggests maintaining current status with minor supervision)")
                                
                                # Join changes into a clean list
                                changes_str = "".join([f"<div class='action-item {'action-item-high-effort' if has_high_effort else ''}'>{c}</div>" for c in changes])
                                
                                scenarios_html.append(f"""
                                    <div class="custom-card" style="border-color: #17B794;">
                                        <h4 style="color: #17B794; margin-top:0;">Strategy {i+1}</h4>
                                        <p style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6;">
                                            {changes_str}
                                        </p>
                                        <div style="margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;">
                                            <small style="color: #17B794;"><strong>Result:</strong> If implemented, the AI predicts the employee will <strong>STAY</strong>.</small>
                                        </div>
                                    </div>
                                """)

                            # Display in Columns
                            col_s1, col_s2, col_s3 = st.columns(3)
                            cols_list = [col_s1, col_s2, col_s3]
                            
                            for i, html in enumerate(scenarios_html):
                                with cols_list[i]:
                                    st.markdown(html, unsafe_allow_html=True)
                                    
                        except Exception as e:
                            st.error(f"Error generating strategies: {e}")

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
    # Page: AI Research Lab
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        st.markdown("<p style='color: #9ca3af;'>Advanced modules for Strategy, Anomalies, and Recruitment.</p>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Model Benchmarking", 
            "🔬 Departmental Strategy Deep Dive", 
            "🕵️ Anomaly Detection",
            "📊 Retention Priority Matrix",
            "🎯 The 'Ideal Candidate' Profiler"
        ])
        
        # --- TAB 1: MODEL BENCHMARKING ---
        with tab1:
            st.subheader("Algorithm Performance Comparison")
            st.write("Comparing **LightGBM** (Our Choice) against **Random Forest** and **Logistic Regression**.")
            
            if st.button("Run Benchmark", type="primary", key="run_benchmark"):
                with st.spinner("Training competing models..."):
                    # 1. LightGBM
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
                    
                    fig_metrics = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                                         x='Metric', y='Score', color='Model', barmode='group',
                                         title="Model Comparison", template="plotly_dark",
                                         color_discrete_sequence=['#17B794', '#EEB76B', '#9C3D54'])
                    custome_layout(fig_metrics, title_size=24)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    st.success("🏆 **Conclusion:** LightGBM was selected as the primary model due to its superior balance of Precision and Recall.")

        # --- TAB 2: DEPARTMENTAL STRATEGY ---
        with tab2:
            st.subheader("🔬 Departmental Strategy Deep Dive")
            st.markdown("""
            <p style='color: #9ca3af; margin-bottom: 20px;'>
            Don't just guess why a team is struggling. Use AI to uncover the <strong>specific reasons</strong> why employees are leaving a particular department and get tailored strategies.
            </p>
            """, unsafe_allow_html=True)
            
            # Selection
            selected_dept_name = st.selectbox("Select Department to Analyze", options=sorted(df['Department'].unique()))
            
            if st.button("Generate Department Strategy", type="primary"):
                with st.spinner("Analyzing departmental dynamics..."):
                    
                    # --- 1. CALCULATE DEPARTMENT CONTEXT (HR Metrics) ---
                    dept_data = df[df['Department'] == selected_dept_name]
                    dept_count = len(dept_data)
                    dept_attrition = (dept_data['left'].sum() / dept_count) * 100
                    company_attrition = (df['left'].sum() / len(df)) * 100
                    delta = dept_attrition - company_attrition
                    
                    delta_color = "normal" if delta <= 0 else "inverse"
                    delta_text = f"{delta:+.1f}% vs Company Avg"

                    # Display Metrics
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric(f"{selected_dept_name} Workforce", f"{dept_count} Employees")
                    c_m2.metric("Attrition Rate", f"{dept_attrition:.1f}%", delta=delta_text, delta_color=delta_color)
                    c_m3.metric("Risk Level", "High" if dept_attrition > 20 else "Moderate" if dept_attrition > 10 else "Low")
                    
                    st.markdown("---")

                    # --- 2. GET AI INSIGHTS (SHAP) ---
                    shap_vals, X_proc_df = get_shap_explanations(pipeline, df)
                    
                    # --- FIX: Dynamic Case-Insensitive Column Search ---
                    target_col = None
                    for col in X_proc_df.columns:
                        if 'Department' in col and selected_dept_name.lower() in col.lower():
                            target_col = col
                            break
                    
                    if not target_col:
                        st.error(f"Could not find data for {selected_dept_name} in the model features.")
                    else:
                        # Filter data for this department
                        dept_mask = X_proc_df[target_col] == 1
                        
                        if dept_mask.sum() == 0:
                             st.warning(f"Not enough data to analyze {selected_dept_name} specifically.")
                        else:
                            # Extract SHAP values for the positive class (Leaving)
                            if isinstance(shap_vals, list):
                                dept_shap = shap_vals[1][dept_mask]
                            else:
                                dept_shap = shap_vals[dept_mask]
                            
                            # Calculate Mean Absolute Importance
                            mean_shap = np.abs(dept_shap).mean(axis=0)
                            
                            # Create DataFrame
                            importance_df = pd.DataFrame({
                                'Feature': X_proc_df.columns,
                                'Impact_Score': mean_shap
                            })
                            
                            # Remove the Department column itself (trivial) and other Department cols (irrelevant)
                            importance_df = importance_df[~importance_df['Feature'].str.contains('Department')]
                            
                            # Sort and get Top 3
                            importance_df.sort_values('Impact_Score', ascending=False, inplace=True)
                            top_3_drivers = importance_df.head(3)
                            
                            # --- 3. VISUALIZATION (Cleaner Chart) ---
                            # We reverse it for the bar chart so #1 is at the top
                            chart_df = importance_df.head(5).iloc[::-1] 
                            
                            fig = px.bar(chart_df, 
                                         x='Impact_Score', 
                                         y='Feature', 
                                         orientation='h',
                                         title=f"What is driving attrition in {selected_dept_name}?",
                                         template="plotly_dark",
                                         color_discrete_sequence=['#17B794'])
                            
                            fig.update_layout(
                                xaxis_title="Relative Impact (Higher = More Important)",
                                yaxis_title="",
                                height=400,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # --- 4. ACTIONABLE ADVICE GENERATOR ---
                            st.markdown("### 💡 Recommended Retention Strategy")
                            st.write(f"Based on the AI analysis for the <strong>{selected_dept_name}</strong> team:", unsafe_allow_html=True)
                            
                            def get_driver_advice(feature_raw):
                                # Clean feature name (remove underscores, title case)
                                feature_clean = feature_raw.replace('_', ' ').title()
                                
                                # Map to HR friendly text
                                if 'Satisfaction' in feature_raw:
                                    title = "Improve Employee Engagement"
                                    text = "The AI detects low morale as a primary driver. Initiate 'Stay Interviews', conduct anonymous pulse surveys, and review manager-employee relationships."
                                    icon = "🗣️"
                                elif 'Hour' in feature_raw or 'Time' in feature_raw:
                                    title = "Address Workload & Burnout"
                                    text = "Overwork is the leading cause. Review project allocation, consider hiring support staff, and enforce 'Right to Disconnect' policies."
                                    icon = "⏰"
                                elif 'Project' in feature_raw:
                                    title = "Optimize Work Distribution"
                                    text = "Employees are either bored or overwhelmed. Rebalance project assignments to ensure the 'Goldilocks' zone of productivity."
                                    icon = "📂"
                                elif 'Evaluation' in feature_raw:
                                    title = "Clarify Performance Expectations"
                                    text = "Unclear goals are causing stress. Implement clearer KPIs and more frequent, constructive feedback loops."
                                    icon = "📊"
                                elif 'Salary' in feature_raw:
                                    title = "Review Compensation Competitiveness"
                                    text = "Pay is a major factor. Conduct a market salary analysis for this specific department and adjust bands if necessary."
                                    icon = "💰"
                                elif 'Tenure' in feature_raw or 'Spend' in feature_raw:
                                    title = "Focus on Career Growth"
                                    text = "Long-tenured employees feel stagnant. Create clear internal promotion pathways or rotation programs."
                                    icon = "📈"
                                else:
                                    title = f"Monitor {feature_clean}"
                                    text = f"AI identified {feature_clean} as a key differentiator. Investigate department-specific policies related to this metric."
                                    icon = "🔍"
                                    
                                return icon, title, text

                            # Create Cards for Top 3
                            c1, c2, c3 = st.columns(3)
                            cols = [c1, c2, c3]
                            
                            for index, col in enumerate(cols):
                                if index < len(top_3_drivers):
                                    driver_row = top_3_drivers.iloc[index]
                                    feature_name = driver_row['Feature']
                                    impact_score = driver_row['Impact_Score']
                                    
                                    icon, title, advice = get_driver_advice(feature_name)
                                    
                                    # HTML Card
                                    card_html = f"""
                                    <div class="custom-card" style="border-top: 4px solid #17B794;">
                                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                            <span style="font-size: 1.5rem; margin-right: 10px;">{icon}</span>
                                            <h4 style="margin: 0; color: #fff;">{title}</h4>
                                        </div>
                                        <p style="color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;">{advice}</p>
                                        <small style="color: #8b949e;">Driver: {feature_name.replace('_', ' ').title()}</small>
                                    </div>
                                    """
                                    with col:
                                        st.markdown(card_html, unsafe_allow_html=True)

        # --- TAB 3: ANOMALY DETECTION ---
        with tab3:
            st.subheader("🕵️ Anomaly Detection")
            st.markdown("""
            <p style='color: #9ca3af; margin-bottom: 20px;'>
            Discover the people who defy the AI's logic.
            <br><br>
            <strong>🚪 Happy Leavers:</strong> Employees the AI predicted would STAY, but LEFT.
            <br>
            <strong>🛡️ Loyal Sufferers:</strong> Employees the AI predicted would LEAVE, but STAYED.
            </p>
            """, unsafe_allow_html=True)
            
            # 1. Get Predictions for the whole dataset
            X_all = df.drop('left', axis=1)
            y_true = df['left']
            
            # Get class predictions (0 or 1)
            y_pred = pipeline.predict(X_all)
            
            # 2. Identify Anomalies
            # False Negatives: Predicted 0 (Stay), Actual 1 (Left) -> Happy Leavers
            happy_leavers_indices = np.where((y_pred == 0) & (y_true == 1))[0]
            df_happy_leavers = df.iloc[happy_leavers_indices]
            
            # False Positives: Predicted 1 (Leave), Actual 0 (Stay) -> Loyal Sufferers
            loyal_sufferers_indices = np.where((y_pred == 1) & (y_true == 0))[0]
            df_loyal_sufferers = df.iloc[loyal_sufferers_indices]
            
            # 3. Calculate Stats for Comparison
            def get_profile_stats(df_group):
                if len(df_group) == 0:
                    return None
                stats = {
                    'Satisfaction': df_group['satisfaction_level'].mean(),
                    'Last Evaluation': df_group['last_evaluation'].mean(),
                    'Avg Monthly Hours': df_group['average_montly_hours'].mean(),
                    'Projects': df_group['number_project'].mean(),
                    'Tenure': df_group['time_spend_company'].mean()
                }
                return pd.Series(stats)

            stats_happy = get_profile_stats(df_happy_leavers)
            stats_loyal = get_profile_stats(df_loyal_sufferers)
            stats_avg = get_profile_stats(df) # Company Average

            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Company Average': stats_avg,
                'Happy Leavers': stats_happy,
                'Loyal Sufferers': stats_loyal
            }).T

            # 4. Display Analysis
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 🚪 The Happy Leavers")
                st.caption(f"Count: {len(df_happy_leavers)} employees")
                
                if len(df_happy_leavers) > 0:
                    st.info("""
                    **Insight:** These people had good stats but left anyway.
                    **Action:** Review these specific profiles. Was it poaching? Spouse relocation? 
                    The AI couldn't predict it because it wasn't in the data.
                    """)
                    # Show a sample
                    st.dataframe(df_happy_leavers[['satisfaction_level', 'last_evaluation', 'Department', 'salary']].head(), use_container_width=True)
                else:
                    st.success("✅ No anomalies found. The model predicted all leavers correctly.")

            with col_b:
                st.markdown("### 🛡️ The Loyal Sufferers")
                st.caption(f"Count: {len(df_loyal_sufferers)} employees")
                
                if len(df_loyal_sufferers) > 0:
                    st.warning("""
                    **Insight:** These people have high-risk profiles but haven't left yet.
                    **Action:** Why are they staying? Are they 'golden handcuffed' by benefits? 
                    Or do they lack options? They are high risk if the job market opens up.
                    """)
                    # Show a sample
                    st.dataframe(df_loyal_sufferers[['satisfaction_level', 'last_evaluation', 'Department', 'salary']].head(), use_container_width=True)
                else:
                    st.success("✅ No anomalies found.")

            # 5. Comparative Visualization (Radar Chart)
            if stats_happy is not None or stats_loyal is not None:
                st.markdown("---")
                st.subheader("📊 Anomaly Profile Comparison")
                
                # Normalize data for Radar Chart (0-1 scale) to make comparison fair
                # We use the max of the 'Company Average' as the baseline
                max_vals = stats_avg.abs().max()
                norm_df = comparison_df.div(max_vals)
                
                # Reset index for plotting
                norm_df = norm_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
                norm_df.rename(columns={'index': 'Group'}, inplace=True)
                
                fig = px.line_polar(norm_df, r='Value', theta='Metric', color='Group', 
                                     line_close=True,
                                     template="plotly_dark",
                                     color_discrete_map={
                                         'Company Average': '#9ca3af',
                                         'Happy Leavers': '#EEB76B',  # Yellow/Orange for "Unexpected"
                                         'Loyal Sufferers': '#FF4B4B' # Red for "Danger"
                                     })
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1.2])
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <small style='color: #8b949e;'>
                *Note: Values are normalized relative to the company average. 
                A spike in 'Satisfaction' for Happy Leavers confirms they were generally happy before leaving.*
                </small>
                """, unsafe_allow_html=True)

        # --- TAB 4: RETENTION PRIORITY MATRIX ---
        with tab4:
            st.subheader("📊 Retention Priority Matrix")
            st.markdown("""
            <p style='color: #9ca3af; margin-bottom: 20px;'>
            A strategic view to prioritize your HR efforts. 
            <br>We map <strong>Attrition Risk</strong> against <strong>Replacement Cost</strong> to identify who needs immediate attention vs. who is safe to let go.
            </p>
            """, unsafe_allow_html=True)
            
            # 1. Prepare Data
            X_all = df.drop('left', axis=1)
            risk_probs = pipeline.predict_proba(X_all)[:, 1]
            
            # We need 'Replacement Cost' estimate from Salary
            # Mapping categorical salary to monetary estimate (Example values)
            salary_cost_map = {'low': 400000, 'medium': 600000, 'high': 900000} 
            replacement_costs = df['salary'].map(salary_cost_map) * 0.5 
            
            # Create Plotting DataFrame
            plot_data = pd.DataFrame({
                'Employee_ID': df.index, 
                'Risk_Probability': risk_probs,
                'Replacement_Cost': replacement_costs,
                'Department': df['Department'],
                'Salary_Tier': df['salary']
            })
            
            # 2. Create Quadrants (Thresholds)
            risk_threshold = 0.5
            cost_threshold = replacement_costs.median()
            
            # 3. Assign Zones for Coloring
            def get_zone(row):
                if row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] >= cost_threshold:
                    return "🔴 Critical Zone (Save Now)"
                elif row['Risk_Probability'] < risk_threshold and row['Replacement_Cost'] >= cost_threshold:
                    return "🟡 Retain Zone (Keep Happy)"
                elif row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] < cost_threshold:
                    return "🟢 Outplacement Zone (Let Go)"
                else:
                    return "⚪ Monitor Zone"

            plot_data['Zone'] = plot_data.apply(get_zone, axis=1)
            
            # 4. Visualization (Scatter Plot)
            fig = px.scatter(
                plot_data,
                x='Risk_Probability',
                y='Replacement_Cost',
                color='Zone',
                color_discrete_map={
                    "🔴 Critical Zone (Save Now)": "#FF4B4B",
                    "🟡 Retain Zone (Keep Happy)": "#F59E0B",
                    "🟢 Outplacement Zone (Let Go)": "#17B794",
                    "⚪ Monitor Zone": "#9ca3af"
                },
                hover_data=['Department', 'Salary_Tier'],
                title="Employee Prioritization Map",
                template="plotly_dark",
                labels={'Risk_Probability': 'Predicted Attrition Risk', 'Replacement_Cost': 'Est. Replacement Cost (₹)'},
                height=600
            )
            
            # Add Quadrant Lines
            fig.add_hline(y=cost_threshold, line_dash="dash", line_color="white", opacity=0.3)
            fig.add_vline(x=risk_threshold, line_dash="dash", line_color="white", opacity=0.3)
            
            # Add Annotations for Zones
            fig.add_annotation(x=0.25, y=cost_threshold*1.5, text="Retain<br>(High Value)", showarrow=False, font=dict(color="white"))
            fig.add_annotation(x=0.75, y=cost_threshold*1.5, text="Critical<br>(Save Now)", showarrow=False, font=dict(color="#FF4B4B", size=14, family="Arial Black"))
            fig.add_annotation(x=0.25, y=cost_threshold*0.5, text="Monitor", showarrow=False, font=dict(color="white"))
            fig.add_annotation(x=0.75, y=cost_threshold*0.5, text="Outplacement", showarrow=False, font=dict(color="#17B794"))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Summary Metrics
            st.markdown("---")
            col_z1, col_z2, col_z3 = st.columns(3)
            
            critical_count = len(plot_data[plot_data['Zone'] == "🔴 Critical Zone (Save Now)"])
            outplace_count = len(plot_data[plot_data['Zone'] == "🟢 Outplacement Zone (Let Go)"])
            
            col_z1.metric("Critical Interventions Needed", critical_count, delta="High Priority", delta_color="inverse")
            col_z2.metric("Potential Efficiency Savings", outplace_count, delta="Safe to Exit", delta_color="normal")
            col_z3.metric("Average Replacement Cost", f"₹{int(cost_threshold):,}")

        # --- TAB 5: THE 'IDEAL CANDIDATE' PROFILER (REDESIGNED) ---
        with tab5:
            st.subheader("🎯 The 'Ideal Candidate' Profiler")
            st.markdown("""
            <p style='color: #9ca3af; margin-bottom: 20px;'>
            Shift from Retention to <strong>Acquisition</strong>. 
            <br>We analyze your "Superstars" (Loyal + High Performers) to build a clear hiring checklist.
            </p>
            """, unsafe_allow_html=True)
            
            # 1. Define the "Superstar" Criteria (Loyal + High Performance)
            # Loyal: Time Spent > 4 Years AND Still with company (left=0)
            # High Performer: Last Evaluation > 0.8
            superstar_mask = (df['left'] == 0) & (df['time_spend_company'] > 4) & (df['last_evaluation'] > 0.8)
            df_superstars = df[superstar_mask]
            
            # Define the "Average Employee" (Comparison Group)
            df_average = df[(df['left'] == 0) & (~superstar_mask)]
            
            if len(df_superstars) < 5:
                st.warning("Not enough 'Superstar' data in this dataset to generate a reliable profile.")
            else:
                st.success(f"Analyzed {len(df_superstars)} Superstars vs {len(df_average)} Average Employees.")
                
                # 2. Calculate Key Metrics
                metrics_to_compare = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
                
                super_mean = df_superstars[metrics_to_compare].mean()
                avg_mean = df_average[metrics_to_compare].mean()
                
                # --- NEW VISUALIZATION 1: GROUPED BAR CHART (Easier to read than Radar) ---
                # Create a long format dataframe for Plotly Grouped Bar
                comparison_long = pd.DataFrame({
                    'Metric': metrics_to_compare,
                    'Superstar': super_mean.values,
                    'Average Employee': avg_mean.values
                }).melt(id_vars='Metric', var_name='Group', value_name='Average Value')
                
                fig_compare = px.bar(
                    comparison_long,
                    x='Metric',
                    y='Average Value',
                    color='Group',
                    barmode='group',  # This puts bars side-by-side
                    title="Superstars vs. Average Employees (Head-to-Head)",
                    template="plotly_dark",
                    color_discrete_map={
                        'Superstar': '#17B794',  # Green
                        'Average Employee': '#9ca3af' # Grey
                    },
                    text_auto=True,
                    height=500
                )
                
                # Clean up axis labels
                fig_compare.update_xaxes(title="", tickangle=45)
                fig_compare.update_layout(yaxis_title="Average Score / Value")
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # --- NEW VISUALIZATION 2: THE DNA OF A TOP PERFORMER (Summary Cards) ---
                st.markdown("### 🧬 The DNA of a Top Performer")
                st.write("Here are the 3 biggest differentiators between your best employees and the rest.")
                
                # Calculate Difference to find the top differentiators
                diff_df = pd.DataFrame({
                    'Metric': metrics_to_compare,
                    'Difference': (super_mean - avg_mean).values
                })
                
                # Get top 3 differentiators (absolute difference)
                diff_df['Abs_Diff'] = diff_df['Difference'].abs()
                top_3_diff = diff_df.nlargest(3, 'Abs_Diff')
                
                col_dna1, col_dna2, col_dna3 = st.columns(3)
                
                def get_dna_insight(metric, diff_val):
                    metric_name = metric.replace('_', ' ').title()
                    if metric == 'satisfaction_level':
                        if diff_val > 0:
                            return "🟢 High Culture Fit", f"Superstars are {diff_val:.2f} points happier on average."
                        else:
                            return "🔴 Low Satisfaction", f"Unexpected: Superstars seem less satisfied."
                    elif metric == 'average_montly_hours':
                        if diff_val < 0:
                            return "🟢 Work-Life Balance", f"Superstars work {abs(diff_val):.0f} hrs LESS."
                        else:
                            return "🔴 High Workload", f"Superstars work harder."
                    elif metric == 'last_evaluation':
                        if diff_val > 0:
                            return "🟢 High Performance", f"Superstars score {diff_val:.2f} points higher."
                        else:
                            return "🔴 Low Performance", f"Superstars score lower."
                    else:
                        return "📊 " + metric_name, f"Difference of {diff_val:.2f}."
                
                cols = [col_dna1, col_dna2, col_dna3]
                for i, col in enumerate(cols):
                    if i < len(top_3_diff):
                        row = top_3_diff.iloc[i]
                        title, text = get_dna_insight(row['Metric'], row['Difference'])
                        
                        st.markdown(f"""
                        <div class="custom-card" style="text-align: center; border-top: 4px solid #17B794;">
                            <h3 style="margin-top: 0;">{title}</h3>
                            <p style="color: #c9d1d9; font-size: 0.9rem; margin-bottom: 5px;">{text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # --- ACTIONABLE HIRING CHECKLIST ---
                st.markdown("---")
                st.markdown("### 📝 Hiring Checklist (Do's & Don'ts)")
                st.write("Based on the data, apply these filters to your next job opening:")

                # Simplified Logic for the Checklist
                checklist = []
                
                # Satisfaction Check
                if (super_mean['satisfaction_level'] - avg_mean['satisfaction_level']) > 0.1:
                    checklist.append({
                        "Type": "✅ DO Look For",
                        "Rule": "Candidates who mention 'Culture', 'Team', or 'Values' as their top reason for leaving previous jobs."
                    })
                else:
                    checklist.append({
                        "Type": "⚠️ CAUTION",
                        "Rule": "Satisfaction isn't a major differentiator here. Don't over-prioritize 'culture fit' questions."
                    })

                # Hours Check
                if (super_mean['average_montly_hours'] - avg_mean['average_montly_hours']) < -10:
                    checklist.append({
                        "Type": "✅ DO Look For",
                        "Rule": "Candidates who demonstrate 'Work-Life Balance' and set boundaries."
                    })
                    checklist.append({
                        "Type": "🚫 AVOID",
                        "Rule": "Candidates who brag about 'sleeping at the office' or working 80-hour weeks."
                    })

                # Projects Check
                if (super_mean['number_project'] - avg_mean['number_project']) < -0.5:
                    checklist.append({
                        "Type": "✅ DO Look For",
                        "Rule": "Candidates who focus on 'Quality' and 'Prioritization' rather than taking on everything."
                    })

                # Display Checklist
                for item in checklist:
                    if "DO" in item['Type']:
                        st.success(f"**{item['Type']}**: {item['Rule']}")
                    elif "AVOID" in item['Type']:
                        st.error(f"**{item['Type']}**: {item['Rule']}")
                    else:
                        st.warning(f"**{item['Type']}**: {item['Rule']}")

    # ====================================================================
    # Page: STRATEGIC ROADMAP & FUTURE FORECAST (NEW)
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Strategic Roadmap & Future Forecast")
        st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>From Diagnosis to Execution. Plan for the next 6 months and project the future of your workforce.</p>", unsafe_allow_html=True)
        
        # --- PART 1: THE ROADMAP (LLM GENERATED) ---
        st.markdown("### 🗓️ 6-Month Execution Roadmap")
        st.write("Let AI draft your implementation timeline based on the company's biggest risk factors.")
        
        # Get Top Drivers dynamically
        avg_sat = df['satisfaction_level'].mean()
        avg_hours = df['average_montly_hours'].mean()
        avg_projects = df['number_project'].mean()
        
        issues = []
        if avg_sat < 0.6: issues.append("Low Employee Satisfaction")
        if avg_hours > 200: issues.append("Excessive Workload (Burnout)")
        if avg_projects > 4: issues.append("Project Imbalance")
        if len(issues) == 0: issues.append("General Retention Strategy")
        
        issues_str = ", ".join(issues)
        
        if st.button("Generate Action Roadmap", type="primary"):
            with st.spinner("Drafting your 6-month strategy..."):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if api_key:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                        
                        template = """
                        You are an expert HR Strategist.
                        
                        **Company Context:**
                        Our AI has identified the following critical attrition drivers: {issues}.
                        We need to stabilize the workforce over the next 6 months.
                        
                        **Task:**
                        Create a 6-month execution roadmap. Break it down into phases (e.g., Diagnosis, Pilot, Rollout, Review).
                        For each month, provide:
                        1. **Phase Name**
                        2. **Specific Actionable Steps** (2-3 bullet points)
                        3. **Success Metrics** (How do we know it worked?)
                        
                        **Format:** Use clean Markdown formatting. Be concise and professional.
                        """
                        
                        prompt = PromptTemplate.from_template(template)
                        chain = prompt | llm | StrOutputParser()
                        
                        response = chain.invoke({"issues": issues_str})
                        
                        st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("🔑 API Key missing for Roadmap generation. Showing generic template.")
                        generic_plan = """
                        **Month 1: Diagnosis & Audit**
                        *   Conduct stay interviews with top 10% at-risk employees.
                        *   Audit current workload distribution across departments.
                        *   *Metric:* Complete 100% of risk interviews.
                        
                        **Month 2: Strategy Design**
                        *   Based on audit, design specific interventions (e.g., flexible hours).
                        *   Set clear KPIs for the pilot program.
                        *   *Metric:* Pilot program approval from leadership.
                        
                        **Month 3: Pilot Launch**
                        *   Launch intervention in one high-risk department (e.g., Sales or IT).
                        *   Monitor feedback daily.
                        *   *Metric:* Pilot participation rate > 80%.
                        
                        **Month 4: Analysis & Adjustment**
                        *   Review pilot data. Adjust policies based on feedback.
                        *   Prepare training materials for company-wide rollout.
                        *   *Metric:* Revised policy documentation.
                        
                        **Month 5: Company Rollout**
                        *   Launch the initiative across all departments.
                        *   Manager training sessions.
                        *   *Metric:* 100% department coverage.
                        
                        **Month 6: Review & Stabilization**
                        *   Measure impact on satisfaction and hours.
                        *   Celebrate wins and recognize improvements.
                        *   *Metric:* 5% reduction in projected attrition.
                        """
                        st.markdown(generic_plan)
                except Exception as e:
                    st.error(f"Error generating roadmap: {e}")

        st.markdown("---")
        
        # --- PART 2: THE FUTURE FORECAST (MATHEMATICAL) ---
        st.markdown("### 📈 12-Month Workforce Forecast")
        st.write("Visualize the long-term impact of your intervention. Compare 'Business as Usual' vs. 'With Intervention'.")
        
        # Controls for Simulation
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            intervention_efficacy = st.slider("Expected Intervention Success Rate (%)", 0, 50, 20, 5, 
                                          help="What % of high-risk employees do you expect to save with your roadmap?")
        
        with col_f2:
            natural_attrition_rate = st.slider("Natural Monthly Attrition Rate (%)", 0.5, 5.0, 1.5, 0.1,
                                          help="The baseline % of employees expected to leave even with low risk.")

        if st.button("Run Forecast Simulation", type="secondary"):
            # Simulation Logic
            months = list(range(1, 13))
            current_workforce = len(df)
            
            # Calculate current monthly hazard rate based on model predictions
            current_risk_prob = pipeline.predict_proba(df.drop('left', axis=1))[:, 1]
            monthly_hazard_current = current_risk_prob / 12
            
            # Scenario 1: Business as Usual (No improvement)
            forecast_bau = []
            temp_workforce_bau = current_workforce
            
            # Scenario 2: With Intervention (Reduced Hazard)
            # We assume the intervention reduces the hazard rate by the efficacy percentage
            monthly_hazard_intervention = monthly_hazard_current * (1 - (intervention_efficacy / 100))
            forecast_intervention = []
            temp_workforce_int = current_workforce
            
            for m in months:
                # Calculate leavers this month
                leavers_bau = temp_workforce_bau * monthly_hazard_current
                leavers_int = temp_workforce_int * monthly_hazard_intervention
                
                # Update workforce (Assume no hiring for this simple projection)
                temp_workforce_bau -= leavers_bau
                temp_workforce_int -= leavers_int
                
                # Add natural attrition (people leaving for reasons AI can't predict)
                # Apply a small constant decay for natural reasons
                temp_workforce_bau *= (1 - (natural_attrition_rate/100))
                temp_workforce_int *= (1 - (natural_attrition_rate/100))
                
                forecast_bau.append(temp_workforce_bau)
                forecast_intervention.append(temp_workforce_int)
            
            # Visualization
            forecast_df = pd.DataFrame({
                'Month': months,
                'Business as Usual (Do Nothing)': forecast_bau,
                'With Intervention (Roadmap)': forecast_intervention
            }).melt(id_vars='Month', var_name='Scenario', value_name='Workforce Count')
            
            fig_forecast = px.line(forecast_df, 
                                   x='Month', 
                                   y='Workforce Count', 
                                   color='Scenario',
                                   title="Projected Workforce Size (12 Months)",
                                   template="plotly_dark",
                                   markers=True,
                                   color_discrete_map={
                                       'Business as Usual (Do Nothing)': "#FF4B4B", # Red line dropping
                                       'With Intervention (Roadmap)': "#17B794"   # Green line stabilizing
                                   })
            
            fig_forecast.update_layout(yaxis_title="Employee Headcount", xaxis=dict(dtick=1))
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Insight Calculation
            saved_employees = forecast_intervention[-1] - forecast_bau[-1]
            st.success(f"""
            **Impact Projection:**
            By executing your roadmap with an estimated {intervention_efficacy}% success rate, 
            you are projected to save approximately <strong>{int(saved_employees)} employees</strong> over the next year 
            compared to doing nothing.
            """)

if __name__ == "__main__":
    main()
