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
</style>
""", unsafe_allow_html=True)

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(showlegend=showlegend, title={"font": {"size": title_size, "family": "tahoma"}}, hoverlabel={"bgcolor": "#000", "font_size": hover_font_size, "font_family": "arial"}, paper_bgcolor="#0E1117", plot_bgcolor="#161b22", font_color="#c9d1d9")

def box_plot(the_df, column):
    fig = px.box(data_frame=the_df, x=column, title=f'{column.title().replace("_", " ")} Distribution & 5-Summary', template="plotly_dark", labels={column: column.title().replace("_", " ")}, height=600, color_discrete_sequence=['#17B794'])
    custome_layout(fig, showlegend=False); return fig

def bar_plot(the_df, column, orientation="v", top_10=False):
    dep = the_df[column].value_counts()
    if top_10: dep = the_df[column].value_counts().nlargest(10)
    fig = px.bar(data_frame=dep, x=dep.index, y=dep.values, orientation=orientation, color=dep.index.astype(str), title=f'Observations Distribution Via {column.title().replace("_", " ")}', color_discrete_sequence=["#17B794"], labels={"x": column.title().replace("_", " "), "y": "Count of Employees"}, template="plotly_dark", text_auto=True, height=650)
    custome_layout(fig, title_size=28); return fig

def pie_chart(the_df, column):
    counts = the_df[column].value_counts()
    fig = px.pie(data_frame=counts, names=counts.index, values=counts.values, title=f'Popularity of {column.title().replace("_", " ")}', color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"], template="plotly_dark", height=650)
    custome_layout(fig, showlegend=True, title_size=28); pulls = np.zeros(len(counts))
    if len(pulls) > 1: pulls[-1] = 0.1
    fig.update_traces(textfont={"size": 16, "family": "arial", "color": "#fff"}, hovertemplate="Label:%{label}<br>Frequency: %{value:0.4s}<br>Percentage: %{percent}", marker=dict(line=dict(color='#000000', width=0.5)), pull=pulls); return fig

def create_heat_map(the_df):
    numeric_df = the_df.select_dtypes(include=np.number); correlation = numeric_df.corr()
    fig = px.imshow(correlation, template="plotly_dark", text_auto="0.2f", aspect=1, color_continuous_scale="greens", title="Correlation Heatmap of Data", height=650)
    custome_layout(fig); return fig

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
# Logic Engine Functions (Safeguarded for Global Use)
# ====================================================================
def analyze_why_people_leave(df):
    st.markdown("### 🔍 Why do people leave?")
    st.markdown("<p style='color: #9ca3af; margin-bottom: 20px;'>Our AI has analyzed the data to find the root causes of attrition.</p>", unsafe_allow_html=True)
    required_cols = ['salary', 'satisfaction_level', 'average_montly_hours', 'number_project']
    if all(col in df.columns for col in required_cols):
        df_causal = df.copy(); salary_map = {'low': 1, 'medium': 2, 'high': 3}
        df_causal['salary_num'] = df_causal['salary'].map(salary_map)
        causal_graph = """digraph { salary_num -> satisfaction_level; satisfaction_level -> left; average_montly_hours -> left; number_project -> average_montly_hours; }"""
        df_model = df_causal[['salary_num', 'satisfaction_level', 'average_montly_hours', 'number_project', 'left']]
        st.markdown("<div class='custom-card'><h4 style='margin-top:0; color: #17B794;'>📊 AI Causal Logic Diagram</h4><p style='font-size: 0.9em; color: #8b949e;'>Internal hypothesis: Salary impacts Satisfaction, which leads to Attrition.</p></div>", unsafe_allow_html=True)
        st.graphviz_chart(causal_graph); st.markdown("<br>", unsafe_allow_html=True)
        effects = {}
        try:
            model_sal = CausalModel(data=df_model, treatment='salary_num', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_sal = model_sal.estimate_effect(model_sal.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression"); effects['Salary'] = abs(est_sal.value)
            model_sat = CausalModel(data=df_model, treatment='satisfaction_level', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_sat = model_sat.estimate_effect(model_sat.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression"); effects['Satisfaction'] = abs(est_sat.value)
            model_hr = CausalModel(data=df_model, treatment='average_montly_hours', outcome='left', graph=causal_graph.replace('\n', ' '))
            est_hr = model_hr.estimate_effect(model_hr.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression"); effects['Overwork'] = abs(est_hr.value) * 10
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
        except Exception as e: st.warning("Could not generate Causal Graph for this specific dataset. Relying on SHAP below.")
    else:
        st.info("📊 *Advanced Causal Graph requires specific columns (satisfaction_level, salary, etc.) which are not in this dataset. Using dynamic SHAP analysis instead.*")

def plan_retention_budget(df, pipeline, budget_limit):
    st.markdown("### 💰 Retention Budget Planner")
    X = df.drop('left', axis=1); probas = pipeline.predict_proba(X)[:, 1]; opt_df = df.copy()
    opt_df['attrition_risk'] = probas; high_risk_df = opt_df[opt_df['attrition_risk'] > 0.5].copy()
    if len(high_risk_df) == 0: st.success("🎉 Great news! The workforce is stable."); return None
    if 'salary' in df.columns:
        salary_val_map = {'low': 400000, 'medium': 600000, 'high': 900000}; high_risk_df['annual_salary'] = high_risk_df['salary'].map(salary_val_map)
    else: high_risk_df['annual_salary'] = 500000
    high_risk_df['replacement_cost'] = high_risk_df['annual_salary'] * 0.5; high_risk_df['expected_loss'] = high_risk_df['attrition_risk'] * high_risk_df['replacement_cost']
    high_risk_df['intervention_cost'] = high_risk_df['annual_salary'] * 0.10; high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['intervention_cost']
    candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
    if len(candidates) == 0: st.warning("⚠️ It is currently not cost-effective to offer raises."); return None
    n = len(candidates); c = -candidates['net_savings'].values; A = np.array([candidates['intervention_cost'].values]); b = np.array([budget_limit]); integrality = np.ones(n)
    with st.spinner("🧮 Calculating optimal resource allocation..."):
        try: res = milp(c=c, constraints=LinearConstraint(A, lb=-np.inf, ub=b), integrality=integrality)
        except Exception as e: st.error(f"Calculation Error: {e}"); return None
    if res.success:
        selected_indices = np.where(res.x == 1)[0]; selected_employees = candidates.iloc[selected_indices]
        total_cost = selected_employees['intervention_cost'].sum(); total_savings = selected_employees['net_savings'].sum()
        return selected_employees, total_cost, total_savings
    else: st.error("❌ Budget is too low."); return None

def run_groq_consultant(employee_name, department, situation, solution, budget):
    st.subheader("✍️ AI Communication Assistant")
    if "overwork" in situation.lower(): root_cause = "High Workload & Potential Burnout"
    elif "salary" in situation.lower(): root_cause = "Compensation & Salary Competitiveness"
    elif "morale" in situation.lower(): root_cause = "Low Job Satisfaction & Morale"
    else: root_cause = "Attrition Risk Factors"
    action_description = solution; cost_str = budget
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key: st.warning("🔑 System Error: API Key missing."); return
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
    except Exception as e: st.error(f"Connection Error: {e}"); return
    template = "You are an expert HR Consultant.\n**Employee:** {employee_name} ({department})\n**Situation:** {situation}\n**Root Cause:** {root_cause}\n**Solution:** {action_description}\n**Cost:** {cost_str}\n**Task:** Write a polite, professional email draft to the employee. Acknowledge value, address situation, propose solution.\n**Tone:** Professional, Supportive."
    prompt = PromptTemplate.from_template(template); chain = prompt | llm | StrOutputParser()
    with st.spinner("Drafting your message..."):
        try:
            response = chain.invoke({"employee_name": employee_name, "department": department, "situation": situation, "root_cause": root_cause, "action_description": action_description, "cost_str": cost_str})
            st.markdown("#### 📧 Generated Email Draft"); st.markdown(f"<div class='llm-response'>{response}</div>", unsafe_allow_html=True)
        except Exception as e: st.error(f"Error generating draft: {e}")

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(page_title="Global Employee Retention AI", page_icon="🤖", layout="wide")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ====================================================================
    # THE GLOBAL ROUTER
    # ====================================================================
    if 'is_global' in st.session_state and st.session_state['is_global']:
        pipeline = st.session_state['global_pipeline']
        df = st.session_state['global_df']
        X_train_ref = st.session_state['global_X_train']
        X_test_cur = st.session_state['global_X_test']
        y_test = st.session_state.get('global_y_test', pd.Series([0]))
        st.toast("✅ Using Custom Uploaded Company Data", icon="📊")
    else:
        @st.cache_data
        def load_data_and_train_model():
            st.write("📂 Step 1/3: Loading Default Dataset...")
            df = pd.read_csv('HR_comma_sep.csv')
            st.write("🧹 Step 2/3: Preprocessing & Splitting Data...")
            df_train = df.drop_duplicates().reset_index(drop=True)
            X = df_train.drop('left', axis=1); y = df_train['left']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            categorical_features = X.select_dtypes(include=['object']).columns; numerical_features = X.select_dtypes(include=np.number).columns
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
            st.write("🤖 Step 3/3: Training AI Model (LightGBM)...")
            best_params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 10, 'random_state': 42, 'verbose': -1, 'class_weight': 'balanced', 'scale_pos_weight': 15}
            final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgb.LGBMClassifier(**best_params))])
            final_pipeline.fit(X_train, y_train)
            return final_pipeline, df, X_train, X_test, y_train, y_test
        pipeline, df, X_train_ref, X_test_cur, y_train, y_test = load_data_and_train_model()
        st.empty()

    # ====================================================================
    # Core AI Functions (SHAP & Strategies)
    # ====================================================================
    @st.cache_data
    def get_shap_explanations(_pipeline, _df):
        model = _pipeline.named_steps['classifier']; preprocessor = _pipeline.named_steps['preprocessor']
        X = _df.drop('left', axis=1).drop_duplicates(); X = X.select_dtypes(include=[np.number, 'object'])
        X_processed = preprocessor.transform(X)
        if issparse(X_processed): X_processed = X_processed.toarray()
        clean_names = [name.split('__')[-1].replace('_', ' ').title() for name in preprocessor.get_feature_names_out()]
        X_processed_df = pd.DataFrame(X_processed, columns=clean_names)
        booster = model.booster_ if hasattr(model, "booster_") else model._Booster if hasattr(model, "_Booster") else model.booster if hasattr(model, "booster") else model
        explainer = shap.TreeExplainer(booster); shap_values = explainer.shap_values(X_processed_df)
        return shap_values, X_processed_df

    def get_retention_strategies(employee_data):
        strategies = []
        if isinstance(employee_data, pd.DataFrame): employee_data = employee_data.iloc[0]
        if 'satisfaction_level' in employee_data.index and employee_data['satisfaction_level'] <= 0.45: strategies.append("🗣️ Conduct 1-on-1 meeting.")
        if 'number_project' in employee_data.index:
            if employee_data['number_project'] <= 2: strategies.append("📈 Discuss career aspirations.")
            if employee_data['number_project'] >= 6: strategies.append("⚠️ Assess workload/burnout.")
        if 'time_spend_company' in employee_data.index and employee_data['time_spend_company'] >= 4 and employee_data.get('promotion_last_5years', 0) == 0: strategies.append("📊 Develop career path.")
        if not strategies: strategies.append("✅ Monitor engagement based on AI risk score.")
        return strategies

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<div style='padding: 20px; text-align: center;'><h1 style='font-size: 1.8rem; color: #17B794; margin-bottom: 0;'>Global AI Retention</h1><p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px;'>Enterprise Dashboard</p></div><hr style='border-color: #30363d; margin: 20px 0;'>", unsafe_allow_html=True)
        page = option_menu(menu_title=None, options=['⚙️ Global Setup', 'Home', 'Employee Insights', 'Predict Attrition', 'Why They Leave', 'Budget Planner', 'AI Assistant', 'AI Research Lab', 'Strategic Roadmap'], icons=['gear', 'house', 'bar-chart-line-fill', "graph-up-arrow", 'helpful-tip-fill', 'currency-rupee', 'robot', 'cpu', 'flag-2-fill'], menu_icon="cast", default_index=0, styles={"container": {"padding": "0!important", "background-color": 'transparent'}, "icon": {"color": "#17B794", "font-size": "18px"}, "nav-link": {"color": "#c9d1d9", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "10px"}, "nav-link-selected": {"background-color": "#21262d", "border-radius": "8px", "color": "#17B794"}})
        st.markdown("<br><br><div style='padding: 20px; text-align: center; color: #8b949e; font-size: 0.8rem;'>Developed by<br><strong>Nisarg Rathod</strong></div>", unsafe_allow_html=True)

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
                        if len(categorical_auto) == 0: preprocessor_global = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_auto)])
                        else: preprocessor_global = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_auto), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_auto)])
                        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        global_pipeline = Pipeline(steps=[('preprocessor', preprocessor_global), ('classifier', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1, class_weight='balanced'))])
                        global_pipeline.fit(X_train_g, y_train_g)
                        y_pred_g = global_pipeline.predict(X_test_g); acc = accuracy_score(y_test_g, y_pred_g)
                        final_df = new_df.loc[valid_idx].copy(); final_df['left'] = y_clean
                        st.session_state['global_pipeline'] = global_pipeline; st.session_state['global_df'] = final_df
                        st.session_state['global_X_train'] = X_train_g; st.session_state['global_X_test'] = X_test_g
                        st.session_state['global_y_test'] = y_test_g; st.session_state['is_global'] = True
                        st.balloons(); st.success(f"🎉 Training Complete! Accuracy: **{acc:.1%}**.")
                        st.info("Go to **'Predict Attrition'** or **'Why They Leave'** to use your data!")
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
        col1, col2 = st.columns(2)
        col1.metric("Total Workforce", f"{total_employees:,}"); col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta="Current")
        if 'satisfaction_level' in df.columns:
            avg_satisfaction = df['satisfaction_level'].mean(); st.columns(1)[0].metric("Avg. Satisfaction", f"{avg_satisfaction:.2f} / 1.0")
        st.markdown("---"); st.markdown("### 📄 Employee Data Snapshot"); st.dataframe(df.head(100), use_container_width=True)

    # ====================================================================
    # PAGE: EMPLOYEE INSIGHTS
    # ====================================================================
    if page == "Employee Insights":
        st.header("📉 Employee Data Analysis"); st.write("Explore the workforce demographics.")
        create_vizualization(df, viz_type="box", data_type="number"); create_vizualization(df, viz_type="bar", data_type="object"); create_vizualization(df, viz_type="pie")
        if len(df.select_dtypes(include=np.number).columns) > 2: st.plotly_chart(create_heat_map(df), use_container_width=True)

    # ====================================================================
    # PAGE: PREDICT ATTRITION (DYNAMIC FORM LOGIC)
    # ====================================================================
    if page == "Predict Attrition":
        st.markdown("<h1 style='margin-bottom: 5px;'>🎯 Predict Attrition</h1>", unsafe_allow_html=True)
        feature_columns = [c for c in df.columns if c != 'left']
        is_default_data = 'satisfaction_level' in feature_columns # Check if we are using the hardcoded dataset
        
        with st.form("Predict_value_form"):
            st.markdown("##### 👤 Employee Profile")
            input_data = {}
            
            if is_default_data:
                # ORIGINAL BEAUTIFUL HR SLIDERS FOR DEFAULT DATA
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
                # DYNAMIC AUTO-GENERATOR FOR GLOBAL DATA
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

        if 'prediction_result' not in st.session_state: st.session_state.prediction_result = None; st.session_state.input_df = None; st.session_state.prediction_probas = None

        if predict_button:
            input_df = pd.DataFrame([input_data])
            with st.spinner('AI is analyzing...'):
                sleep(1); input_df = input_df[feature_columns]
                prediction = pipeline.predict(input_df)[0]; prediction_probas = pipeline.predict_proba(input_df)[0]
                st.session_state.prediction_result = prediction; st.session_state.input_df = input_df; st.session_state.prediction_probas = prediction_probas

        if st.session_state.prediction_result is not None:
            st.markdown("---"); pred_col, stay_prob_col, leave_prob_col = st.columns(3)
            with pred_col:
                if st.session_state.prediction_result == 0: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #17B794;'><h2 style='color: #17B794;'>STAY</h2><p>Employee is likely to stay.</p></div>", unsafe_allow_html=True)
                else: st.markdown("<div class='custom-card' style='text-align: center; border: 1px solid #FF4B4B;'><h2 style='color: #FF4B4B;'>LEAVE</h2><p>High risk of attrition.</p></div>", unsafe_allow_html=True)
            with stay_prob_col: st.metric("Stay Probability", f"{st.session_state.prediction_probas[0]:.1%}")
            with leave_prob_col: st.metric("Leave Probability", f"{st.session_state.prediction_probas[1]:.1%}")
            
            if st.session_state.prediction_result == 1:
                st.markdown("---"); st.markdown("### 💡 Recommended Actions")
                for rec in get_retention_strategies(st.session_state.input_df): st.info(rec)
                st.markdown("---"); st.markdown("### 🔮 AI Retention Strategies (What-If Simulator)")
                if st.button("💡 Show Me How to Keep Them", type="primary", key="gen_cf"):
                    with st.spinner("Simulating retention strategies..."):
                        try:
                            query_instance = st.session_state.input_df; continuous_features = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                            if not continuous_features: st.error("No numerical columns found for simulation.")
                            else:
                                d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='left'); m = dice_ml.Model(model=pipeline, backend='sklearn')
                                exp = Dice(d, m, method='random'); cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
                                cf_df = cf.cf_examples_list[0].final_cfs_df; original = query_instance.iloc[0]; scenarios_html = []
                                for i in range(len(cf_df)):
                                    changes = []; cf_row = cf_df.iloc[i]; has_high_effort = False
                                    for col in original.index:
                                        orig_val = original[col]; new_val = cf_row[col]
                                        if isinstance(orig_val, float) and abs(orig_val - new_val) > 0.05:
                                            changes.append(f"<div class='action-item'>• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val:.2f} to {new_val:.2f}.</div>")
                                        elif not isinstance(orig_val, float) and orig_val != new_val:
                                            has_high_effort = True; changes.append(f"<div class='action-item action-item-high-effort'>• <strong>{col.replace('_', ' ').title()}</strong>: Change from {orig_val} to {new_val}.</div>")
                                    if not changes: changes.append("<div class='action-item'>• Minor adjustments needed.</div>")
                                    changes_str = "".join(changes)
                                    scenarios_html.append(f"<div class='custom-card' style='border-color: #17B794;'><h4 style='color: #17B794; margin-top:0;'>Strategy {i+1}</h4>{changes_str}<div style='margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px;'><small style='color: #17B794;'><strong>Result:</strong> If implemented, the AI predicts the employee will <strong>STAY</strong>.</small></div></div>")
                                col_s1, col_s2, col_s3 = st.columns(3)
                                for i, html in enumerate(scenarios_html):
                                    with [col_s1, col_s2, col_s3][i]: st.markdown(html, unsafe_allow_html=True)
                        except Exception as e: st.error(f"Error generating strategies: {e}")

    # ====================================================================
    # PAGE: WHY THEY LEAVE
    # ====================================================================
    if page == "Why They Leave":
        st.header("🧠 Key Attrition Drivers"); st.write("---"); analyze_why_people_leave(df)
        with st.spinner("Analyzing model insights..."):
            try:
                shap_values, X_processed_df = get_shap_explanations(pipeline, df)
                if isinstance(shap_values, list): vals = np.abs(shap_values[1]).mean(0)
                else: vals = np.abs(shap_values).mean(0)
                feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance'])
                feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True); top_3 = feature_importance.head(3)
                c1, c2, c3 = st.columns(3); cols = [c1, c2, c3]
                for idx, row in enumerate(top_3.iterrows()):
                    feature = row[1]['Feature']; card_html = f"<div class='custom-card' style='text-align: center; height: 100%;'><h3 style='color: #17B794; margin-top: 0;'>{feature}</h3><p style='color: #c9d1d9; font-size: 0.9rem;'>Top {idx+1} Driver</p></div>"
                    with cols[idx]: st.markdown(card_html, unsafe_allow_html=True)
                with st.expander("🔧 Technical Deep Dive (SHAP)"):
                    fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False); st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)
                    fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False, plot_type='dot'); st.pyplot(fig1, bbox_inches='tight'); plt.close(fig1)
            except Exception as e: st.error(f"Could not generate SHAP plots: {e}")

    # ====================================================================
    # PAGE: BUDGET PLANNER
    # ====================================================================
    if page == "Budget Planner":
        st.markdown("<h1 style='margin-bottom: 5px;'>💰 Budget Planner</h1>", unsafe_allow_html=True)
        analyze_why_people_leave(df); st.markdown("---"); st.markdown("### 💰 Budget Optimization Tool")
        col1, col2 = st.columns([2, 1])
        with col1: budget = st.number_input("Total Retention Budget (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
        with col2: st.write("<br>", unsafe_allow_html=True); optimize_btn = st.button("🚀 Generate Plan", type="primary")
        if optimize_btn:
            results = plan_retention_budget(df, pipeline, budget)
            if results:
                selected_df, total_cost, total_savings = results
                st.markdown("<br>", unsafe_allow_html=True); st.success("✅ **Optimization Complete.**")
                m1, m2 = st.columns(2); m1.metric("Investment Needed", f"₹{total_cost:,.0f}"); m2.metric("Projected Savings", f"₹{total_savings:,.0f}", delta="ROI Positive")
                st.dataframe(selected_df.head(10), use_container_width=True)

    # ====================================================================
    # PAGE: AI ASSISTANT
    # ====================================================================
    if page == "AI Assistant":
        st.header("🤖 AI Assistant")
        with st.form("llm_form"):
            c1, c2 = st.columns(2)
            with c1: emp_name = st.text_input("Employee Name", value="Rahul Sharma"); emp_dept = st.text_input("Department", value="Sales")
            with c2: situation_input = st.selectbox("Situation?", ["Overworked", "Seeking Salary", "Low Morale"]); solution_input = st.text_input("Proposed Solution", value="Offer Flexible Hours")
            generate_btn = st.form_submit_button("🚀 Generate Email Draft")
            if generate_btn: run_groq_consultant(emp_name, emp_dept, situation_input, solution_input, "₹50,000")

    # ====================================================================
    # PAGE: AI RESEARCH LAB (FULL TABS RESTORED)
    # ====================================================================
    if page == "AI Research Lab":
        st.header("🧪 AI Research Lab")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Benchmarking", "🕵️ Anomaly Detection", "📊 Retention Matrix", "🎯 Ideal Candidate Profiler"])
        
        with tab1:
            if st.button("Run Benchmark", type="primary", key="run_benchmark"):
                with st.spinner("Training competing models..."):
                    y_pred_lgbm = pipeline.predict(X_test_cur); proba_lgbm = pipeline.predict_proba(X_test_cur)[:, 1]
                    rf_pipeline = Pipeline(steps=[('preprocessor', pipeline.named_steps['preprocessor']), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
                    rf_pipeline.fit(X_train_ref, y_test); y_pred_rf = rf_pipeline.predict(X_test_cur); proba_rf = rf_pipeline.predict_proba(X_test_cur)[:, 1]
                    lr_pipeline = Pipeline(steps=[('preprocessor', pipeline.named_steps['preprocessor']), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
                    lr_pipeline.fit(X_train_ref, y_test); y_pred_lr = lr_pipeline.predict(X_test_cur); proba_lr = lr_pipeline.predict_proba(X_test_cur)[:, 1]
                    metrics = {'Model': ['LightGBM', 'Random Forest', 'Logistic Regression'], 'Accuracy': [accuracy_score(y_test, y_pred_lgbm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)], 'ROC AUC': [roc_auc_score(y_test, proba_lgbm), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_lr)]}
                    st.dataframe(pd.DataFrame(metrics).style.highlight_max(axis=0, color='#17B794'), use_container_width=True)

        with tab2:
            X_all = df.drop('left', axis=1); y_true = df['left']; y_pred = pipeline.predict(X_all)
            happy_leavers_indices = np.where((y_pred == 0) & (y_true == 1))[0]; loyal_sufferers_indices = np.where((y_pred == 1) & (y_true == 0))[0]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🚪 Happy Leavers"); st.caption(f"Count: {len(happy_leavers_indices)}")
                if len(happy_leavers_indices) > 0: st.dataframe(df.iloc[happy_leavers_indices].head(), use_container_width=True)
                else: st.success("None found.")
            with col_b:
                st.markdown("### 🛡️ Loyal Sufferers"); st.caption(f"Count: {len(loyal_sufferers_indices)}")
                if len(loyal_sufferers_indices) > 0: st.dataframe(df.iloc[loyal_sufferers_indices].head(), use_container_width=True)
                else: st.success("None found.")

        with tab3:
            st.subheader("📊 Retention Priority Matrix")
            X_all = df.drop('left', axis=1); risk_probs = pipeline.predict_proba(X_all)[:, 1]
            if 'salary' in df.columns: salary_cost_map = {'low': 400000, 'medium': 600000, 'high': 900000}; replacement_costs = df['salary'].map(salary_cost_map) * 0.5
            else: replacement_costs = pd.Series([500000]*len(df))
            plot_data = pd.DataFrame({'Risk_Probability': risk_probs, 'Replacement_Cost': replacement_costs})
            risk_threshold = 0.5; cost_threshold = replacement_costs.median()
            def get_zone(row):
                if row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] >= cost_threshold: return "🔴 Critical Zone (Save Now)"
                elif row['Risk_Probability'] < risk_threshold and row['Replacement_Cost'] >= cost_threshold: return "🟡 Retain Zone (Keep Happy)"
                elif row['Risk_Probability'] >= risk_threshold and row['Replacement_Cost'] < cost_threshold: return "🟢 Outplacement Zone (Let Go)"
                else: return "⚪ Monitor Zone"
            plot_data['Zone'] = plot_data.apply(get_zone, axis=1)
            fig = px.scatter(plot_data, x='Risk_Probability', y='Replacement_Cost', color='Zone', color_discrete_map={"🔴 Critical Zone (Save Now)": "#FF4B4B", "🟡 Retain Zone (Keep Happy)": "#F59E0B", "🟢 Outplacement Zone (Let Go)": "#17B794", "⚪ Monitor Zone": "#9ca3af"}, title="Employee Prioritization Map", template="plotly_dark", height=600)
            fig.add_hline(y=cost_threshold, line_dash="dash", line_color="white", opacity=0.3); fig.add_vline(x=risk_threshold, line_dash="dash", line_color="white", opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("🎯 The 'Ideal Candidate' Profiler")
            # Dynamic definition of Superstar based on available columns
            if 'time_spend_company' in df.columns and 'last_evaluation' in df.columns:
                superstar_mask = (df['left'] == 0) & (df['time_spend_company'] > 4) & (df['last_evaluation'] > 0.8)
            else:
                # Fallback for global: just high performers who stayed
                st.info("Dataset missing specific tenure columns. Defining Superstars as top retained performers.")
                superstar_mask = (df['left'] == 0)
            df_superstars = df[superstar_mask]; df_average = df[(df['left'] == 0) & (~superstar_mask)]
            if len(df_superstars) < 5: st.warning("Not enough data to profile.")
            else:
                st.success(f"Analyzed {len(df_superstars)} Superstars vs {len(df_average)} Average.")
                metrics_to_compare = df.select_dtypes(include=np.number).columns.drop('left', errors='ignore').tolist()
                super_mean = df_superstars[metrics_to_compare].mean(); avg_mean = df_average[metrics_to_compare].mean()
                comparison_long = pd.DataFrame({'Metric': metrics_to_compare, 'Superstar': super_mean.values, 'Average Employee': avg_mean.values}).melt(id_vars='Metric', var_name='Group', value_name='Average Value')
                fig_compare = px.bar(comparison_long, x='Metric', y='Average Value', color='Group', barmode='group', title="Superstars vs. Average", template="plotly_dark", color_discrete_map={'Superstar': '#17B794', 'Average Employee': '#9ca3af'}, height=500)
                fig_compare.update_xaxes(title="", tickangle=45); st.plotly_chart(fig_compare, use_container_width=True)

    # ====================================================================
    # PAGE: STRATEGIC ROADMAP
    # ====================================================================
    if page == "Strategic Roadmap":
        st.header("🚀 Future Planning & Projections")
        avg_sat = df['satisfaction_level'].mean() if 'satisfaction_level' in df.columns else 0.5
        issues = []
        if avg_sat < 0.6: issues.append("Low Employee Satisfaction")
        else: issues.append("General Workforce Stabilization")
        issues_str = ", ".join(issues)
        st.markdown(f"<div class='custom-card'><h4 style='color: #17B794; margin-top: 0;'>🩺 AI Diagnostic Summary</h4><p style='color: #c9d1d9;'><strong style='color: #FF4B4B;'>➤ {issues_str}</strong></p></div>", unsafe_allow_html=True)
        
        if st.button("✍️ Draft My 6-Month HR Action Plan", type="primary"):
            try:
                api_key = st.secrets.get("GROQ_API_KEY", None)
                if api_key:
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.5)
                    template = "Create a 6-month HR roadmap to fix: {issues}. Use plain English."
                    chain = PromptTemplate.from_template(template) | llm | StrOutputParser()
                    st.markdown(f"<div class='llm-response'>{chain.invoke({'issues': issues_str})}</div>", unsafe_allow_html=True)
                else: st.warning("API Key missing.")
            except Exception as e: st.error(f"Error: {e}")

        st.markdown("---"); st.markdown("### 📈 12-Month Projection")
        col_f1, col_f2 = st.columns(2)
        with col_f1: intervention_efficacy = st.slider("If we take action, % of at-risk people we save:", 10, 50, 20, 5)
        with col_f2: natural_attrition_rate = st.slider("People who leave for personal reasons (%):", 0.5, 2.0, 1.0, 0.1)

        if st.button("📈 Show Me the 12-Month Projection", type="primary"):
            months = list(range(1, 13)); current_workforce = len(df)
            total_risk_score = pipeline.predict_proba(df.drop('left', axis=1))[:, 1].sum()
            monthly_leavers_no_action = total_risk_score / 12.0
            monthly_leavers_with_action = monthly_leavers_no_action * (1 - (intervention_efficacy / 100.0))
            forecast_bau = []; forecast_intervention = []; temp_bau = float(current_workforce); temp_int = float(current_workforce)
            for m in months:
                temp_bau = max(0, temp_bau - monthly_leavers_no_action - (temp_bau * (natural_attrition_rate / 100.0)))
                temp_int = max(0, temp_int - monthly_leavers_with_action - (temp_int * (natural_attrition_rate / 100.0)))
                forecast_bau.append(temp_bau); forecast_intervention.append(temp_int)
            
            forecast_df = pd.DataFrame({'Month': months, 'If We Do Nothing': forecast_bau, 'If We Follow the Plan': forecast_intervention}).melt(id_vars='Month', var_name='Scenario', value_name='Workforce Count')
            fig_forecast = px.line(forecast_df, x='Month', y='Workforce Count', color='Scenario', title="Projected Workforce Size", template="plotly_dark", markers=True, color_discrete_map={'If We Do Nothing': "#FF4B4B", 'If We Follow the Plan': "#17B794"})
            fig_forecast.update_layout(yaxis_title="Employee Headcount", xaxis=dict(dtick=1)); st.plotly_chart(fig_forecast, use_container_width=True)
            
            saved_employees = forecast_intervention[-1] - forecast_bau[-1]
            avg_salary = df['salary'].map({'low': 400000, 'medium': 600000, 'high': 900000}).mean() if 'salary' in df.columns else 500000
            total_money_saved = int(saved_employees) * (avg_salary * 0.5)
            
            col_sum_1, col_sum_2 = st.columns(2)
            with col_sum_1: st.metric("Employees Saved", f"{int(saved_employees)} People")
            with col_sum_2: st.metric("Costs Prevented", f"₹{total_money_saved:,.0f}")
            st.success(f"**Bottom Line:** We finish the year with **{int(forecast_intervention[-1])} employees** instead of **{int(forecast_bau[-1])}**.")

if __name__ == "__main__":
    main()
