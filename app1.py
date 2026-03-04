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
from sklearn.model_model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from time import sleep
from scipy.sparse import issparse

# --- Imports for Evaluation 1 (Logic Engine) ---
import dowhy
from dowhy import CausalModel
from scipy.optimize import milp, LinearConstraint, Bounds
# ====================================================================

# ====================================================================
# Visualization Functions
# ====================================================================
def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={"font": {"size": title_size, "family": "tahoma"}},
        hoverlabel={"bgcolor": "#000", "font_size": hover_font_size, "font_family": "arial"}
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
# Logic Engine Functions (HR Friendly Versions)
# ====================================================================

def analyze_why_people_leave(df):
    """
    HR Friendly Version: Insight Cards (No Graphs).
    """
    st.subheader("🔍 Why do people leave? (Root Cause Analysis)")
    st.write("Our AI has analyzed the data and ranked the **Top 3 Reasons** why employees quit. Here is what matters most:")
    
    # 1. Run Causal Logic (Hidden Math)
    df_causal = df.copy()
    salary_map = {'low': 1, 'medium': 2, 'high': 3}
    df_causal['salary_num'] = df_causal['salary'].map(salary_map)
    
    causal_graph = """digraph {
        salary_num -> satisfaction_level;
        satisfaction_level -> left;
        average_montly_hours -> left;
        number_project -> average_montly_hours;
    }"""
    
    df_model = df_causal[['salary_num', 'satisfaction_level', 'average_montly_hours', 'number_project', 'left']]
    
    # --- Calculate Effects (Logic) ---
    effects = {}
    
    # Salary Effect
    model_sal = CausalModel(data=df_model, treatment='salary_num', outcome='left', graph=causal_graph.replace('\n', ' '))
    est_sal = model_sal.estimate_effect(model_sal.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
    effects['Salary'] = abs(est_sal.value)
    
    # Satisfaction Effect
    model_sat = CausalModel(data=df_model, treatment='satisfaction_level', outcome='left', graph=causal_graph.replace('\n', ' '))
    est_sat = model_sat.estimate_effect(model_sat.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
    effects['Satisfaction'] = abs(est_sat.value)
    
    # Hours Effect
    model_hr = CausalModel(data=df_model, treatment='average_montly_hours', outcome='left', graph=causal_graph.replace('\n', ' '))
    est_hr = model_hr.estimate_effect(model_hr.identify_effect(proceed_when_unidentifiable=True), method_name="backdoor.linear_regression")
    effects['Overwork'] = abs(est_hr.value) * 10 # Scale up for comparison

    # --- Rank the Effects ---
    sorted_effects = sorted(effects.items(), key=lambda item: item[1], reverse=True)
    
    # Define the display logic
    def get_display_info(rank, factor, value):
        if rank == 1:
            color = "#FF4B4B" # Red
            status = "CRITICAL DRIVER"
            advice = "This is the #1 reason people leave. Focus here first."
        elif rank == 2:
            color = "#FFA500" # Orange
            status = "MAJOR FACTOR"
            advice = "Important to address, but secondary to the #1 driver."
        else:
            color = "#FFD700" # Yellow/Gold
            status = "MODERATE FACTOR"
            advice = "Monitor this, but it is not the main cause of attrition."
            
        return color, status, advice

    # --- Display Insight Cards ---
    c1, c2, c3 = st.columns(3)
    
    for idx, (col, factor_value) in enumerate(sorted_effects):
        color, status, advice = get_display_info(idx + 1, col, factor_value)
        
        with [c1, c2, c3][idx]:
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border: 1px solid {color}; text-align: center;">
                <h2 style="color: {color}; margin: 0;">#{idx+1} {col}</h2>
                <h4 style="color: white; margin: 10px 0;">{status}</h4>
                <p style="color: #ccc; font-size: 14px;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)

    # Validation (Hidden)
    with st.expander("Show Technical Validation"):
        st.write("AI Model Confidence Checks:")
        refute = model_sal.refute_estimate(model_sal.identify_effect(), est_sal, method_name="random_common_cause")
        st.table(refute.refutation_result)


def plan_retention_budget(df, pipeline, budget_limit):
    """
    HR Friendly Version: Budget Planner.
    """
    st.subheader("💰 Retention Budget Planner")
    st.write("We have identified employees who are at **High Risk** of leaving. Use this tool to see who is worth investing in to save the company money.")
    
    # Get predictions
    X = df.drop('left', axis=1)
    probas = pipeline.predict_proba(X)[:, 1]
    
    opt_df = df.copy()
    opt_df['attrition_risk'] = probas
    high_risk_df = opt_df[opt_df['attrition_risk'] > 0.5].copy()
    
    if len(high_risk_df) == 0:
        st.success("Great news! Our model predicts very few employees are currently at high risk of leaving.")
        return None

    # --- Economics (INR) ---
    salary_val_map = {'low': 400000, 'medium': 600000, 'high': 900000}
    high_risk_df['annual_salary'] = high_risk_df['salary'].map(salary_val_map)
    high_risk_df['replacement_cost'] = high_risk_df['annual_salary'] * 0.5
    high_risk_df['expected_loss'] = high_risk_df['attrition_risk'] * high_risk_df['replacement_cost']
    
    # Cost of Fixing them: 10% Raise
    high_risk_df['intervention_cost'] = high_risk_df['annual_salary'] * 0.10
    high_risk_df['net_savings'] = high_risk_df['expected_loss'] - high_risk_df['intervention_cost']

    candidates = high_risk_df[high_risk_df['net_savings'] > 0].copy()
    
    if len(candidates) == 0:
        st.warning("It is currently not cost-effective to offer raises to the high-risk group based on the calculated replacement costs.")
        return None

    # --- Optimization ---
    n = len(candidates)
    c = -candidates['net_savings'].values 
    A = np.array([candidates['intervention_cost'].values])
    b = np.array([budget_limit])
    integrality = np.ones(n)
    
    with st.spinner("Calculating the best employees to invest in..."):
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
        st.error("The budget provided is too low to retain any high-risk employees effectively.")
        return None

# ====================================================================
# Main App Function
# ====================================================================
def main():
    st.set_page_config(
        page_title="Employee Retention AI", page_icon="🤖", layout="wide"
    )
    warnings.simplefilter(action='ignore', category=FutureWarning)

    @st.cache_data
    def load_data_and_train_model():
        df = pd.read_csv('HR_comma_sep.csv')
        df_original = df.copy()
        df_train = df.drop_duplicates().reset_index(drop=True)
        X = df_train.drop('left', axis=1)
        y = df_train['left']
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=np.number).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
        best_params = {'n_estimators': 1888, 'learning_rate': 0.019, 'num_leaves': 22, 'max_depth': 11, 'random_state': 42}
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lgb.LGBMClassifier(**best_params))])
        final_pipeline.fit(X, y)
        return final_pipeline, df_original

    pipeline, df = load_data_and_train_model()

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
        if employee_data['satisfaction_level'] <= 0.45: strategies.append("Conduct a one-on-one meeting to discuss job satisfaction and well-being.")
        if employee_data['number_project'] <= 2: strategies.append("Employee may be under-utilized. Discuss career aspirations and find new projects.")
        if employee_data['number_project'] >= 6: strategies.append("Employee is at high risk of burnout. Assess workload and prioritize projects.")
        if employee_data['time_spend_company'] >= 4 and employee_data['promotion_last_5years'] == 0: strategies.append("Develop a clear career path and discuss opportunities for growth.")
        if employee_data['last_evaluation'] >= 0.8 and employee_data['satisfaction_level'] < 0.6: strategies.append("This is a high-performer but may be unhappy. Acknowledge contributions with rewards.")
        if not strategies: strategies.append("No specific high-risk factors detected, but continue to monitor engagement.")
        return strategies

    st.markdown(
        """<style>
        .main { text-align: center; }
        .st-emotion-cache-16txtl3 h1 { font: bold 29px arial; text-align: center; margin-bottom: 15px; }
        div[data-testid=stSidebarContent] { background-color: #111; border-right: 4px solid #222; padding: 8px!important; }
        div[data-testid=stFormSubmitButton]> button{ width: 100%; background-color: #111; border: 2px solid #17B794; padding: 18px; border-radius: 30px; opacity: 0.8; }
        div[data-testid=stFormSubmitButton]  p{ font-weight: bold; font-size : 20px; }
        div[data-testid=stFormSubmitButton]> button:hover{ opacity: 1; border: 2px solid #17B794; color: #fff; }
        </style>""", unsafe_allow_html=True
    )

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "white", "font-size": "16px"},
        "nav-link": {"color": "#fff", "font-size": "18px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
        "nav-link-selected": {"background-color": "#17B794", "font-size": "15px"},
    }

    with st.sidebar:
        st.title(":green[AI Retention] Assistant")
        st.title(":green[Develop by]-Nisarg Rathod")
        # CHANGED MENU ITEM
        page = option_menu(
            menu_title=None,
            options=['Home', 'Vizualizations', 'Prediction', 'Explain Predictions', 'Retention Strategy'],  
            icons=['diagram-3-fill', 'bar-chart-line-fill', "graph-up-arrow", 'lightbulb-fill', 'currency-rupee'], 
            menu_icon="cast", default_index=0, styles=side_bar_options_style
        )

    # ====================================================================
    # Pages
    # ====================================================================
    if page == "Home":
        st.header('Employee Retention Classification 👨‍💼')
        st.dataframe(df.head(100), use_container_width=True)
        st.write("***"); st.subheader("Data Summary Overview 🧐"); st.table(df.describe().T)

    if page == "Vizualizations":
        st.header("Data Vizualizations 📉")
        st.subheader("Numerical Data Distributions")
        create_vizualization(df, viz_type="box", data_type="number")
        st.subheader("Categorical Data Distributions")
        create_vizualization(df, viz_type="bar", data_type="object")
        st.subheader("Low-Cardinality Feature Distributions")
        create_vizualization(df, viz_type="pie")
        st.subheader("Feature Correlation Heatmap")
        st.plotly_chart(create_heat_map(df), use_container_width=True)

    if page == "Prediction":
        st.header("🎯 Predict Attrition & Get Recommendations")
        with st.form("Predict_value_form"):
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
            predict_button = st.form_submit_button(label='Get Prediction & Advice')

        if predict_button:
            satisfaction_level = satisfaction_map[satisfaction_text]; last_evaluation = evaluation_map[evaluation_text]
            Work_accident = 1 if work_accident_text == 'Yes' else 0; promotion_last_5years = 1 if promotion_text == 'Yes' else 0
            input_data = {'satisfaction_level': satisfaction_level, 'last_evaluation': last_evaluation,
                          'number_project': number_project, 'average_montly_hours': average_montly_hours,
                          'time_spend_company': time_spend_company, 'Work_accident': Work_accident,
                          'promotion_last_5years': promotion_last_5years, 'Department': Department, 'salary': salary}
            input_df = pd.DataFrame([input_data])
            with st.spinner('Analyzing Employee Profile...'):
                sleep(1); prediction = pipeline.predict(input_df)[0]; prediction_probas = pipeline.predict_proba(input_df)[0]
                st.subheader('Prediction Result'); st.write("---")
                pred_col, stay_prob_col, leave_prob_col = st.columns(3)
                with pred_col:
                    if prediction == 0: st.subheader("Employee is Likely to"); st.subheader(":green[STAY]")
                    else: st.subheader("Employee is Likely to"); st.subheader(":red[LEAVE]")
                with stay_prob_col: st.subheader("Probability to Stay"); st.subheader(f"{prediction_probas[0]:.2%}")
                with leave_prob_col: st.subheader("Probability to Leave"); st.subheader(f"{prediction_probas[1]:.2%}")
                st.write("---")
                if prediction == 1:
                    st.subheader('💡 Recommended Retention Strategies:')
                    recommendations = get_retention_strategies(input_df)
                    for i, rec in enumerate(recommendations, 1): st.info(f'{i}. {rec}')

    if page == "Explain Predictions":
        # CHANGED: HR Friendly Title and Intro
        st.header("🧠 AI Model Insights (Manager's Summary)")
        st.write("You don't need to be a data scientist to understand our AI. Here are the **top factors** the model looks at to predict if someone will leave, and what you should do about them.")
        
        st.write("---")
        
        with st.spinner("Analyzing model insights..."):
            shap_values, X_processed_df = get_shap_explanations(pipeline, df)
            
            # --- HR LOGIC: Process SHAP values into Insights ---
            # Calculate mean absolute SHAP values to find importance
            # shap_values[1] is usually for class 1 (Leaving) in binary classification
            if isinstance(shap_values, list):
                vals = np.abs(shap_values[1]).mean(0)
            else:
                vals = np.abs(shap_values).mean(0)
            
            feature_importance = pd.DataFrame(list(zip(X_processed_df.columns, vals)), columns=['Feature','Importance'])
            feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
            top_3 = feature_importance.head(3)

            # --- Insight Cards Logic ---
            def get_feature_advice(feature_name):
                # Simplified mapping based on the feature name
                if 'satisfaction' in feature_name.lower():
                    return "Employee Morale", "Conduct regular engagement surveys and 1-on-1s."
                elif 'project' in feature_name.lower():
                    return "Workload Balance", "Review project allocation to prevent burnout."
                elif 'time' in feature_name.lower() or 'tenure' in feature_name.lower():
                    return "Tenure", "Watch for turnover at the 3-5 year mark."
                elif 'salary' in feature_name.lower():
                    return "Compensation", "Review market rates annually."
                else:
                    return "Performance Factor", "Keep track of evaluation scores."

            c1, c2, c3 = st.columns(3)
            
            cols = [c1, c2, c3]
            for idx, row in enumerate(top_3.iterrows()):
                feature = row[1]['Feature']
                advice_title, advice_text = get_feature_advice(feature)
                color = "#17B794" # Green for "Safe/Info"
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border: 1px solid {color}; text-align: center;">
                        <h3 style="color: {color}; margin: 0;">{advice_title}</h3>
                        <h4 style="color: white; margin: 5px 0;">Key Driver</h4>
                        <p style="color: #ccc; font-size: 14px;">{advice_text}</p>
                        <p style="color: #888; font-size: 12px; margin-top: 10px;">(Based on: {feature})</p>
                    </div>
                    """, unsafe_allow_html=True)

        # --- Hide Technical Details in Expander ---
        with st.expander("🔧 For Data Scientists: Deep Dive into SHAP Values"):
            st.write("Below are the raw SHAP (SHapley Additive exPlanations) plots.")
            st.caption("Bar Chart: Global Feature Importance.")
            fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, plot_type='bar', show=False)
            st.pyplot(fig2, bbox_inches='tight'); plt.close(fig2)
            
            st.write("---")
            st.caption("Beeswarm Plot: How feature values affect the model output.")
            with st.expander("How to Read This Chart"):
                st.markdown("""- **Each dot** is an employee.\n- **Right side** = Increases chance of leaving.\n- **Left side** = Decreases chance.\n- **Red** = High value.\n- **Blue** = Low value.""")
            fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X_processed_df, show=False, plot_type='dot')
            st.pyplot(fig1, bbox_inches='tight'); plt.close(fig1)

    # ====================================================================
    # Page: Retention Strategy (HR Friendly)
    # ====================================================================
    if page == "Retention Strategy":
        st.header("🧠 Retention Strategy & Budget")
        st.markdown("Welcome to the Strategy Center. Here, we don't just predict *who* might leave, we help you decide **how to stop them** in the most cost-effective way.")
        
        st.markdown("---")
        
        # Section 1: Why they leave
        analyze_why_people_leave(df)
        
        st.markdown("---")
        
        # Section 2: Budget Planner
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("### 💼 Plan Your Budget")
            budget = st.number_input("How much budget can you allocate for raises? (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000, help="Enter the total amount in Rupees you are willing to spend on retention efforts.")
            
        with col2:
            st.write("### ")
            optimize_btn = st.button("Generate Retention Plan", type="primary", help="Click to calculate the best use of your budget.")
            
        if optimize_btn:
            results = plan_retention_budget(df, pipeline, budget)
            
            if results:
                selected_df, total_cost, total_savings = results
                
                st.success("✅ **Plan Generated Successfully!**")
                st.write("Based on your budget, here is the most cost-effective group to target with raises. Targeting these employees maximizes your savings on replacement costs.")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Your Budget", f"₹{budget:,.0f}")
                m2.metric("Required Investment", f"₹{total_cost:,.0f}", delta=f"{(total_cost/budget)*100:.1f}% of Budget")
                m3.metric("Estimated Savings", f"₹{total_savings:,.0f}", delta="Money Saved on Hiring")
                
                st.write("### 📋 Recommended Action List")
                st.caption("These are the specific employees who should receive a 10% raise. It is cheaper to keep them than to replace them.")
                
                display_cols = ['Department', 'salary', 'satisfaction_level', 'number_project', 
                                'attrition_risk', 'intervention_cost', 'net_savings']
                
                display_df = selected_df[display_cols].copy()
                display_df.columns = ['Department', 'Salary Tier', 'Satisfaction', 'Projects', 
                                     'Risk of Leaving', 'Cost to Retain (Raise)', 'Savings if Retained']
                
                display_df['Cost to Retain (Raise)'] = display_df['Cost to Retain (Raise)'].apply(lambda x: f"₹{x:,.0f}")
                display_df['Savings if Retained'] = display_df['Savings if Retained'].apply(lambda x: f"₹{x:,.0f}")
                display_df['Risk of Leaving'] = (display_df['Risk of Leaving'] * 100).apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()
