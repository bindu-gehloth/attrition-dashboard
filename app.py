import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Attrition Risk Dashboard", layout="wide")

st.title("Employee Attrition Risk Dashboard")
st.caption("Predictive risk scoring and explainability for workforce retention planning - Palo Alto Networks")

@st.cache_data
def load_data():
    return pd.read_csv("full_employee_data.csv")

@st.cache_resource
def load_model_files():
    model = joblib.load("xgb_attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, explainer, feature_columns

df = load_data()
model, scaler, explainer, feature_columns = load_model_files()

company_avg_probability = df["AttritionProbability"].mean()

st.sidebar.header("Filter Options")

departments = sorted(df["Department"].unique())
selected_departments = st.sidebar.multiselect("Department", departments, default=departments)

roles = sorted(df["JobRole"].unique())
selected_roles = st.sidebar.multiselect("Job Role", roles, default=roles)

risk_categories = ["Low Risk", "Medium Risk", "High Risk"]
selected_risk = st.sidebar.multiselect("Risk Category", risk_categories, default=risk_categories)

risk_threshold = st.sidebar.slider("Minimum attrition probability to display (%)", 0, 100, 0, step=5)

with st.sidebar.expander("How to use this dashboard"):
    st.write("""
    - Department / Job Role: narrow the view to specific teams
    - Risk Category: show only Low, Medium, or High risk employees
    - Minimum probability: hide employees below a chosen risk level
    - Employee Profile tab: look up one person and see their personal risk drivers
    - Explainability tab: see company-wide drivers and test what-if scenarios
    """)

filtered_df = df[
    (df["Department"].isin(selected_departments)) &
    (df["JobRole"].isin(selected_roles)) &
    (df["RiskCategory"].isin(selected_risk)) &
    (df["AttritionProbability"] * 100 >= risk_threshold)
]

total_employees = len(filtered_df)
high_risk_count = (filtered_df["RiskCategory"] == "High Risk").sum()
avg_probability = filtered_df["AttritionProbability"].mean() if total_employees > 0 else 0
high_risk_pct = (high_risk_count / total_employees * 100) if total_employees > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees Shown", f"{total_employees:,}")
col2.metric("High Risk Employees", f"{high_risk_count:,}", f"{high_risk_pct:.1f}% of shown group")
col3.metric(
    "Avg. Attrition Probability",
    f"{avg_probability*100:.1f}%",
    delta=f"{(avg_probability - company_avg_probability)*100:.1f} pts vs. company avg",
    delta_color="inverse"
)
col4.metric("Departments Shown", f"{filtered_df['Department'].nunique()}")

st.divider()

tab_overview, tab_department, tab_profile, tab_explain, tab_about = st.tabs(
    ["Overview", "Department View", "Employee Profile", "Explainability", "About This Analysis"]
)

with tab_overview:
    st.subheader("Risk Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        risk_counts = (
            filtered_df["RiskCategory"].value_counts()
            .reindex(["Low Risk", "Medium Risk", "High Risk"])
            .fillna(0).reset_index()
        )
        risk_counts.columns = ["Risk Category", "Count"]

        fig_risk = px.bar(
            risk_counts, x="Risk Category", y="Count", color="Risk Category",
            color_discrete_map={"Low Risk": "#4C78A8", "Medium Risk": "#F2A93B", "High Risk": "#D64545"},
            title="Employees by Risk Category"
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        # no order dates in this dataset, so tenure is used as a stand-in for a time trend
        bins = [0, 2, 5, 10, 20, 100]
        labels = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"]
        tenure_group = pd.cut(filtered_df["YearsAtCompany"], bins=bins, labels=labels)
        tenure_trend = filtered_df.groupby(tenure_group, observed=True)["AttritionProbability"].mean().reset_index()
        tenure_trend.columns = ["Tenure Group", "Avg Probability"]

        fig_tenure = px.line(tenure_trend, x="Tenure Group", y="Avg Probability", markers=True,
                              title="Attrition Risk by Tenure")
        fig_tenure.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig_tenure, use_container_width=True)

    if total_employees > 0:
        st.info(
            f"{high_risk_count} employees ({high_risk_pct:.1f}% of the current view) are flagged High Risk "
            f"and would benefit from prioritized retention conversations."
        )
    else:
        st.warning("No employees match the current filter selection. Try widening your filters.")

    st.subheader("Export")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered employee data as CSV", data=csv_data,
                        file_name="filtered_employee_risk.csv", mime="text/csv")

with tab_department:
    st.subheader("Attrition Risk by Department")

    if total_employees > 0:
        dept_summary = (
            filtered_df.groupby("Department")
            .agg(Employees=("EmployeeID", "count"),
                 AvgProbability=("AttritionProbability", "mean"),
                 HighRiskCount=("RiskCategory", lambda x: (x == "High Risk").sum()))
            .reset_index().sort_values("AvgProbability", ascending=False)
        )

        fig_dept = px.bar(dept_summary, x="AvgProbability", y="Department", orientation="h",
                           color="AvgProbability", color_continuous_scale="Reds",
                           title="Average Attrition Probability by Department",
                           labels={"AvgProbability": "Average Attrition Probability"})
        fig_dept.update_xaxes(tickformat=".0%")
        fig_dept.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_dept, use_container_width=True)

        st.dataframe(dept_summary.style.format({"AvgProbability": "{:.1%}"}), use_container_width=True)

        st.subheader("By Job Role")
        role_summary = (
            filtered_df.groupby("JobRole")
            .agg(Employees=("EmployeeID", "count"), AvgProbability=("AttritionProbability", "mean"))
            .reset_index().sort_values("AvgProbability", ascending=False)
        )

        fig_role = px.bar(role_summary, x="AvgProbability", y="JobRole", orientation="h",
                           color="AvgProbability", color_continuous_scale="Reds",
                           title="Average Attrition Probability by Job Role",
                           labels={"AvgProbability": "Average Attrition Probability"})
        fig_role.update_xaxes(tickformat=".0%")
        fig_role.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_role, use_container_width=True)

        worst_dept = dept_summary.iloc[0]
        best_dept = dept_summary.iloc[-1]
        st.info(
            f"{worst_dept['Department']} carries the highest average risk ({worst_dept['AvgProbability']*100:.1f}%) "
            f"with {int(worst_dept['HighRiskCount'])} employees flagged High Risk, while {best_dept['Department']} "
            f"is the most stable ({best_dept['AvgProbability']*100:.1f}%)."
        )

        dept_csv = dept_summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download department summary as CSV", data=dept_csv,
                            file_name="department_risk_summary.csv", mime="text/csv")
    else:
        st.warning("No data matches the current filter selection.")

with tab_profile:
    st.subheader("Individual Employee Lookup")

    if total_employees > 0:
        employee_id = st.selectbox("Select Employee ID", filtered_df["EmployeeID"].tolist())
        employee_row = df[df["EmployeeID"] == employee_id].iloc[0]

        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Attrition Probability", f"{employee_row['AttritionProbability']*100:.1f}%",
                      delta=f"{(employee_row['AttritionProbability'] - company_avg_probability)*100:.1f} pts vs. company avg",
                      delta_color="inverse")
        col_y.metric("Risk Category", employee_row["RiskCategory"])
        col_z.metric("Department", employee_row["Department"])

        st.write(f"Job Role: {employee_row['JobRole']}")
        st.write(f"Age: {int(employee_row['Age'])}")
        st.write(f"Years at Company: {int(employee_row['YearsAtCompany'])}")

        employee_features = employee_row[feature_columns].values.reshape(1, -1)
        employee_scaled = scaler.transform(employee_features)
        shap_vals = explainer.shap_values(employee_scaled)[0]

        reason_df = pd.DataFrame({"Feature": feature_columns, "Contribution": shap_vals}).sort_values(
            "Contribution", ascending=False
        )
        top_reasons = reason_df.head(5)

        st.subheader("Why this employee was flagged")
        fig_reason = px.bar(top_reasons.sort_values("Contribution"), x="Contribution", y="Feature",
                             orientation="h", title="Top Factors Increasing This Employee's Risk",
                             color="Contribution", color_continuous_scale="Reds")
        st.plotly_chart(fig_reason, use_container_width=True)

        st.write("In plain terms, this employee's risk is driven mainly by:")
        for _, row in top_reasons.head(3).iterrows():
            st.write(f"- {row['Feature']}")
    else:
        st.warning("No employees match the current filter selection.")

with tab_explain:
    st.subheader("Global Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig_importance = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature",
                             orientation="h", title="Top 10 Drivers of Attrition (Company-Wide)",
                             color="Importance", color_continuous_scale="Blues")
    st.plotly_chart(fig_importance, use_container_width=True)

    st.divider()
    st.subheader("What-If Scenario")
    st.write("Adjust an employee's OverTime and Work-Life Balance to see how their predicted risk would change.")

    if total_employees > 0:
        whatif_id = st.selectbox("Select Employee ID for scenario testing", filtered_df["EmployeeID"].tolist(),
                                  key="whatif_selector")
        base_row = df[df["EmployeeID"] == whatif_id].iloc[0]

        new_overtime = st.radio("OverTime", ["Yes", "No"], index=0 if base_row["OverTime"] == 1 else 1)
        new_wlb = st.slider("Work-Life Balance (1=Poor, 4=Excellent)", 1, 4, int(base_row["WorkLifeBalance"]))

        scenario_row = base_row.copy()
        scenario_row["OverTime"] = 1 if new_overtime == "Yes" else 0
        scenario_row["WorkLifeBalance"] = new_wlb

        scenario_features = scenario_row[feature_columns].values.reshape(1, -1)
        scenario_scaled = scaler.transform(scenario_features)
        new_probability = model.predict_proba(scenario_scaled)[0, 1]
        original_probability = base_row["AttritionProbability"]

        col_p, col_q = st.columns(2)
        col_p.metric("Original Probability", f"{original_probability*100:.1f}%")
        col_q.metric("Scenario Probability", f"{new_probability*100:.1f}%",
                      delta=f"{(new_probability - original_probability)*100:.1f} pts", delta_color="inverse")
    else:
        st.warning("No employees match the current filter selection.")

with tab_about:
    st.subheader("About This Dashboard")
    st.write(f"""
    This dashboard analyzes {len(df):,} employee records at Palo Alto Networks to identify
    attrition risk and its key drivers.

    Dataset was fully populated with no missing values. Class imbalance (roughly 16% attrition)
    was handled with SMOTE during training, applied only to the training set to keep evaluation realistic.

    Final model: XGBoost, chosen for the best balance of Precision, F1-Score, and ROC-AUC among
    the three models tested (Logistic Regression, Random Forest, XGBoost).

    Risk categories: Low Risk (<30%), Medium Risk (30-60%), High Risk (>60%), based on predicted
    attrition probability. Individual reason codes are generated live using SHAP.

    Built by: Bindu
    """)

st.divider()
st.caption("Dashboard built as part of a data science internship project for Palo Alto Networks.")