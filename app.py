
import streamlit as st
import pandas as pd
import joblib

# ACT LAYER: STREAMLIT UI

st.set_page_config(
    page_title="Automated Approval Agent",
    layout="wide"
)

st.title("Health Insurance AI: Automated Approval Agent")

#1. load model and system-ready dataset
try:
    model = joblib.load("medical_cost_model.pkl")
    df = pd.read_csv("system_ready_data.csv")
    st.success("System Online")
except FileNotFoundError:
    st.error(
        "SYSTEM FAILURE: Required files missing. "
        "Please run these commands first: "
        "python clean_data.py, python feature_selection.py, python train_model.py"
    )
    st.stop()

# Check required columns
required_columns = [
    "Age",
    "BMI",
    "Dependents",
    "Smoker",
    "Diabetes",
    "Sex",
    "Hereditary_Diseases",
    "Medical_Cost"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns in system_ready_data.csv: {missing_columns}")
    st.stop()

st.divider()

col1, col2 = st.columns([1, 1.5])


with col1:
    st.subheader("Project Overview")

    st.write("""
    This AI agent predicts the expected insurance claim cost of a health insurance applicant.
    Based on the predicted cost, the system automatically decides how the application should be processed.
    """)

    st.write("### SENSE Layer")
    st.write("""
    - Data cleaning
    - Feature selection
    - Correlation analysis
    - Prepare system-ready dataset
    """)

    st.write("### THINK Layer")
    st.write("""
    - Train regression models
    - Evaluate MAE, MSE, RMSE, and R2
    - Save the best model using joblib
    """)

    st.write("### ACT Layer")
    st.write("""
    - User input form
    - Model prediction
    - Autonomous decision
    """)

    st.write("### Final Selected Features")
    st.write("""
    - Age
    - Sex
    - BMI
    - Dependents
    - Smoker
    - Diabetes
    - Hereditary Diseases
    """)

    st.write("### Removed Features")
    st.write("""
    - Weight: removed because BMI already represents body-weight health risk
    - Blood Pressure: removed because correlation was weak
    - Regular Exercise: removed because relationship with cost was weak
    - City: removed to avoid location-based bias
    - Job Title: removed to avoid occupation-based bias
    """)

    st.write("### Dataset Summary")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.write("Average Medical Cost:")
    st.write(f"${df['Medical_Cost'].mean():,.2f}")

    st.write("Highest Medical Cost:")
    st.write(f"${df['Medical_Cost'].max():,.2f}")

with col2:
    st.subheader("Applicant Medical Cost Prediction")

    st.write("Enter applicant information below.")

    with st.form("prediction_form"):

        age = st.slider(
            "Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median())
        )

        sex = st.selectbox(
            "Sex",
            sorted(df["Sex"].unique())
        )

        bmi = st.slider(
            "BMI",
            min_value=float(df["BMI"].min()),
            max_value=float(df["BMI"].max()),
            value=float(df["BMI"].median()),
            step=0.1
        )

        dependents = st.slider(
            "Number of Dependents",
            min_value=int(df["Dependents"].min()),
            max_value=int(df["Dependents"].max()),
            value=int(df["Dependents"].median())
        )

        smoker_choice = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )
        smoker = 1 if smoker_choice == "Yes" else 0

        diabetes_choice = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )
        diabetes = 1 if diabetes_choice == "Yes" else 0

        hereditary_diseases = st.selectbox(
            "Hereditary Diseases",
            sorted(df["Hereditary_Diseases"].unique())
        )

        submitted = st.form_submit_button("Predict and Decide")

    # PREDICTION AND AGENT ACTION
    if submitted:

        input_data = pd.DataFrame({
            "Age": [age],
            "BMI": [bmi],
            "Dependents": [dependents],
            "Smoker": [smoker],
            "Diabetes": [diabetes],
            "Sex": [sex],
            "Hereditary_Diseases": [hereditary_diseases]
        })

        predicted_cost = model.predict(input_data)[0]

        st.subheader("Output: Model Prediction")

        st.metric(
            label="Predicted Medical Claim Cost",
            value=f"${predicted_cost:,.2f}"
        )

        st.subheader("Agent Action")

        if predicted_cost < 10000:
            st.success("AUTO-APPROVED: Low expected medical claim cost.")

        elif predicted_cost < 30000:
            st.warning("FLAGGED FOR MANUAL ACTUARY REVIEW: Medium expected medical claim cost.")

        else:
            st.error("APPLICATION REJECTED - HIGH RISK: High expected medical claim cost.")

        st.info(
            "This is a prototype AI agent. In a real insurance company, "
            "manual review is required before making final approval decisions."
        )