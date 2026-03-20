import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn based on input features.")

st.sidebar.header("About")
st.sidebar.markdown("""
**Project:** Customer Churn Prediction  
**Model:** XGBoost  

This app helps identify customers likely to leave.
""")

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "Contract": [contract],
    "InternetService": [internet_service]
})

input_data = pd.get_dummies(input_data)

model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

st.header("🔍 Prediction")

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" High Risk of Churn ({probability:.2%})")
    else:
        st.success(f" Low Risk of Churn ({probability:.2%})")

    st.progress(float(probability))

st.header("Feature Importance")

importance = model.feature_importances_
features = model.feature_names_in_

fig, ax = plt.subplots()
ax.barh(features[:10], importance[:10])
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)

st.header("Key Insights")

st.markdown("""
- Customers with **month-to-month contracts** churn more  
- Higher **monthly charges increase churn risk**  
- Longer **tenure reduces churn probability**  
""")

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

