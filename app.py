import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Example inputs (adjust based on your features)
tenure = st.slider("Tenure", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create input dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "Contract": [contract],
    "InternetService": [internet_service]
})

# 🔥 IMPORTANT: apply same preprocessing
input_data = pd.get_dummies(input_data)

# Align with training columns
model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer likely to stay (Probability: {probability:.2f})")