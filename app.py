import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Telecom Customer Churn Prediction")

st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure in Months",
    min_value=0,
    max_value=100,
    value=12
)

offer = st.selectbox(
    "Offer",
    ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
)

internet_type = st.selectbox(
    "Internet Type",
    ["DSL", "Fiber Optic", "Cable"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-Month", "One Year", "Two Year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Bank Withdrawal", "Credit Card", "Mailed Check"]
)

monthly_charge = st.number_input(
    "Monthly Charge",
    min_value=0.0,
    max_value=500.0,
    value=75.0
)

satisfaction_score = st.slider(
    "Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

cltv = st.number_input(
    "CLTV",
    min_value=0,
    max_value=10000,
    value=5000
)

if st.button("Predict Churn"):

    # Encode categorical values
    gender = encoders["Gender"].transform([gender])[0]
    offer = encoders["Offer"].transform([offer])[0]
    internet_type = encoders["Internet Type"].transform([internet_type])[0]
    contract = encoders["Contract"].transform([contract])[0]
    payment_method = encoders["Payment Method"].transform([payment_method])[0]

    data = pd.DataFrame([[
        gender,
        age,
        tenure,
        offer,
        internet_type,
        contract,
        payment_method,
        monthly_charge,
        satisfaction_score,
        cltv
    ]], columns=[
        "Gender",
        "Age",
        "Tenure in Months",
        "Offer",
        "Internet Type",
        "Contract",
        "Payment Method",
        "Monthly Charge",
        "Satisfaction Score",
        "CLTV"
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"Churn Probability: {probability*100:.2f}%")