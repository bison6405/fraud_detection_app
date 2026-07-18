import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ✅ Load pipeline model
model = joblib.load("fraud_pipeline.pkl")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💳 Fraud Detection Dashboard</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📝 Transaction Form", "📊 Prediction Result"])

# Ensure session state key exists
if "input_df" not in st.session_state:
    st.session_state["input_df"] = None
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "probability" not in st.session_state:
    st.session_state["probability"] = None

with tab1:
    st.header("Enter Transaction Details")

    # Use columns for cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hours since start)", min_value=0, step=1)
        tx_type = st.radio("Transaction Type", ["CASH_OUT","PAYMENT","TRANSFER","DEBIT"])
        amount = st.number_input("💵 Transaction Amount", min_value=0.0, step=10.0)
    with col2:
        flagged = st.selectbox("Is Flagged Fraud?", [0,1])
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=10.0)
        newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=10.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=10.0)
        newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=10.0)

    # Build input dictionary
    input_data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": flagged,
        "type_CASH_OUT": 1 if tx_type=="CASH_OUT" else 0,
        "type_DEBIT": 1 if tx_type=="DEBIT" else 0,
        "type_PAYMENT": 1 if tx_type=="PAYMENT" else 0,
        "type_TRANSFER": 1 if tx_type=="TRANSFER" else 0
    }

    expected_order = [
        "step","amount","oldbalanceOrg","newbalanceOrig",
        "oldbalanceDest","newbalanceDest","isFlaggedFraud",
        "type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"
    ]

    input_df = pd.DataFrame([input_data], columns=expected_order)

    # Prediction button now in Tab1
    if st.button("Predict Fraud"):
        st.session_state["input_df"] = input_df
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.success("Prediction complete! Switch to the 'Prediction Result' tab to view details.")

with tab2:
    st.header("Prediction Result")

    if st.session_state["input_df"] is not None and st.session_state["prediction"] is not None:
        input_df = st.session_state["input_df"]
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]

        # Show transaction summary
        st.markdown("#### Transaction Summary")
        st.dataframe(input_df, use_container_width=True)

        # Show quick metrics
        st.metric("💵 Transaction Amount", f"${input_df['amount'][0]:,.2f}")
        st.metric("🏦 Origin Balance", f"${input_df['oldbalanceOrg'][0]:,.2f}")
        st.metric("📥 Destination Balance", f"${input_df['oldbalanceDest'][0]:,.2f}")

        # Prediction result with styled alert + progress bar
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-box fraud-alert'>🚨 Fraudulent Transaction Detected! (Confidence: {probability:.2%})</div>",
                unsafe_allow_html=True
            )
            st.progress(int(probability * 100))

            # Gauge chart for fraud probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Fraud Risk (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(
                f"<div class='prediction-box legit-alert'>✅ Transaction is Legitimate (Confidence: {1 - probability:.2%})</div>",
                unsafe_allow_html=True
            )
            st.progress(int((1 - probability) * 100))

            # Gauge chart for legit confidence
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=(1 - probability) * 100,
                title={'text': "Legit Confidence (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
