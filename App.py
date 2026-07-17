import streamlit as st
import pandas as pd
import joblib

# ✅ Load pipeline model
model = joblib.load("fraud_pipeline.pkl")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💳 Fraud Detection Dashboard</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📝 Transaction Form", "📊 Prediction Result"])

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

    # Save to session state so tab2 can access it
    st.session_state["input_df"] = input_df

with tab2:
    st.header("Prediction Result")

    if "input_df" in st.session_state and st.button("Predict Fraud"):
        input_df = st.session_state["input_df"]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

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
            import plotly.graph_objects as go
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
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=(1 - probability) * 100,
                title={'text': "Legit Confidence (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
