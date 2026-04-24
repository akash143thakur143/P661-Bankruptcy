import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
model = joblib.load("bankruptcy_model.pkl")

st.title("🏢 Bankruptcy Prevention Predictor")
st.write("Enter company risk details to predict bankruptcy probability.")

# =========================
# USER INPUT
# =========================

st.subheader("Company Risk Profile")

col1, col2 = st.columns(2)

with col1:
    industrial_risk = st.selectbox("Industrial Risk", [0.0, 0.5, 1.0],
                                    format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])
    management_risk = st.selectbox("Management Risk", [0.0, 0.5, 1.0],
                                    format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])
    financial_flexibility = st.selectbox("Financial Flexibility", [0.0, 0.5, 1.0],
                                          format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])

with col2:
    credibility = st.selectbox("Credibility", [0.0, 0.5, 1.0],
                                format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])
    competitiveness = st.selectbox("Competitiveness", [0.0, 0.5, 1.0],
                                    format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])
    operating_risk = st.selectbox("Operating Risk", [0.0, 0.5, 1.0],
                                   format_func=lambda x: {0.0:"Low",0.5:"Medium",1.0:"High"}[x])

# =========================
# PREDICT
# =========================

if st.button("Predict"):

    input_data = np.array([[
        industrial_risk,
        management_risk,
        financial_flexibility,
        credibility,
        competitiveness,
        operating_risk
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Bankruptcy Risk Detected")
    else:
        st.success(f"✅ Non-Bankruptcy")

    # Probability Bar
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        "Class": ["Non-Bankruptcy", "Bankruptcy"],
        "Probability": [probability[0], probability[1]]
    })

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(prob_df["Class"], prob_df["Probability"], color=["green", "red"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Confidence")
    for i, v in enumerate(prob_df["Probability"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    st.pyplot(fig)

    # Risk Profile Chart
    st.subheader("Risk Profile")
    features = ["Industrial Risk", "Management Risk", "Financial Flexibility",
                "Credibility", "Competitiveness", "Operating Risk"]
    values = [industrial_risk, management_risk, financial_flexibility,
              credibility, competitiveness, operating_risk]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["red" if v >= 0.5 else "green" for v in values]
    ax2.bar(features, values, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Risk Level")
    ax2.set_title("Company Risk Profile")
    plt.xticks(rotation=15)
    st.pyplot(fig2)
