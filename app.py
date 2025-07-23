import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Set page title
st.set_page_config(page_title="Student Score Predictor", layout="wide")

# Title at the top (centered)
st.markdown("<h2 style='text-align: center;'>Score Prediction</h2>", unsafe_allow_html=True)

# Create two columns (left for input, right for output)
col1, col2 = st.columns(2)

# -------- LEFT SIDE: Inputs --------
with col1:
    st.markdown("### Enter Student Details")
    study_time = st.slider(" Study Time (hours)", 0, 10, 3)
    attendance = st.slider(" Attendance (%)", 0, 100, 83)
    gender = st.selectbox(" Gender", ["Female", "Male"])

# -------- RIGHT SIDE: Prediction Output --------
with col2:
    st.markdown("### Prediction Result")
    if st.button("Predict"):
        gender_encoded = 0 if gender.lower() == "female" else 1
        input_data = pd.DataFrame([[study_time, attendance, gender_encoded]],
                                  columns=["study_time", "attendance", "gender"])
        prediction = model.predict(input_data)

        # Display with reduced font sizes
        st.markdown(f"""
        <p style='font-size:16px;'>Study Time: <b>{study_time}</b></p>
        <p style='font-size:16px;'>Attendance: <b>{attendance}</b></p>
        <p style='font-size:16px;'>Gender: <b>{gender}</b></p>
        <p style='font-size:18px; color: green;'><b>Predicted score is: {int(prediction[0])}</b></p>
        """, unsafe_allow_html=True)

