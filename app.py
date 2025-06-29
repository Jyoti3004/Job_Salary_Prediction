# ...................RUN APP USING STREAMLIT RUN APP.PY.....................

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoders (you can replace this with re-training logic if needed)
# For this case, we simulate with re-declared model and encoders inline as no joblib is used

# Dummy training setup to simulate already-trained model from notebook
# In a real case, you should save and load model using joblib or pickle

data = pd.read_csv("Dataset09-Employee-salary-prediction.csv")
data = data.dropna().copy()
if 'Gender' in data.columns:
    data = data.drop(columns=['Gender'])

# Encode categorical data
label_encoders = {}
for column in ['Education Level', 'Job Title']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Ensure numerical type
data['Years of Experience'] = pd.to_numeric(data['Years of Experience'], errors='coerce')

X = data.drop(columns=['Salary'])
y = data['Salary']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Employee Salary Prediction")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
education = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
job_title = st.selectbox("Job Title", label_encoders['Job Title'].classes_)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

if st.button("Predict Salary"):
    input_df = pd.DataFrame([[age, education, job_title, experience]],
                            columns=["Age", "Education Level", "Job Title", "Years of Experience"])

    for col in ['Education Level', 'Job Title']:
        input_df[col] = label_encoders[col].transform(input_df[col])

    salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${salary:.2f}")
