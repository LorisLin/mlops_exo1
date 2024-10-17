import streamlit as st
import joblib

model = joblib.load("regression.joblib")

st.title("House Price Prediction App")

with st.form(key='house_form'):
    size = st.number_input("Enter the size of the house (in square meters):", min_value=1, max_value=10000, value=50)
    bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, max_value=20, value=3)
    garden = st.number_input("Does the house have a garden? (1 for Yes, 0 for No):", min_value=0, max_value=1, value=0)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_features = [[size, bedrooms, garden]]
    prediction = model.predict(input_features)
    st.write(f"The predicted house price is: ${prediction[0]:,.2f}")
