import streamlit as st
import joblib
import pandas as pd

st.title('Bank Customer Churn Prediction')
st.write('Enter the following information to predict customer churn:')
st.write('---')

# loading artefact
loaded_model = joblib.load('best_customer_churn_prediction.pkl')
encoded_loaded = joblib.load('customer_churn_encoder.pkl')
scaler_load = joblib.load('customer_churn_scaler.pkl')

# creating user inputs
credit = st.number_input('Credit Score')
age = st.number_input('Age')
tenure = st.number_input('Tenure')
balance = st.number_input('Balance')
products = st.selectbox('Products', [1, 3, 2, 4])
has_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_member = st.selectbox('Is Member', ['Yes', 'No'])
salary = st.number_input('Salary')
gender = st.selectbox('Gender', ['Female', 'Male'])
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])

data = [[age, balance, is_member, credit, tenure, salary, geography, gender]]
# x = pd.DataFrame(data, columns=['Age', 'Balance',  'IsActiveMember', 'CreditScore',  'Tenure', 'EstimatedSalary', 'Geography_encoded', 'Gender_encoded', 'NumOfProducts', 'HasCrCard'])

x = pd.DataFrame(data, columns=['Age', 'Balance',  'IsActiveMember', 'CreditScore',  'Tenure', 'EstimatedSalary', 'Geography_encoded', 'Gender_encoded'])
# x['HasCrCard'] = x['HasCrCard'].map({'Yes': 1, 'No': 0})
x['IsActiveMember'] = x['IsActiveMember'].map({'Yes': 1, 'No': 0})
x['Gender_encoded'] = encoded_loaded.fit_transform(x['Gender_encoded'])
x['Geography_encoded'] = encoded_loaded.fit_transform(x['Geography_encoded'])

x_scaled = scaler_load.transform(x)

generate = st.button("Generate")

if generate:
    predicted_output = loaded_model.predict(x_scaled)
    predicted_output = (predicted_output > 0.5).astype(int)
    st.write("Will Exit", "yes" if predicted_output == 1 else "no")
