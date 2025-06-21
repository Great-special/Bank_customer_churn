from tensorflow.keras.models import load_model
import streamlit as st
import joblib
import pandas as pd

st.title('Bank Customer Churn Prediction')
st.write('Enter the following information to predict customer churn:')
st.write('---')

# loading articft
loaded_model = load_model('bank_customer_churn.h5')
scaler_load = joblib.load('scaler_bank_customer.h5')
encoded_loaded = joblib.load('/content/drive/MyDrive/Colab Notebooks/saved_models/encoder_bank_customer.h5')

# creating user inputs
credit = st.number_input('Credit Score')
age = st.number_input('Age')
tenure = st.number_input('Tenure')
balance = st.number_input('Balance')
products = st.number_input('Products')
has_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_member = st.selectbox('Is Member', ['Yes', 'No'])
salary = st.number_input('Salary')
gender = st.selectbox('Gender', ['Female', 'Male'])
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])

data = [[credit, age, tenure, balance, products, has_card, is_member, salary, gender, geography]]
x = pd.DataFrame(data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_encoded', 'Geography_encoded'])
x['HasCrCard'] = x['HasCrCard'].map({'Yes': 1, 'No': 0})
x['IsActiveMember'] = x['IsActiveMember'].map({'Yes': 1, 'No': 0})
x['Gender_encoded'] = encoded_loaded.fit_transform(x['Gender_encoded'])
x['Geography_encoded'] = encoded_loaded.fit_transform(x['Geography_encoded'])

x_scaled = scaler_load.transform(x)

predicted_output = loaded_model.predict(x_scaled)
predicted_output = (predicted_output > 0.5).astype(int)
st.write("Will Exit", "yes" if predicted_output == 1 else "no")
