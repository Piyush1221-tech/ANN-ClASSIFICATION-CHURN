import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np  
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load the scaler and encoder
with open('scaler.pkl', 'rb') as file:
    scaler =pickle.load(file)
    
with open('One_hot_encoder_geography.pkl','rb') as file:
    onehot=pickle.load(file)

with open('Label_encoder_gender.pkl','rb') as file:
    label_enc=pickle.load(file)
    
## Streamlit app
st.title("Customer Churn Prediction App")

geography = st.selectbox('Geography', onehot.categories_[0])
gender = st.selectbox('Gender', label_enc.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data =pd.DataFrame({ 
    'CreditScore': [credit_score],
    'Gender': label_enc.transform([gender])[0],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot.get_feature_names_out(['Geography']))

# combine one hot encoded geography with input data
input_data=pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f"Churn Probability: {prediction_prob:.2f}")

if prediction_prob >0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")