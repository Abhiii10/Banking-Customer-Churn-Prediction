import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and preprocessing objects
model = joblib.load('churn_model.pkl')
label_encoder_geography = joblib.load('label_encoder_geography.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
scaler = joblib.load('scaler.pkl')
columns_order = joblib.load('columns_order.pkl')

# Streamlit UI components for user input
st.title("Customer Churn Prediction")

# Collect user input
credit_score = st.number_input('Credit Score', min_value=0, max_value=850, value=650)
age = st.number_input('Age', min_value=18, max_value=100, value=40)
balance = st.number_input('Balance', min_value=0, value=50000)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=3)
tenure = st.number_input('Tenure (Years)', min_value=0, max_value=10, value=3)
has_cr_card = st.selectbox('Has Credit Card', [1, 0])
is_active_member = st.selectbox('Is Active Member', [1, 0])
estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])

# Button for triggering the prediction
if st.button("Predict Churn"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'Tenure': [tenure],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
        'Gender': [gender]
    })

    # Ensure the input data has the same columns and order as the model's training data
    input_data = input_data[columns_order]

    # Encode categorical variables
    input_data['Geography'] = label_encoder_geography.transform(input_data['Geography'])
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

    # Standardize numerical features
    columns_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

    # Make prediction and get prediction probabilities
    prediction_prob = model.predict_proba(input_data)[0]

    # Determine the predicted class (1 for churn, 0 for no churn)
    predicted_class = model.predict(input_data)[0]

    # Display prediction result and confidence score
    if predicted_class == 1:
        st.write("Prediction: **Customer will churn**")
        st.write(f"Confidence: **{prediction_prob[1] * 100:.2f}%**")
    else:
        st.write("Prediction: **Customer will not churn**")
        st.write(f"Confidence: **{prediction_prob[0] * 100:.2f}%**")
