import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings

# Suppress the FutureWarning from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

# Set the title and a brief description
st.title('Credit Card Fraud Detection')
st.write('Enter the transaction details to check if it is fraudulent.')

# Load the trained model and scaler
try:
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'trained_model.sav' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Create input fields for user to enter data
st.header('Transaction Details')

# Using a sidebar for a cleaner layout
with st.sidebar:
    st.header('Input Features')

    # Add an explanation for the anonymized features
    st.markdown("""
    **Understanding the Features:**

    The `V1` to `V28` features are anonymized. They are the result of a **Principal Component Analysis (PCA)** transformation of the original data. This was done to protect the privacy of the credit card holders and merchants.
    
    The only non-anonymized features are `Time` and `Amount`.
    """)
    
    # Input fields remain the same, requiring user input
    time = st.number_input('Time (seconds since first transaction)')
    v1 = st.number_input('V1')
    v2 = st.number_input('V2')
    v3 = st.number_input('V3')
    v4 = st.number_input('V4')
    v5 = st.number_input('V5')
    v6 = st.number_input('V6')
    v7 = st.number_input('V7')
    v8 = st.number_input('V8')
    v9 = st.number_input('V9')
    v10 = st.number_input('V10')
    v11 = st.number_input('V11')
    v12 = st.number_input('V12')
    v13 = st.number_input('V13')
    v14 = st.number_input('V14')
    v15 = st.number_input('V15')
    v16 = st.number_input('V16')
    v17 = st.number_input('V17')
    v18 = st.number_input('V18')
    v19 = st.number_input('V19')
    v20 = st.number_input('V20')
    v21 = st.number_input('V21')
    v22 = st.number_input('V22')
    v23 = st.number_input('V23')
    v24 = st.number_input('V24')
    v25 = st.number_input('V25')
    v26 = st.number_input('V26')
    v27 = st.number_input('V27')
    v28 = st.number_input('V28')
    amount = st.number_input('Amount')

# Create a button to trigger the prediction
if st.button('Predict Transaction'):
    # Prepare the input data
    input_data = [
        time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
        v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26,
        v27, v28, amount
    ]

    # Convert input to a numpy array and reshape for a single prediction
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Create a DataFrame to apply the scaler with correct column names
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame(input_data_reshaped, columns=feature_names)
    
    # Scale the 'Time' and 'Amount' features using the loaded scaler
    features_to_scale = ['Time', 'Amount']
    input_df[features_to_scale] = loaded_scaler.transform(input_df[features_to_scale])
    
    # Make the prediction
    try:
        prediction = loaded_model.predict(input_df)
        
        # Display the result
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('✅ **The transaction is legitimate.**')
        else:
            st.warning('🚨 **The transaction is fraudulent!**')
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("This app uses a Logistic Regression model trained on a balanced subset of the credit card transaction dataset.")