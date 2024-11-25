import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Define the exact feature names in the correct order used during model training
feature_names = [
    'radius2', 'texture2', 'smoothness2', 'compactness3', 'concavity2'
    # Add all feature names here, in the same order as the training data
]

# Streamlit app
st.title("Disease Prediction Model")
st.write("Enter the feature values in the fields below to predict the diagnosis.")

# Create input fields for each feature
inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    # Convert inputs into a DataFrame and ensure the column names match
    input_data = pd.DataFrame([inputs], columns=feature_names)

    try:
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
        st.write(f"Probability: {prediction_proba[0][1]*100:.2f}% Positive, {prediction_proba[0][0]*100:.2f}% Negative")
    except Exception as e:
        st.error(f"Error: {e}")
