import streamlit as st
import pandas as pd
import joblib

# Load the model and important features
model, important_features = joblib.load("model_with_features.pkl")

# Original feature order used during training
original_order = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1',
    'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2',
    'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
    'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
    'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
    'symmetry3', 'fractal_dimension3'
]

# Align important features with the original order
important_features_in_order = [feature for feature in original_order if feature in important_features]

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write("Enter the values for the important features to predict if the tumor is benign or malignant.")

# Collect user inputs for all features
input_data = {}
for feature in original_order:
    if feature in important_features_in_order:
        value = st.number_input(f"Enter value for {feature}:", step=0.01)
    else:
        value = 0.0  # Default value for features not in the important list
    input_data[feature] = value

# When the "Predict" button is clicked
if st.button("Predict"):
    # Convert input data to a DataFrame, ensuring the correct order
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        prediction_label = "Malignant" if prediction == 1 else "Benign"
        st.success(f"The predicted result is: {prediction_label}")
    except ValueError as e:
        st.error(f"Error during prediction: {str(e)}")

st.write("Model Loaded Successfully!")
