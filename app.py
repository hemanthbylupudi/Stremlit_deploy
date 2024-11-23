import streamlit as st
import pandas as pd
import pickle
from PIL import Image

import streamlit as st
import pandas as pd
import pickle

# Load models from the pickle file
with open('breast_cancer_models.pkl', 'rb') as file:
    models = pickle.load(file)

# Verify loaded models
if not isinstance(models, dict):
    st.error("Loaded object is not a dictionary. Please check the pickle file.")
else:
    model_no_smote = models.get('no_smote')
    model_with_smote = models.get('with_smote')

    if not model_no_smote or not model_with_smote:
        st.error("Models not found in the dictionary. Please check the keys in the pickle file.")


# Define feature names and descriptions
feature_descriptions = {
    'radius2': "Mean radius of the cell's perimeter.",
    'texture2': "Standard deviation of gray-scale values.",
    'smoothness2': "Local variation in radius lengths.",
    'concavity2': "Severity of concave portions of the cell contour.",
    'symmetry2': "Symmetry of the cell.",
    'fractal_dimension2': "Measure of boundary complexity.",
    'compactness3': "Measure of compactness of the cell."
}

# Load a header image (optional)
header_image = "header_image.jpg"
try:
    image = Image.open(header_image)
    st.image(image, use_column_width=True)
except FileNotFoundError:
    pass

# App title and intro
st.markdown(
    """
    <h1 style='
        text-align: center; 
        color: #4CAF50; 
        font-family: Arial, sans-serif; 
        font-size: 3rem; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);'>
        Breast Cancer Prediction App
    </h1>
    <h3 style='text-align: center;'>An interactive tool powered by machine learning to predict breast cancer outcomes.</h3>
    """, unsafe_allow_html=True)

# Sidebar with app info
st.sidebar.title("ℹ️ About the App")
st.sidebar.info(
    """
    This app uses two machine learning models to predict whether a tumor is **malignant** or **benign**.
    Compare predictions from models trained **with SMOTE** and **without SMOTE**, or choose to view one.
    """
)

# Feature input section
st.markdown("## 🔍 **Input Feature Values**:")
st.write("Use the input boxes below to provide the values for the tumor features.")

user_input = {}
for feature, description in feature_descriptions.items():
    with st.expander(f"📝 {feature.capitalize()}"):
        st.markdown(f"**Description:** {description}")
        user_input[feature] = st.number_input(
            f"Enter value for {feature.capitalize()}:",
            min_value=0.0,  # Adjust based on feature range
            value=10.0,
            step=0.1
        )

# Convert input to DataFrame
input_data = pd.DataFrame([user_input])

# Display raw input data (optional)
if st.checkbox("Show input data"):
    st.write("### Input Data Preview", input_data)

# Prediction section with checkboxes
st.markdown("## ⚙️ **Choose Model(s) to Predict**")
use_smote = st.checkbox("Use Model with SMOTE", value=True)
use_no_smote = st.checkbox("Use Model without SMOTE", value=True)

if st.button("💡 Predict"):
    try:
        if use_no_smote:
            # Results for model without SMOTE
            st.markdown("### Model Without SMOTE")
            prediction_no_smote = model_no_smote.predict(input_data)[0]
            probability_no_smote = model_no_smote.predict_proba(input_data)[0]

            if prediction_no_smote == 1:
                st.markdown(
                    f"<h2 style='text-align: center; color: red;'>🔴 Cancerous</h2>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h2 style='text-align: center; color: green;'>🟢 Non-Cancerous</h2>",
                    unsafe_allow_html=True
                )
            st.info(f"**Malignant Probability:** {probability_no_smote[1]:.2f}")
            st.info(f"**Benign Probability:** {probability_no_smote[0]:.2f}")

        if use_smote:
            # Results for model with SMOTE
            st.markdown("### Model With SMOTE")
            prediction_with_smote = model_with_smote.predict(input_data)[0]
            probability_with_smote = model_with_smote.predict_proba(input_data)[0]

            if prediction_with_smote == 1:
                st.markdown(
                    f"<h2 style='text-align: center; color: red;'>🔴 Cancerous</h2>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h2 style='text-align: center; color: green;'>🟢 Non-Cancerous</h2>",
                    unsafe_allow_html=True
                )
            st.info(f"**Malignant Probability:** {probability_with_smote[1]:.2f}")
            st.info(f"**Benign Probability:** {probability_with_smote[0]:.2f}")

        if not (use_smote or use_no_smote):
            st.warning("Please select at least one model to predict.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <small>© 2024 Breast Cancer Prediction App | Powered by Streamlit</small>
    </div>
    """, unsafe_allow_html=True)
