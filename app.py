import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the trained model
with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
    This app uses a machine learning model trained on breast cancer data to predict whether a tumor is **malignant** or **benign**.
    Simply provide the feature values below and click **Predict**.
    """
)

# Sidebar placeholder for future settings
st.sidebar.header("⚙️ Settings")
st.sidebar.write("More settings coming soon!")

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

# Prediction section
if st.button("💡 Predict"):
    try:
        prediction = model.predict(input_data)[0]  # Predict outcome
        probability = model.predict_proba(input_data)[0]  # Get probabilities

        # Styling for prediction result
        if prediction == 1:
            st.markdown(
                f"<h2 style='text-align: center; color: red;'>🔴 Cancerous</h2>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='text-align: center; color: green;'>🟢 Non-Cancerous</h2>", 
                unsafe_allow_html=True
            )
        
        # Display prediction probabilities
        st.write("### Prediction Confidence")
        malignant_confidence = probability[1]
        benign_confidence = probability[0]
        st.progress(malignant_confidence)
        st.info(f"**Malignant Probability:** {malignant_confidence:.2f}")
        st.info(f"**Benign Probability:** {benign_confidence:.2f}")

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
