import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# App title and description
st.title("Breast Cancer Prediction")
st.markdown("""
This application uses machine learning to predict breast cancer diagnosis based on patient data
including demographic information and clinical features.
""")

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Load the model from the file
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'model.pkl' is in the same directory as the app.")
        return None

# Load the model
model = load_model()

# Create a sidebar for inputs
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Please input the following patient information:")

# Feature descriptions for tooltips
feature_descriptions = {
    "Age": "Patient's age in years",
    "Menopause": "Menopausal status of the patient",
    "Tumor_Size": "Size of the tumor in mm",
    "Inv_Nodes": "Number of axillary lymph nodes that contain metastatic breast cancer",
    "Breast": "Breast where tumor was found",
    "Metastasis": "Whether distant metastases were detected",
    "Breast_Quadrant": "Location of the tumor within the breast",
    "History": "Family history of breast cancer"
}

# Create input fields with options and tooltips
def user_input_features():
    # Dictionary to store input values
    input_features = {}
    
    # Demographic Information
    st.sidebar.subheader("Demographic Information")
    
    # Age
    age_ranges = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
    input_features["Age"] = st.sidebar.selectbox(
        "Age Range", 
        options=age_ranges,
        index=2,  # Default to 40-49
        help=feature_descriptions["Age"]
    )
    
    # Menopausal Status
    menopause_options = ["premenopausal", "perimenopausal", "postmenopausal"]
    input_features["Menopause"] = st.sidebar.radio(
        "Menopausal Status",
        options=menopause_options,
        index=1,
        help=feature_descriptions["Menopause"]
    )
    
    # Family History
    history_options = ["yes", "no"]
    input_features["History"] = st.sidebar.radio(
        "Family History of Breast Cancer",
        options=history_options,
        index=1,
        help=feature_descriptions["History"]
    )
    
    # Clinical Features
    st.sidebar.subheader("Clinical Features")
    
    # Tumor Size
    tumor_size_ranges = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"]
    input_features["Tumor_Size"] = st.sidebar.selectbox(
        "Tumor Size (mm)",
        options=tumor_size_ranges,
        index=2,  # Default to 10-14mm
        help=feature_descriptions["Tumor_Size"]
    )
    
    # Invaded Nodes
    inv_nodes_ranges = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"]
    input_features["Inv_Nodes"] = st.sidebar.selectbox(
        "Invaded Lymph Nodes",
        options=inv_nodes_ranges,
        index=0,  # Default to 0-2
        help=feature_descriptions["Inv_Nodes"]
    )
    
    # Breast
    breast_options = ["left", "right"]
    input_features["Breast"] = st.sidebar.radio(
        "Breast",
        options=breast_options,
        index=0,
        help=feature_descriptions["Breast"]
    )
    
    # Breast Quadrant
    quadrant_options = ["left-up", "left-low", "right-up", "right-low", "central"]
    input_features["Breast_Quadrant"] = st.sidebar.selectbox(
        "Breast Quadrant",
        options=quadrant_options,
        index=0,
        help=feature_descriptions["Breast_Quadrant"]
    )
    
    # Metastasis
    metastasis_options = ["yes", "no"]
    input_features["Metastasis"] = st.sidebar.radio(
        "Distant Metastasis",
        options=metastasis_options,
        index=1,
        help=feature_descriptions["Metastasis"]
    )
    
    # Convert dictionary to DataFrame
    features_df = pd.DataFrame([input_features])
    return features_df

# Get user input
input_df = user_input_features()

# Function to encode categorical variables for the model
def preprocess_features(df):
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Encode categorical variables as needed
    # This should match the encoding used during model training
    
    # Example encoding (modify based on your model's requirements):
    # Age ranges - convert to midpoint of range
    age_map = {
        "20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55,
        "60-69": 65, "70-79": 75, "80-89": 85, "90-99": 95
    }
    df_encoded["Age"] = df_encoded["Age"].map(age_map)
    
    # Tumor size - convert to midpoint of range
    tumor_size_map = {
        "0-4": 2, "5-9": 7, "10-14": 12, "15-19": 17, "20-24": 22,
        "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42, 
        "45-49": 47, "50-54": 52, "55-59": 57
    }
    df_encoded["Tumor_Size"] = df_encoded["Tumor_Size"].map(tumor_size_map)
    
    # Invaded nodes - convert to midpoint of range
    inv_nodes_map = {
        "0-2": 1, "3-5": 4, "6-8": 7, "9-11": 10, "12-14": 13,
        "15-17": 16, "18-20": 19, "21-23": 22, "24-26": 25,
        "27-29": 28, "30-32": 31, "33-35": 34, "36-39": 37
    }
    df_encoded["Inv_Nodes"] = df_encoded["Inv_Nodes"].map(inv_nodes_map)
    
    # Binary encoding for yes/no features
    binary_map = {"yes": 1, "no": 0}
    df_encoded["Metastasis"] = df_encoded["Metastasis"].map(binary_map)
    df_encoded["History"] = df_encoded["History"].map(binary_map)
    
    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=["Menopause", "Breast", "Breast_Quadrant"],
        drop_first=False
    )
    
    return df_encoded

# Main panel
st.header("Patient Data Analysis")

# Display the input data in a more readable format
st.subheader("Patient Information")

# Create a more human-readable version of the input data
display_df = input_df.copy()
display_df.columns = [col.replace("_", " ") for col in display_df.columns]

# Creating a two-column layout for displaying patient info
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographic Information:**")
    st.write(f"• **Age Range:** {input_df['Age'].values[0]}")
    st.write(f"• **Menopausal Status:** {input_df['Menopause'].values[0]}")
    st.write(f"• **Family History:** {input_df['History'].values[0]}")

with col2:
    st.markdown("**Clinical Features:**")
    st.write(f"• **Tumor Size:** {input_df['Tumor_Size'].values[0]} mm")
    st.write(f"• **Invaded Lymph Nodes:** {input_df['Inv_Nodes'].values[0]}")
    st.write(f"• **Breast:** {input_df['Breast'].values[0]}")
    st.write(f"• **Breast Quadrant:** {input_df['Breast_Quadrant'].values[0]}")
    st.write(f"• **Distant Metastasis:** {input_df['Metastasis'].values[0]}")

# Make predictions when model is loaded and user clicks the button
if st.button("Predict Diagnosis"):
    if model:
        try:
            # Preprocess the input data
            processed_data = preprocess_features(input_df)
            
            # Ensure all expected columns are present for the model
            # This might need adjustment based on your specific model requirements
            
            # Make prediction
            prediction = model.predict(processed_data)
            
            try:
                # Try to get probability scores if the model supports it
                prediction_proba = model.predict_proba(processed_data)
                has_proba = True
            except:
                has_proba = False
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Create columns for the prediction display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Assuming binary classification where:
                # 0 = Benign, 1 = Malignant
                # Adjust based on your model's specific output
                if prediction[0] == 0:
                    st.success("🎉 Benign")
                    diagnosis = "Benign"
                else:
                    st.error("⚠️ Malignant")
                    diagnosis = "Malignant"
                
                # Show probability if available
                if has_proba:
                    st.metric(
                        label=f"Probability of {diagnosis}", 
                        value=f"{prediction_proba[0][prediction[0]]:.2%}"
                    )
            
            with col2:
                # Display probability bar chart if available
                if has_proba:
                    prob_df = pd.DataFrame({
                        'Category': ['Benign', 'Malignant'],
                        'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
                    })
                    
                    st.bar_chart(prob_df.set_index('Category'))
                else:
                    st.write("Probability scores not available for this model.")
            
            # Show interpretation
            st.subheader("Interpretation")
            if prediction[0] == 0:
                st.write("""
                The model predicts that the breast mass is likely **benign** (non-cancerous).
                
                A benign tumor is not an indication of cancer and is usually not harmful.
                However, please consult with a healthcare professional for proper medical advice.
                """)
            else:
                st.write("""
                The model predicts that the breast mass is potentially **malignant** (cancerous).
                
                This result suggests that further medical evaluation should be conducted.
                Please consult with a healthcare professional immediately for proper diagnosis and treatment options.
                """)
                
            st.info("⚠️ **Disclaimer**: This prediction is based on a machine learning model and should not replace professional medical diagnosis.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("This could be due to a mismatch between the input features and what the model expects. Please check your model requirements.")
    else:
        st.warning("Please ensure the model is properly loaded before making predictions.")

# Risk Factor Analysis
st.header("Risk Factor Analysis")

# Create a simple risk assessment based on known factors
risk_factors = []
risk_level = "Low"

if input_df["Age"].values[0] in ["60-69", "70-79", "80-89", "90-99"]:
    risk_factors.append("Age above 60")
    
if input_df["Menopause"].values[0] == "postmenopausal":
    risk_factors.append("Postmenopausal status")
    
if input_df["History"].values[0] == "yes":
    risk_factors.append("Family history of breast cancer")
    risk_level = "Moderate"
    
if input_df["Tumor_Size"].values[0] in ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"]:
    risk_factors.append("Large tumor size (>25mm)")
    risk_level = "High"
    
if input_df["Inv_Nodes"].values[0] not in ["0-2"]:
    risk_factors.append("Multiple invaded lymph nodes")
    risk_level = "High"

if input_df["Metastasis"].values[0] == "yes":
    risk_factors.append("Presence of distant metastasis")
    risk_level = "Very High"

# Display risk assessment
col1, col2 = st.columns([1, 2])

with col1:
    if risk_level == "Low":
        st.success(f"Risk Level: {risk_level}")
    elif risk_level == "Moderate":
        st.warning(f"Risk Level: {risk_level}")
    else:
        st.error(f"Risk Level: {risk_level}")

with col2:
    if risk_factors:
        st.write("**Identified risk factors:**")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.write("**No significant risk factors identified based on the input data.**")

# Prevention and Next Steps
st.header("Prevention and Next Steps")

st.write("""
### Recommendations:

1. **Regular Screening**: Continue with regular mammograms and clinical breast exams as recommended by your healthcare provider.

2. **Lifestyle Choices**: Maintain a healthy weight, stay physically active, and limit alcohol consumption.

3. **Follow-up**: Discuss these results with your healthcare provider to determine appropriate next steps.

4. **Additional Testing**: Your healthcare provider may recommend additional tests such as ultrasound, MRI, or biopsy for further evaluation.
""")

# About section
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This app demonstrates the use of machine learning for breast cancer prediction.
The model was trained on breast cancer patient data.

**Note**: This app is for educational purposes only and should not be used for actual medical diagnosis.
Always consult with healthcare professionals.
""")

# Instructions for deploying the app
if __name__ == "__main__":
    # This would be executed when running the script directly
    # streamlit run app.py
    pass