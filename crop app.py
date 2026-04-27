import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Crop Prediction App",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌾 Crop Prediction Application")
st.markdown("---")
st.write("Welcome to the Crop Prediction App. Get personalized crop recommendations based on your farm conditions.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Make Prediction", "About"]
)

# Load Model and Preprocessors
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, label_encoder
    except:
        st.error("Model files not found. Please ensure model.pkl, scaler.pkl, and label_encoder.pkl are in the directory.")
        return None, None, None

# Home Page
if page == "Home":
    st.header("Welcome to Crop Prediction!")
    st.write("""
    This application helps you predict the best crop recommendations based on:
    - **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
    - **Weather Conditions**: Temperature, Humidity, Rainfall
    - **Soil pH**: Acidity/Alkalinity level
    
    Simply navigate to "Make Prediction" and input your farm conditions to get personalized recommendations.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Crops", "Multiple")
    with col2:
        st.metric("Model Accuracy", "High")
    with col3:
        st.metric("Status", "Ready")

# Prediction Page
elif page == "Make Prediction":
    st.header("Make a Crop Prediction")
    
    model, scaler, label_encoder = load_model()
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Nutrients (NPK)")
            nitrogen = st.slider("Nitrogen (N) (kg/ha)", 0, 140, 50, step=1)
            phosphorus = st.slider("Phosphorus (P) (kg/ha)", 0, 145, 50, step=1)
            potassium = st.slider("Potassium (K) (kg/ha)", 0, 205, 50, step=1)
        
        with col2:
            st.subheader("Climate & Soil Conditions")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, step=0.1)
            ph = st.slider("pH Level", 0.0, 14.0, 7.0, step=0.1)
        
        rainfall = st.slider("Rainfall (mm)", 0, 300, 100, step=1)
        
        if st.button("🔮 Predict Crop", use_container_width=True):
            try:
                # Prepare input data
                input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
                
                # Scale the input
                scaled_data = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(scaled_data)
                predicted_crop = label_encoder.inverse_transform(prediction)[0]
                
                st.success(f"✅ Recommended Crop: **{predicted_crop}**")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Unable to load prediction model. Please check model files.")

# About Page
elif page == "About":
    st.header("About This Application")
    st.write("""
    **Crop Prediction Application**
    
    This machine learning application provides intelligent crop recommendations 
    based on soil and weather conditions. The model is trained to predict 
    the most suitable crop for your specific farm conditions.
    
    **Key Features:**
    - Real-time crop predictions
    - Easy-to-use interface
    - Multiple input parameters
    - Accurate ML model
    
    **Version:** 1.0
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. Input your farm conditions (nutrients, weather, pH)
    2. The ML model analyzes your data
    3. Get an intelligent crop recommendation
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2024 Crop Prediction App | Built with Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
