# app.py
# Step 4: Streamlit Web App for California Housing Price Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration (Professional look)
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('housing_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("🏠 California House Price Predictor")
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    ### Predict median house prices based on location and features
    *This model uses Random Forest with 80% accuracy (R² = 0.81)*
""")

# Create two columns for layout
col1, col2 = st.columns(2)

# Input features
with col1:
    st.subheader("📍 Location Features")
    latitude = st.slider("Latitude", 32.0, 42.0, 37.88, 0.01)
    longitude = st.slider("Longitude", -124.0, -114.0, -122.23, 0.01)
    
    st.subheader("🏘️ Neighborhood Features")
    med_inc = st.number_input("Median Income ($100,000s)", 0.0, 15.0, 3.0, 0.01)
    house_age = st.slider("House Age (years)", 1, 100, 30)
    
with col2:
    st.subheader("🏠 Property Features")
    ave_rooms = st.number_input("Average Rooms per Household", 1.0, 10.0, 5.0, 0.1)
    ave_bedrms = st.number_input("Average Bedrooms per Household", 0.5, 5.0, 1.0, 0.1)
    
    st.subheader("👨‍👩‍👧‍👦 Population Features")
    population = st.number_input("Population", 100, 50000, 1000, 100)
    ave_occup = st.number_input("Average Occupancy per Household", 0.5, 10.0, 2.5, 0.1)

# Predict button
if st.button("💰 Predict House Price", type="primary"):
    # Prepare input data
    input_data = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms, 
        population, ave_occup, latitude, longitude
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    price = prediction * 100000  # Convert from $100,000s to actual dollars
    
    # Display results
    st.divider()
    st.success(f"### 🎯 Predicted Median House Price: **${price:,.0f}**")
    
    # Show confidence range
    st.info(f"**Confidence:** ±$33,000 (Based on model's MAE)")
    
    # Add a gauge-like visual
    st.progress(min(prediction / 5.0, 1.0))  # max price is ~$5,000,000
    
    # Show input summary
    with st.expander("🔍 View Input Summary"):
        st.write("### Features Used:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"- **Median Income:** ${med_inc*100000:,.0f}")
            st.write(f"- **House Age:** {house_age} years")
            st.write(f"- **Avg Rooms:** {ave_rooms}")
            st.write(f"- **Avg Bedrooms:** {ave_bedrms}")
        with col_b:
            st.write(f"- **Population:** {population:,}")
            st.write(f"- **Avg Occupancy:** {ave_occup}")
            st.write(f"- **Location:** ({latitude:.2f}, {longitude:.2f})")

# Sidebar with model info
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("R² Score", "0.81", "Very Good")
    st.metric("Mean Absolute Error", "$33,000", "Low")
    st.metric("Training Data", "16,512 rows", "")
    
    st.divider()
    st.header("📁 Dataset Features")
    st.write("""
    - **MedInc**: Median income in block group
    - **HouseAge**: Median house age
    - **AveRooms**: Average rooms per household
    - **AveBedrms**: Average bedrooms per household  
    - **Population**: Block group population
    - **AveOccup**: Average household size
    - **Latitude/Longitude**: Location
    """)
    
    st.divider()
    st.caption("Built with Streamlit • Random Forest Regressor")

# Footer
st.divider()
st.caption("🏗️ Deployed with Streamlit • Predictions are estimates based on historical data")