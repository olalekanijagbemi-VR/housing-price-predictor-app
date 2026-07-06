# app_advanced.py
# Advanced Housing Price Predictor with Multiple Models & Visual Metrics

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ssl

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Page config
st.set_page_config(
    page_title="Advanced Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for glassmorphism effect
st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
    }
    .metric-glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("🏠 Housing Predictor")
    st.sidebar.markdown("### Advanced ML Models")
    
    # Load data
    @st.cache_data
    def load_data():
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target
        return df, data.feature_names
    
    df, feature_names = load_data()
    
    # Split data
    @st.cache_data
    def split_data(df):
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    X_train, X_test, y_train, y_test, scaler = split_data(df)
    
    # Model selection
    st.sidebar.subheader("🤖 Choose Model Classifier")
    classifier = st.sidebar.selectbox(
        "Select Model Classifiers",
        [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ]
    )
    
    # Hyperparameters
    st.sidebar.subheader("⚙️ Tune Hyperparameters")
    
    if classifier == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model_name = "Random Forest"
        
    elif classifier == "Gradient Boosting":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model_name = "Gradient Boosting"
        
    elif classifier == "Linear Regression":
        st.sidebar.info("No hyperparameters to tune for Linear Regression")
        model = LinearRegression()
        model_name = "Linear Regression"
        
    elif classifier == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, 0.01)
        model = Ridge(alpha=alpha, random_state=42)
        model_name = "Ridge Regression"
        
    else:  # Lasso
        alpha = st.sidebar.slider("Alpha (Regularization)", 0.001, 1.0, 0.01, 0.001)
        model = Lasso(alpha=alpha, random_state=42)
        model_name = "Lasso Regression"
    
    st.sidebar.divider()
    
    # 🚀 TRAIN BUTTON - MOVED ABOVE METRICS (always visible)
    train_clicked = st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True)
    
    st.sidebar.divider()
    
    # Metrics to plot
    st.sidebar.subheader("📊 Metrics to Display")
    show_metrics = st.sidebar.multiselect(
        "Select metrics:",
        ["MAE", "RMSE", "R² Score", "Feature Importance", "Actual vs Predicted", "Residual Plot"],
        default=["MAE", "R² Score"]
    )
    
    # Show raw data
    show_raw = st.sidebar.checkbox("Show Raw Data", False)
    
    # Main area
    st.title("🏠 Advanced California Housing Predictor")
    st.markdown("""
    *Compare multiple machine learning models with interactive visualizations*
    """)
    
    # Input features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("📍 **Location**")
        latitude = st.slider("Latitude", 32.0, 42.0, 37.88, 0.01)
        longitude = st.slider("Longitude", -124.0, -114.0, -122.23, 0.01)
    
    with col2:
        st.markdown("🏘️ **Demographics**")
        med_inc = st.number_input("Median Income ($100k)", 0.0, 15.0, 3.0, 0.01)
        house_age = st.slider("House Age (years)", 1, 100, 30)
    
    with col3:
        st.markdown("🏠 **Property**")
        ave_rooms = st.number_input("Avg Rooms", 1.0, 10.0, 5.0, 0.1)
        ave_bedrms = st.number_input("Avg Bedrooms", 0.5, 5.0, 1.0, 0.1)
        population = st.number_input("Population", 100, 50000, 1000, 100)
        ave_occup = st.number_input("Avg Occupancy", 0.5, 10.0, 2.5, 0.1)
    
    st.divider()
    
    # Train and predict
    if train_clicked:
        with st.spinner(f"Training {model_name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics in glass cards
        st.subheader(f"📊 {model_name} Performance")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        if "MAE" in show_metrics:
            col_m1.metric("MAE", f"${mae*100000:,.0f}", "Lower is better")
        if "RMSE" in show_metrics:
            col_m2.metric("RMSE", f"${rmse*100000:,.0f}", "Lower is better")
        if "R² Score" in show_metrics:
            col_m3.metric("R² Score", f"{r2:.4f}", "Higher is better")
        col_m4.metric("Model", model_name, "Selected")
        
        # Individual prediction
        st.subheader("🎯 Predict Your House")
        input_data = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        price = prediction * 100000
        
        st.success(f"### Predicted Median House Price: **${price:,.0f}**")
        st.progress(min(prediction / 5.0, 1.0))
        
        # Visualizations
        if "Feature Importance" in show_metrics and hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 4))
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            ax.barh(range(len(indices)), importance[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
            plt.close()
        
        if "Actual vs Predicted" in show_metrics:
            st.subheader("📈 Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Price ($100k)")
            ax.set_ylabel("Predicted Price ($100k)")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            plt.close()
        
        if "Residual Plot" in show_metrics:
            st.subheader("📉 Residual Plot")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel("Predicted Price ($100k)")
            ax.set_ylabel("Residuals ($100k)")
            ax.set_title("Residual Plot")
            st.pyplot(fig)
            plt.close()
    
    # Show raw data
    if show_raw:
        st.subheader("📋 Raw Data")
        st.write(df)
        st.markdown("""
        **California Housing Dataset (1990 Census)**
        
        | Feature | Description |
        |---------|-------------|
        | MedInc | Median income in block group |
        | HouseAge | Median house age |
        | AveRooms | Average rooms per household |
        | AveBedrms | Average bedrooms per household |
        | Population | Block group population |
        | AveOccup | Average household size |
        | Latitude | Latitude coordinate |
        | Longitude | Longitude coordinate |
        | MedHouseVal | Median house value (target) |
        """)

if __name__ == "__main__":
    main()