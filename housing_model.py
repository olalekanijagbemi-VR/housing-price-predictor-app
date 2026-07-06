# housing_model.py
# Step 3: Train the Machine Learning Model

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ssl
import joblib

# Fix SSL issue
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Load & prepare data
print("📊 Loading data...")
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN THE MODEL (Random Forest - Industry Standard)
print("🤖 Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. MAKE PREDICTIONS
y_pred = model.predict(X_test_scaled)

# 6. EVALUATE PERFORMANCE (Employers love these numbers)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"   Mean Absolute Error: ${mae:.2f} (in $100,000s)")
print(f"   Root Mean Squared Error: ${rmse:.2f} (in $100,000s)")
print(f"   R² Score: {r2:.4f} (Higher is better, max=1.0)")

# 7. SAVE THE MODEL & SCALER (For deployment)
joblib.dump(model, 'housing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model saved as 'housing_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("\n💡 The model is ready for deployment!")