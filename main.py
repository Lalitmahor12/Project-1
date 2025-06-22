# Climate-Adaptive Crop Yield Optimization

"""
Project Overview:
Forecast regional crop yields under different climate scenarios and recommend adaptive planting strategies using Python.
Modules:
1. Random Data Generation
2. Climate Scenario Simulation
3. Crop Yield Prediction (ML Model)
4. Strategy Recommendation Engine
"""

# === 1. IMPORT REQUIRED LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === 2. GENERATE RANDOMIZED DATA ===
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'region': np.random.choice(['RegionA', 'RegionB'], n_samples),
    'year': np.random.randint(2000, 2024, size=n_samples),
    'crop': np.random.choice(['Wheat', 'Rice'], n_samples),
    'temperature': np.random.normal(25, 3, n_samples),
    'rainfall': np.random.normal(1000, 150, n_samples),
    'soil_moisture': np.random.uniform(10, 30, n_samples),
    'yield': np.random.normal(3, 0.6, n_samples)
})

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['region', 'crop'], drop_first=True)

# Prepare features and target
X = data.drop(columns=['yield'])
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. TRAIN ML MODEL TO PREDICT YIELD ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 4. EVALUATE MODEL ===
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === 5. SIMULATE CLIMATE SCENARIOS ===
# Example: temperature +2°C and rainfall -10%
future_data = X_test.copy()
future_data['temperature'] += 2
future_data['rainfall'] *= 0.9
future_pred = model.predict(future_data)

# === 6. STRATEGY RECOMMENDATION ENGINE ===
def recommend_strategy(X, predictions):
    result = X.copy()
    result['predicted_yield'] = predictions
    if 'region_RegionB' in result.columns:
        grouped = result.groupby('region_RegionB').mean(numeric_only=True).reset_index()
        print("\nRecommended Strategies:")
        for idx, row in grouped.iterrows():
            region_name = 'RegionB' if row['region_RegionB'] == 1 else 'RegionA'
            if row['predicted_yield'] < 2.5:
                print(f"{region_name}: Consider switching crop or improving irrigation.")
            else:
                print(f"{region_name}: Continue current practices with slight modifications.")

recommend_strategy(future_data, future_pred)

# === 7. VISUALIZATION ===
plt.figure(figsize=(10, 5))
sns.kdeplot(y_test, label="Actual", shade=True)
sns.kdeplot(y_pred, label="Predicted", shade=True)
plt.title("Actual vs Predicted Yield Distribution")
plt.legend()
plt.show()
