import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
weather_df = pd.read_csv(r'D:\STET-details\msc-\Thillainayaki\Thillai\Machine-Learning-Model-for-Weather-Forecasting-main\kanpur.csv', parse_dates=['date_time'], index_col='date_time')

# Feature selection
features = ['maxtempC', 'mintempC', 'cloudcover', 'humidity', 'tempC', 
            'sunHour', 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph']
weather_df = weather_df[features]

# Split features & label
X = weather_df.drop("tempC", axis=1)
y = weather_df["tempC"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, max_depth=90, random_state=0)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'weather_model.pkl')

# Save accuracy info
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

with open("model_metrics.txt", "w") as f:
    f.write(f"R2 Score: {r2:.2f}\nMAE: {mae:.2f}")
