import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
weather_df = pd.read_csv(r'D:\STET-details\msc-\Thillainayaki\Thillai\Machine-Learning-Model-for-Weather-Forecasting-main\kanpur.csv', 
                         parse_dates=['date_time'], index_col='date_time')

# Print all available columns
print("All columns in dataset:")
print(weather_df.columns)

# Select numerical features including the target variable
weather_df_num = weather_df.loc[:, ['maxtempC', 'mintempC', 'cloudcover', 'humidity', 'tempC', 
                                     'sunHour', 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph']]

print("Selected features:")
print(weather_df_num.columns)

# Separate features and label
weather_y = weather_df_num.pop("tempC")  # Target
weather_x = weather_df_num               # Features

# 🔧 Split into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=42)

# 🌲 Random Forest Model
regr = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)
regr.fit(train_X, train_y)

# ✅ Predict
prediction3 = regr.predict(test_X)

# 📊 Evaluation Metrics
print("Mean Absolute Error: %.2f" % mean_absolute_error(test_y, prediction3))
print("Mean Squared Error: %.2f" % np.mean((prediction3 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, prediction3))
print("Model Accuracy (Variance Score): %.2f" % regr.score(test_X, test_y))
