import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
#import matplotlib.pyplot as plt

from ModelV2.AdaptiveRTModel import (
    AdaptiveModel,
    baseline_model,
    RollingZScore,
    detect_anomaly,
    z_threshold,
    features
)

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

window = 30

temp_history = []
humidity_history = []
pressure_history = []

model = baseline_model(df)
roller = RollingZScore()

previous_temp = None
previous_humidity = None
previous_pressure = None

for i, row in df.iterrows():

    row = row.to_dict()

    result = detect_anomaly(row, model, roller)

    print(result)


    # Temperature Z-score

    if len(temp_history) >= 5:
        temp_mean = np.mean(temp_history)
        temp_std = np.std(temp_history) or 1e-6

        temperature_z = (
            row["temperature_c"] - temp_mean
        ) / temp_std
    else:
        temperature_z = 0.0

    
    # Humidity Z-score

    if len(humidity_history) >= 5:
        humidity_mean = np.mean(humidity_history)
        humidity_std = np.std(humidity_history) or 1e-6

        humidity_z = (
            row["humidity_pct"] - humidity_mean
        ) / humidity_std
    else:
        humidity_z = 0.0

    
    # Pressure Z-score

    if len(pressure_history) >= 5:
        pressure_mean = np.mean(pressure_history)
        pressure_std = np.std(pressure_history) or 1e-6

        pressure_z = (
            row["pressure_hpa"] - pressure_mean
        ) / pressure_std
    else:
        pressure_z = 0.0

    
    # Update history
    

    temp_history.append(row["temperature_c"])
    humidity_history.append(row["humidity_pct"])
    pressure_history.append(row["pressure_hpa"])

    # Keep only latest 30 readings

    if len(temp_history) > window:
        temp_history.pop(0)

    if len(humidity_history) > window:
        humidity_history.pop(0)

    if len(pressure_history) > window:
        pressure_history.pop(0)

    
    # Count unusual sensors

    unusual_count = 0

    if abs(temperature_z) > z_threshold:
        unusual_count += 1

    if abs(humidity_z) > z_threshold:
        unusual_count += 1

    if abs(pressure_z) > z_threshold:
        unusual_count += 1

    
    # Calculate changes
    

    if previous_temp is not None:
        temperature_change = row["temperature_c"] - previous_temp
        humidity_change = row["humidity_pct"] - previous_humidity
        pressure_change = row["pressure_hpa"] - previous_pressure
    else:
        temperature_change = 0
        humidity_change = 0
        pressure_change = 0

    # Update previous values

    previous_temp = row["temperature_c"]
    previous_humidity = row["humidity_pct"]
    previous_pressure = row["pressure_hpa"]