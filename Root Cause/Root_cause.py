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
previous_timestamp = None
previous_all_spike = False
common_shift_active = False

temperature_frozen_count = 0
humidity_frozen_count = 0
pressure_frozen_count = 0
all_spike_count = 0

for i, row in df.iterrows():

    row = row.to_dict()

    row["timestamp"] = pd.to_datetime(row["timestamp"])

    result = detect_anomaly(row, model, roller)

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

    # Communication check

    if previous_timestamp is not None:
     timestamp_change = row["timestamp"] - previous_timestamp
    else:
     timestamp_change = None  

    communication_gap = (
        timestamp_change is not None
        and timestamp_change > pd.Timedelta(minutes=1)
    )

    communication_duplicate = (
        timestamp_change is not None
        and timestamp_change == pd.Timedelta(0)
    )

    communication_out_of_order = (
        timestamp_change is not None
        and timestamp_change < pd.Timedelta(0)
    )

    communication_error = (
        communication_gap
        or communication_duplicate
        or communication_out_of_order
    )  

    result["communication_gap"] = communication_gap
    result["communication_duplicate"] = communication_duplicate
    result["communication_out_of_order"] = communication_out_of_order
    result["communication_error"] = communication_error

    # Calculate changes

    if previous_temp is not None:
        temperature_change = row["temperature_c"] - previous_temp
        humidity_change = row["humidity_pct"] - previous_humidity
        pressure_change = row["pressure_hpa"] - previous_pressure
    else:
        temperature_change = 0
        humidity_change = 0
        pressure_change = 0  

    # Frozen sensor counters

    if abs(temperature_change) < 0.001:
        temperature_frozen_count += 1
    else:
        temperature_frozen_count = 0

    if abs(humidity_change) < 0.001:
        humidity_frozen_count += 1
    else:
        humidity_frozen_count = 0

    if abs(pressure_change) < 0.001:
        pressure_frozen_count += 1
    else:
        pressure_frozen_count = 0      

    # Spike detection

    temperature_spike = abs(temperature_change) > 1.1677
    humidity_spike = abs(humidity_change) > 3.4070
    pressure_spike = abs(pressure_change) > 1.5923

    # Count spikes
    spike_count = 0

    if temperature_spike:
        spike_count += 1

    if humidity_spike:
        spike_count += 1

    if pressure_spike:
        spike_count += 1    

    result["temperature_spike"] = temperature_spike
    result["humidity_spike"] = humidity_spike
    result["pressure_spike"] = pressure_spike
    result["spike_count"] = spike_count  

    simultaneous_shift = spike_count == 3

    simultaneous_reversal = (
    common_shift_active and
    simultaneous_shift
)

    result["simultaneous_reversal"] = simultaneous_reversal

    if simultaneous_reversal:
     common_shift_active = False

    elif simultaneous_shift:
     common_shift_active = True

    result["common_shift_active"] = common_shift_active
    result["simultaneous_shift"] = simultaneous_shift

     # Root cause: single-sensor spike
    
    if spike_count == 1:
     if temperature_spike:
        root_cause = "Temperature sensor anomaly"
     elif humidity_spike:
        root_cause = "Humidity sensor anomaly"
     elif pressure_spike:
        root_cause = "Pressure sensor anomaly"
    else:
     root_cause = "Undetermined"

    result["root_cause"] = root_cause 

    # Frozen sensor detection

    temperature_frozen = temperature_frozen_count >= 5
    humidity_frozen = humidity_frozen_count >= 5
    pressure_frozen = pressure_frozen_count >= 5

    result["temperature_frozen"] = temperature_frozen
    result["humidity_frozen"] = humidity_frozen
    result["pressure_frozen"] = pressure_frozen 

    current_all_spike = spike_count == 3

    if current_all_spike:
     all_spike_count += 1
    else:
     all_spike_count = 0

    if communication_error:
        root_cause = "Communication error"

    # Root cause: frozen sensor
    elif temperature_frozen:
        root_cause = "Temperature sensor frozen"
    elif humidity_frozen:
        root_cause = "Humidity sensor frozen"
    elif pressure_frozen:
        root_cause = "Pressure sensor frozen"
    elif simultaneous_reversal and not weather_consistent:
     root_cause = "Common/system-level data anomaly"    
    
    # Root cause: single-sensor spike
    elif spike_count == 1:
        if temperature_spike:
            root_cause = "Temperature sensor anomaly"
        elif humidity_spike:
            root_cause = "Humidity sensor anomaly"
        elif pressure_spike:
            root_cause = "Pressure sensor anomaly"
    else:
        root_cause = "Undetermined"
    
    result["root_cause"] = root_cause

    # Update previous value

    weather_consistent =temperature_humidity_consistent = (
    temperature_change < 0 and
    humidity_change > 0
    )

    # Root cause: potential genuine weather change
    if result['is_anomaly'] and weather_consistent and spike_count >= 2:
        root_cause = "Potential genuine weather change"

    result["root_cause"] = root_cause
    print(result)    

    # Dew point calculation

    a = 17.27
    b = 237.7
    gamma = (
        (a * row["temperature_c"]) / (b + row["temperature_c"])
        + np.log(row["humidity_pct"] / 100)
    )

    dew_point = (b * gamma) / (a - gamma)

    physically_impossible = dew_point > row["temperature_c"]

    # Check if weather is consistent

    weather_consistent = (
    temperature_humidity_consistent
    and pressure_change < 0
)

    previous_temp = row["temperature_c"]
    previous_humidity = row["humidity_pct"]
    previous_pressure = row["pressure_hpa"]
    previous_timestamp = row["timestamp"]
    previous_all_spike = current_all_spike



