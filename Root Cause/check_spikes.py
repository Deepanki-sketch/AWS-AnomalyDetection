import pandas as pd

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

# Calculate absolute changes
df["temperature_change"] = df["temperature_c"].diff().abs()
df["humidity_change"] = df["humidity_pct"].diff().abs()
df["pressure_change"] = df["pressure_hpa"].diff().abs()

# Candidate thresholds = 99.5th percentile
temp_threshold = df["temperature_change"].quantile(0.995)
humidity_threshold = df["humidity_change"].quantile(0.995)
pressure_threshold = df["pressure_change"].quantile(0.995)

print("Temperature threshold:", temp_threshold)
print("Humidity threshold:", humidity_threshold)
print("Pressure threshold:", pressure_threshold)

print("\n--- Temperature spikes ---")
print(
    df[df["temperature_change"] > temp_threshold][
        ["timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
         "temperature_change"]
    ]
)

print("\n--- Humidity spikes ---")
print(
    df[df["humidity_change"] > humidity_threshold][
        ["timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
         "humidity_change"]
    ]
)

print("\n--- Pressure spikes ---")
print(
    df[df["pressure_change"] > pressure_threshold][
        ["timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
         "pressure_change"]
    ]
)