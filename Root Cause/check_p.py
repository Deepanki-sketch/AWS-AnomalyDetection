import pandas as pd

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

pressure_change = df["pressure_hpa"].diff()

print(pressure_change.describe())

print("\nAbsolute pressure-change percentiles:")

print(
    pressure_change.abs().quantile(
        [0.50, 0.90, 0.95, 0.99, 0.995, 1.00]
    )
)