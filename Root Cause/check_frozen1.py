import pandas as pd

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

TOLERANCE = 0.001
MIN_CONSECUTIVE = 5

sensors = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

for sensor in sensors:

    changes = df[sensor].diff().abs()

    frozen_count = 0
    frozen_runs = []

    for change in changes:

        if pd.notna(change) and change < TOLERANCE:
            frozen_count += 1
        else:
            if frozen_count >= MIN_CONSECUTIVE:
                frozen_runs.append(frozen_count + 1)
            frozen_count = 0

    if frozen_count >= MIN_CONSECUTIVE:
        frozen_runs.append(frozen_count + 1)

    print(f"\n--- {sensor} ---")
    print("Frozen runs:", frozen_runs)