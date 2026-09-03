import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

sensors = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

for sensor in sensors:

    print(f"\n--- {sensor} ---")

    unchanged = df[sensor].diff().eq(0)

    run_length = 0
    runs = []

    for value in unchanged:

        if value:
            run_length += 1
        else:
            if run_length > 0:
                runs.append(run_length + 1)
            run_length = 0

    if run_length > 0:
        runs.append(run_length + 1)

    print("Repeated-value run lengths:", runs)

    temperature = df["temperature_c"]

unchanged = temperature.diff().eq(0)

start = None

for i, value in enumerate(unchanged):

    if value and start is None:
        start = i - 1

    elif not value and start is not None:
        end = i - 1

        if end - start + 1 >= 3:
            print("\nTemperature repeated from:")
            print(df.loc[start, "timestamp"])
            print("to:")
            print(df.loc[end, "timestamp"])

            print("\nValues during this period:")
            print(df.loc[start:end,
                        ["timestamp", "temperature_c",
                         "humidity_pct", "pressure_hpa"]])

        start = None