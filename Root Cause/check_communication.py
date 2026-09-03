import pandas as pd

df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

expected_interval = pd.Timedelta(minutes=1)

time_difference = df["timestamp"].diff()

communication_gap = time_difference > expected_interval

print("\n--- Communication gaps ---")

if communication_gap.any():
    print(df.loc[communication_gap, ["timestamp"]])
else:
    print("No communication gaps found.")

# Sort by timestamp so we can inspect the time sequence
df = df.sort_values("timestamp").reset_index(drop=True)

time_difference = df["timestamp"].diff()

print("Expected interval: 1 minute")

# 1. Missing / gapped readings
gaps = df[time_difference > pd.Timedelta(minutes=1)]

print("\n--- Time gaps ---")
if len(gaps) == 0:
    print("No gaps found.")
else:
    print(gaps[["timestamp"]])
    print("Number of gaps:", len(gaps))

# 2. Duplicate timestamps
duplicates = df[df["timestamp"].duplicated(keep=False)]

print("\n--- Duplicate timestamps ---")
if len(duplicates) == 0:
    print("No duplicate timestamps found.")
else:
    print(duplicates[["timestamp"]])

# 3. Out-of-order timestamps
# Check the original sequence before sorting
original_df = pd.read_csv("ModelV2/TestAnomalyUnlabled.csv")
original_df["timestamp"] = pd.to_datetime(original_df["timestamp"])

original_difference = original_df["timestamp"].diff()

out_of_order = original_df[original_difference < pd.Timedelta(0)]

print("\n--- Out-of-order timestamps ---")
if len(out_of_order) == 0:
    print("No out-of-order timestamps found.")
else:
    print(out_of_order[["timestamp"]])



print("\n--- Test: Artificial communication gap ---")

test_times = pd.to_datetime([
    "2026-08-01 10:00:00",
    "2026-08-01 10:01:00",
    "2026-08-01 10:04:00"
])

test_difference = test_times.to_series().diff()

for difference in test_difference.dropna():

    if difference > expected_interval:
        print("Communication gap detected:", difference)
    else:
        print("Normal interval:", difference)    


print("\n--- Test: Artificial duplicate timestamp ---")

test_duplicate = pd.to_datetime([
    "2026-08-01 10:00:00",
    "2026-08-01 10:01:00",
    "2026-08-01 10:01:00"
])

duplicate_found = test_duplicate.duplicated()

if duplicate_found.any():
    print("Duplicate timestamp detected.")
else:
    print("No duplicate timestamp detected.")


print("\n--- Test: Artificial out-of-order timestamp ---")

test_order = pd.to_datetime([
    "2026-08-01 10:00:00",
    "2026-08-01 10:02:00",
    "2026-08-01 10:01:00"
])

order_difference = test_order.to_series().diff()

if (order_difference < pd.Timedelta(0)).any():
    print("Out-of-order timestamp detected.")
else:
    print("No out-of-order timestamp detected.")        