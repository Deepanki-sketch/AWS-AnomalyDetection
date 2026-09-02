from ARTModel2 import AdaptiveModel,RollingZScore,detect_anomaly,TrackChanges
import pandas as pd

df = pd.read_csv("./ModelV2/TestAnomalyUnlabled.csv")
df = df.sort_values("timestamp")
df = df.reset_index(drop=True)

baseline_df = df.iloc[:200]
baseline_df["temperature_c_change"] = baseline_df["temperature_c"].diff().fillna(0)
baseline_df["humidity_pct_change"] = baseline_df["humidity_pct"].diff().fillna(0)
baseline_df["pressure_hpa_change"] = baseline_df["pressure_hpa"].diff().fillna(0)

stream_df = df.iloc[200:]

adaptive = AdaptiveModel(baseline_df)
roller = RollingZScore()

results = []
trackchange = TrackChanges()
for _, row in stream_df.iterrows():
    row_dict = row.to_dict()
    changes = trackchange.compute_changes(row_dict)
    row_dict.update(changes)
    result = detect_anomaly(row_dict, adaptive.get_model(), roller)
    results.append(result)
    adaptive.addnrefit(row_dict)

results_df = pd.DataFrame(results)
anomalies_df = results_df[results_df["is_anomaly"] == True]
print(anomalies_df)
print(f"Flagged {len(anomalies_df)} out of {len(results_df)} as anomalies")