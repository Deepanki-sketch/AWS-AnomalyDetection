import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ModelV2.AdaptiveRTModel import AdaptiveModel, baseline_model, RollingZScore, detect_anomaly, features
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Injecting Labeled Synthetic Anomalies

def inject_spikes(df, n_spikes=10, spike_magnitude=10):
    df_with_spikes = df.copy()

    if "is_anomaly" not in df_with_spikes.columns:
        df_with_spikes["is_anomaly"] = 0
        df_with_spikes["anomaly_type"] = "normal"

    spike_indices = np.random.choice(df_with_spikes.index, size=n_spikes, replace=False)

    for idx in spike_indices:
        df_with_spikes.loc[idx, "temperature_c"] += spike_magnitude * np.random.choice([-1, 1])
        df_with_spikes.loc[idx, "is_anomaly"] = 1
        df_with_spikes.loc[idx, "anomaly_type"] = "spike"

    return df_with_spikes

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nConfusion matrix (rows=actual, cols=predicted) [0, 1]:")
    print(cm)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "ModelV2", "TestAnomalyUnlabled.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Split BEFORE injecting anomalies:
    # baseline_df -> clean data, used only to train/warm up the model
    # stream_df   -> separate chunk, gets spikes injected, used to TEST detection
    #ts sepration is still to be decided 
    baseline_df = df.iloc[:400].reset_index(drop=True)
    stream_df_raw = df.iloc[400:1000].reset_index(drop=True)   # 400 rows to stream through

    n_spikes = 15
    stream_df = inject_spikes(stream_df_raw, n_spikes=n_spikes, spike_magnitude=10)

    print(f"Injected {n_spikes} spikes into {len(stream_df)} stream rows.\n")

    # Build the adaptive model on CLEAN baseline only
    adaptive = AdaptiveModel(baseline_df)
    roller = RollingZScore()

    results = []
    for _, row in stream_df.iterrows():
        row_dict = row.to_dict()
        result = detect_anomaly(row_dict, adaptive.get_model(), roller)
        results.append(result)
        adaptive.addnrefit(row_dict)

    results_df = pd.DataFrame(results)

    # Compare model predictions against the ground truth we injected
    y_true = stream_df["is_anomaly"].astype(int).values
    y_pred = results_df["is_anomaly"].astype(int).values

    caught = int(np.sum((y_true == 1) & (y_pred == 1)))
    total_injected = int(np.sum(y_true == 1))
    catch_pct = (caught / total_injected * 100) if total_injected > 0 else 0.0

    print(f"Caught {caught} out of {total_injected} injected spikes ({catch_pct:.1f}%)")
    print(f"Flagged {int(y_pred.sum())} rows as anomalies out of {len(y_pred)} total rows.\n")

    evaluate(y_true, y_pred)