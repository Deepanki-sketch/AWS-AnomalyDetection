import sys
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np


# =========================================================
# FIND THE MODEL FOLDER
# =========================================================

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "ModelV2"

sys.path.insert(0, str(model_path))


# =========================================================
# IMPORT ART3 MODEL
# =========================================================

from ART3.ARTmodel import (
    AdaptiveModel,
    RollingZScore,
    FEATURES,
    detect_anomaly,
    calculate_sensor_deltas,
    detect_spikes,
    update_frozen_counts,
    detect_persistent_drift,
    check_physical_limits,
    check_dew_point_consistency,
    check_communication,
    detect_simultaneous_shift,
    check_weather_consistency,
    classify_triggers,
)


# =========================================================
# STREAM PROCESSOR
# =========================================================

class StreamProcessor:
    """
    Controls how weather data is sent to the ART3 model.

    ART3 combines:
    - Isolation Forest
    - Rolling Z-Score
    - Physical-limit checks
    - Dew-point consistency
    - Communication-gap checks
    - Frozen-sensor detection
    - Spike detection
    - Persistent-drift detection
    - Simultaneous sensor-shift detection
    - Adaptive Isolation Forest retraining

    The dashboard-facing interface is intentionally kept compatible
    with the previous ART1 StreamProcessor.
    """

    def __init__(
        self,
        df,
        baseline_size=200,
        start_index=200,
        end_index=None,
        batch_size=50
    ):

        # -------------------------------------------------
        # Prepare dataset
        # -------------------------------------------------

        self.df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Store settings
        # -------------------------------------------------

        self.baseline_size = int(baseline_size)
        self.start_index = int(start_index)

        if end_index is None:
            self.end_index = len(self.df)
        else:
            self.end_index = min(int(end_index), len(self.df))

        self.batch_size = int(batch_size)

        # -------------------------------------------------
        # Validate settings
        # -------------------------------------------------

        if self.baseline_size <= 0:
            raise ValueError("Baseline size must be greater than 0.")

        if self.baseline_size > len(self.df):
            raise ValueError("Baseline size cannot be larger than the dataset.")

        if self.start_index < self.baseline_size:
            raise ValueError(
                "Detection start cannot be smaller than the baseline size."
            )

        if self.start_index >= self.end_index:
            raise ValueError("Detection start must be smaller than detection end.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        # -------------------------------------------------
        # Create baseline
        # -------------------------------------------------

        self.baseline_df = (
            self.df.iloc[:self.baseline_size].copy()
        )

        # -------------------------------------------------
        # Create detection stream
        # -------------------------------------------------

        self.stream_df = (
            self.df.iloc[self.start_index:self.end_index].copy()
        )

        # -------------------------------------------------
        # Create ART3 components
        # -------------------------------------------------

        self.adaptive = AdaptiveModel(self.baseline_df)
        self.roller = RollingZScore()

        # ART3 state that must persist across batches.
        self.sensor_history = {
            sensor: deque(maxlen=10)
            for sensor in FEATURES
        }
        self.frozen_counts = {sensor: 0 for sensor in FEATURES}
        self.previous_row = None
        self.previous_timestamp = None

        # -------------------------------------------------
        # Result storage
        # -------------------------------------------------

        self.results = []
        self.batch_history = []

        # -------------------------------------------------
        # Processing position
        # -------------------------------------------------

        self.position = 0
        self.batch_number = 0

        # -------------------------------------------------
        # Current information
        # -------------------------------------------------

        self.last_batch = None
        self.last_anomaly = None

    # =====================================================
    # PROCESS NEXT BATCH
    # =====================================================

    def process_next(self, batch_size=None):
        """Process one batch using the stateful ART3 pipeline."""

        if self.finished():
            return pd.DataFrame(self.results)

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = int(batch_size)

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        # -------------------------------------------------
        # Determine batch boundaries
        # -------------------------------------------------

        batch_start_position = self.position
        end_position = min(
            self.position + batch_size,
            len(self.stream_df)
        )

        current_batch = (
            self.stream_df
            .iloc[self.position:end_position]
            .copy()
        )

        batch_results = []
        model_refitted = False

        # =================================================
        # PROCESS EACH READING
        # =================================================

        for _, row in current_batch.iterrows():
            row_dict = row.to_dict()

            # ---------------------------------------------
            # Isolation Forest prediction
            # ---------------------------------------------

            # ART3's run_art3_pipeline performs these predictions
            # before adaptive training on the current reading.
            model = self.adaptive.get_model()
            values = pd.DataFrame([row_dict])[FEATURES]

            iforest_prediction = int(model.predict(values)[0])
            iforest_score = float(model.decision_function(values)[0])
            iforest_flag = iforest_prediction == -1

            # ---------------------------------------------
            # ML anomaly detection + rolling Z-score
            # ---------------------------------------------

            ml_result = detect_anomaly(
                row_dict,
                iforest_flag,
                iforest_score,
                self.roller
            )

            # ---------------------------------------------
            # ART3 rule-based detectors
            # ---------------------------------------------

            deltas = calculate_sensor_deltas(
                row_dict,
                self.previous_row
            )

            spike_flags = detect_spikes(deltas)
            frozen_flags = update_frozen_counts(
                deltas,
                self.frozen_counts
            )

            for sensor in FEATURES:
                try:
                    value = float(row_dict[sensor])
                except (TypeError, ValueError):
                    value = np.nan
                self.sensor_history[sensor].append(value)

            drift_flags = detect_persistent_drift(
                self.sensor_history,
                ml_result["z_scores"]
            )

            physical_flags = check_physical_limits(row_dict)
            dew_point_issue = check_dew_point_consistency(row_dict)
            communication_issue = check_communication(
                self.previous_timestamp,
                row_dict["timestamp"]
            )
            simultaneous_shift = detect_simultaneous_shift(spike_flags)
            weather_consistent = check_weather_consistency(deltas)

            rule_based_anomaly = bool(
                any(physical_flags.values())
                or dew_point_issue
                or communication_issue
                or any(frozen_flags.values())
                or any(spike_flags.values())
                or any(drift_flags.values())
            )

            is_anomaly = bool(
                ml_result["is_anomaly"] or rule_based_anomaly
            )

            triggers = classify_triggers(
                ml_result,
                physical_flags,
                dew_point_issue,
                communication_issue,
                frozen_flags,
                spike_flags,
                drift_flags,
                simultaneous_shift
            )

            if is_anomaly and not triggers:
                triggers = ["final_anomaly_decision"]

            # ---------------------------------------------
            # Preserve the dashboard-compatible result shape
            # while exposing ART3's additional information.
            # ---------------------------------------------

            result = {
                "timestamp": row_dict.get("timestamp"),
                "is_anomaly": is_anomaly,
                "anomaly_type": (
                    " + ".join(triggers)
                    if is_anomaly else "normal"
                ),
                "iforest_flag": ml_result["iforest_flag"],
                "iforest_score": ml_result["iforest_score"],
                "z_flag": ml_result["z_flag"],
                "zscore_max": ml_result["zscore_max"],
                "z_scores": ml_result["z_scores"],
                "rule_based_flag": rule_based_anomaly,
                "simultaneous_shift": simultaneous_shift,
                "weather_consistent": weather_consistent,
                "physical_flags": physical_flags,
                "dew_point_issue": dew_point_issue,
                "communication_issue": communication_issue,
                "frozen_flags": frozen_flags,
                "spike_flags": spike_flags,
                "drift_flags": drift_flags,
                "triggers": triggers,
            }

            self.results.append(result)
            batch_results.append(result)

            # ---------------------------------------------
            # Save latest anomaly
            # ---------------------------------------------

            if result.get("is_anomaly", False):
                self.last_anomaly = result.copy()

            # ---------------------------------------------
            # Adaptive ART3 model update
            # ---------------------------------------------

            previous_refit_count = self.adaptive.count_since_refit

            self.adaptive.addnrefit(row_dict)

            if (
                previous_refit_count > 0
                and self.adaptive.count_since_refit == 0
            ):
                model_refitted = True

            # ---------------------------------------------
            # Advance state for next reading
            # ---------------------------------------------

            self.previous_row = {
                sensor: row_dict.get(sensor)
                for sensor in FEATURES
            }
            self.previous_timestamp = row_dict.get("timestamp")

        # =================================================
        # UPDATE PROCESSING POSITION
        # =================================================

        self.position = end_position
        self.batch_number += 1

        # =================================================
        # ANALYZE THIS BATCH
        # =================================================

        result_df = pd.DataFrame(batch_results)

        if not result_df.empty:
            anomaly_count = int(result_df["is_anomaly"].sum())
        else:
            anomaly_count = 0

        latest_reading = current_batch.iloc[-1].to_dict()

        anomaly_results = [
            result
            for result in batch_results
            if result.get("is_anomaly", False)
        ]

        batch_status = "ANOMALY" if anomaly_count > 0 else "NORMAL"

        # =================================================
        # CREATE BATCH INFORMATION
        # =================================================

        batch_info = {
            "batch_number": self.batch_number,
            "start_position": batch_start_position,
            "end_position": end_position - 1,
            "readings_processed": len(current_batch),
            "start_timestamp": current_batch.iloc[0]["timestamp"],
            "end_timestamp": current_batch.iloc[-1]["timestamp"],
            "latest_reading": latest_reading,
            "anomaly_count": anomaly_count,
            "status": batch_status,
            "anomalies": anomaly_results,
            "model_refitted": model_refitted,
            "processed_total": self.position,
            "stream_total": len(self.stream_df),
        }

        self.last_batch = batch_info
        self.batch_history.append(batch_info)

        return pd.DataFrame(self.results)

    # =====================================================
    # PROCESS ALL REMAINING DATA
    # =====================================================

    def process_all(self):
        while not self.finished():
            self.process_next()

        return pd.DataFrame(self.results)

    # =====================================================
    # RESET PROCESSING
    # =====================================================

    def reset(self):
        """Reset the ART3 model and all stateful detector history."""

        self.adaptive = AdaptiveModel(self.baseline_df, load_saved=False)
        self.roller = RollingZScore()

        self.sensor_history = {
            sensor: deque(maxlen=10)
            for sensor in FEATURES
        }
        self.frozen_counts = {sensor: 0 for sensor in FEATURES}
        self.previous_row = None
        self.previous_timestamp = None

        self.results = []
        self.batch_history = []
        self.position = 0
        self.batch_number = 0
        self.last_batch = None
        self.last_anomaly = None

    # =====================================================
    # CHECK WHETHER PROCESSING IS FINISHED
    # =====================================================

    def finished(self):
        return self.position >= len(self.stream_df)

    # =====================================================
    # PROCESSING PROGRESS
    # =====================================================

    def progress(self):
        total = len(self.stream_df)
        processed = self.position
        remaining = total - processed
        return processed, total, remaining
