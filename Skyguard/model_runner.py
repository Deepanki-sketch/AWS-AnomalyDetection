import sys
from pathlib import Path

import pandas as pd


# =========================================================
# FIND THE MODELV2 FOLDER
# =========================================================

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "ModelV2"

sys.path.insert(0, str(model_path))


# =========================================================
# IMPORT EXISTING ART1 MODEL
# =========================================================

from AdaptiveRTModel import (
    AdaptiveModel,
    RollingZScore,
    detect_anomaly
)


# =========================================================
# STREAM PROCESSOR
# =========================================================

class StreamProcessor:
    """
    Controls how weather data is sent to the existing ART1 model.

    This class does NOT perform anomaly detection itself.

    AdaptiveRTModel.py remains responsible for:
    - Isolation Forest
    - Rolling Z-Score
    - Anomaly detection
    - Adaptive model retraining
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
            self.end_index = min(
                int(end_index),
                len(self.df)
            )

        self.batch_size = int(batch_size)

        # -------------------------------------------------
        # Validate settings
        # -------------------------------------------------

        if self.baseline_size <= 0:

            raise ValueError(
                "Baseline size must be greater than 0."
            )

        if self.baseline_size > len(self.df):

            raise ValueError(
                "Baseline size cannot be larger "
                "than the dataset."
            )

        if self.start_index < self.baseline_size:

            raise ValueError(
                "Detection start cannot be smaller "
                "than the baseline size."
            )

        if self.start_index >= self.end_index:

            raise ValueError(
                "Detection start must be smaller "
                "than detection end."
            )

        if self.batch_size <= 0:

            raise ValueError(
                "Batch size must be greater than 0."
            )

        # =================================================
        # CREATE BASELINE
        # =================================================

        self.baseline_df = (
            self.df
            .iloc[:self.baseline_size]
            .copy()
        )

        # =================================================
        # CREATE DETECTION STREAM
        # =================================================

        self.stream_df = (
            self.df
            .iloc[
                self.start_index:self.end_index
            ]
            .copy()
        )

        # =================================================
        # CREATE ART1 COMPONENTS
        # =================================================

        self.adaptive = AdaptiveModel(
            self.baseline_df
        )

        self.roller = RollingZScore()

        # =================================================
        # RESULT STORAGE
        # =================================================

        # Every individual detection result
        self.results = []

        # Information about every processed batch
        self.batch_history = []

        # =================================================
        # PROCESSING POSITION
        # =================================================

        # Position inside stream_df
        self.position = 0

        # Number of batches processed
        self.batch_number = 0

        # =================================================
        # CURRENT INFORMATION
        # =================================================

        # Most recently processed batch
        self.last_batch = None

        # Most recently detected anomaly
        self.last_anomaly = None


    # =====================================================
    # PROCESS NEXT BATCH
    # =====================================================

    def process_next(self, batch_size=None):
        """
        Process one batch of weather readings.

        The actual anomaly detection is still performed
        by detect_anomaly() from AdaptiveRTModel.py.
        """

        # -------------------------------------------------
        # Check whether processing is finished
        # -------------------------------------------------

        if self.finished():

            return pd.DataFrame(
                self.results
            )

        # -------------------------------------------------
        # Use configured batch size
        # -------------------------------------------------

        if batch_size is None:

            batch_size = self.batch_size

        else:

            batch_size = int(batch_size)

        if batch_size <= 0:

            raise ValueError(
                "Batch size must be greater than 0."
            )

        # -------------------------------------------------
        # Remember where this batch starts
        # -------------------------------------------------

        batch_start_position = self.position

        # -------------------------------------------------
        # Calculate where this batch ends
        # -------------------------------------------------

        end_position = min(
            self.position + batch_size,
            len(self.stream_df)
        )

        # -------------------------------------------------
        # Extract current batch
        # -------------------------------------------------

        current_batch = (
            self.stream_df
            .iloc[
                self.position:end_position
            ]
            .copy()
        )

        # -------------------------------------------------
        # Store results for ONLY this batch
        # -------------------------------------------------

        batch_results = []

        # -------------------------------------------------
        # Track whether ART1 retrained
        # -------------------------------------------------

        model_refitted = False

        # =================================================
        # PROCESS EACH READING
        # =================================================

        for _, row in current_batch.iterrows():

            # Convert pandas row to dictionary
            row_dict = row.to_dict()

            # ---------------------------------------------
            # RUN EXISTING ANOMALY DETECTOR
            # ---------------------------------------------

            result = detect_anomaly(
                row_dict,
                self.adaptive.get_model(),
                self.roller
            )

            # ---------------------------------------------
            # Store individual result
            # ---------------------------------------------

            self.results.append(
                result
            )

            batch_results.append(
                result
            )

            # ---------------------------------------------
            # Save latest anomaly
            # ---------------------------------------------

            if result.get(
                "is_anomaly",
                False
            ):

                self.last_anomaly = (
                    result.copy()
                )

            # ---------------------------------------------
            # Remember refit counter
            # ---------------------------------------------

            previous_refit_count = (
                self.adaptive.count_since_refit
            )

            # ---------------------------------------------
            # Update adaptive ART1 model
            # ---------------------------------------------

            self.adaptive.addnrefit(
                row_dict
            )

            # ---------------------------------------------
            # Detect whether retraining happened
            # ---------------------------------------------

            if (
                previous_refit_count > 0
                and
                self.adaptive.count_since_refit == 0
            ):

                model_refitted = True

        # =================================================
        # UPDATE PROCESSING POSITION
        # =================================================

        self.position = end_position

        self.batch_number += 1

        # =================================================
        # ANALYZE THIS BATCH
        # =================================================

        result_df = pd.DataFrame(
            batch_results
        )

        if not result_df.empty:

            anomaly_count = int(
                result_df["is_anomaly"].sum()
            )

        else:

            anomaly_count = 0

        # -------------------------------------------------
        # Latest reading in this batch
        # -------------------------------------------------

        latest_reading = (
            current_batch
            .iloc[-1]
            .to_dict()
        )

        # -------------------------------------------------
        # All anomalies in this batch
        # -------------------------------------------------

        anomaly_results = [

            result

            for result in batch_results

            if result.get(
                "is_anomaly",
                False
            )

        ]

        # =================================================
        # BATCH STATUS
        # =================================================

        if anomaly_count > 0:

            batch_status = "ANOMALY"

        else:

            batch_status = "NORMAL"

        # =================================================
        # CREATE BATCH INFORMATION
        # =================================================

        batch_info = {

            # Basic batch information
            "batch_number":
                self.batch_number,

            "start_position":
                batch_start_position,

            "end_position":
                end_position - 1,

            "readings_processed":
                len(current_batch),

            # Time information
            "start_timestamp":
                current_batch.iloc[0]["timestamp"],

            "end_timestamp":
                current_batch.iloc[-1]["timestamp"],

            # Latest sensor reading
            "latest_reading":
                latest_reading,

            # Detection information
            "anomaly_count":
                anomaly_count,

            "status":
                batch_status,

            "anomalies":
                anomaly_results,

            # Model information
            "model_refitted":
                model_refitted,

            # Processing information
            "processed_total":
                self.position,

            "stream_total":
                len(self.stream_df)
        }

        # =================================================
        # SAVE BATCH HISTORY
        # =================================================

        self.last_batch = batch_info

        self.batch_history.append(
            batch_info
        )

        # =================================================
        # RETURN ALL RESULTS SO FAR
        # =================================================

        return pd.DataFrame(
            self.results
        )


    # =====================================================
    # PROCESS ALL REMAINING DATA
    # =====================================================

    def process_all(self):

        while not self.finished():

            self.process_next()

        return pd.DataFrame(
            self.results
        )


    # =====================================================
    # RESET PROCESSING
    # =====================================================

    def reset(self):

        # -------------------------------------------------
        # Create a fresh ART1 model
        # -------------------------------------------------

        self.adaptive = AdaptiveModel(
            self.baseline_df
        )

        # -------------------------------------------------
        # Create fresh rolling Z-score
        # -------------------------------------------------

        self.roller = RollingZScore()

        # -------------------------------------------------
        # Clear results
        # -------------------------------------------------

        self.results = []

        # -------------------------------------------------
        # Clear batch history
        # -------------------------------------------------

        self.batch_history = []

        # -------------------------------------------------
        # Reset position
        # -------------------------------------------------

        self.position = 0

        # -------------------------------------------------
        # Reset batch number
        # -------------------------------------------------

        self.batch_number = 0

        # -------------------------------------------------
        # Clear latest information
        # -------------------------------------------------

        self.last_batch = None

        self.last_anomaly = None


    # =====================================================
    # CHECK WHETHER PROCESSING IS FINISHED
    # =====================================================

    def finished(self):

        return (
            self.position
            >= len(self.stream_df)
        )


    # =====================================================
    # PROCESSING PROGRESS
    # =====================================================

    def progress(self):

        total = len(
            self.stream_df
        )

        processed = self.position

        remaining = (
            total
            - processed
        )

        return (
            processed,
            total,
            remaining
        )
