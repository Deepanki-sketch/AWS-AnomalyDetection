import os
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# ART MODEL - AWS WEATHER ANOMALY DETECTION
#
# Location:
# ModelV2/
# ├── AdaptiveRTModel.py
# ├── TestAnomalyUnlabled.csv
# └── ART3/
#     ├── ARTmodel.py
#     └── AWS_Weather_5000_With_Anomalies.csv
#
# INPUT:
#     AWS_Weather_5000_With_Anomalies.csv
#
# BASELINE:
#     TestAnomalyUnlabled.csv
#
# IMPORTANT:
#     The is_anomaly and anomaly_type columns in the 5000-row file
#     are NOT used by the model.
#
# FINAL SCREEN OUTPUT:
#     Only anomaly rows:
#     timestamp | is_anomaly | anomaly_type
#
# NO extra dataset/index/result file is created.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL2_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
)

ANOMALY_CSV = os.path.join(
    BASE_DIR,
    "AWS_Weather_5000_With_Anomalies.csv"
)

CLEAN_BASELINE_CSV = os.path.join(
    MODEL2_DIR,
    "TestAnomalyUnlabled.csv"
)


# ============================================================
# SENSOR FEATURES
# ============================================================

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]


# ============================================================
# ISOLATION FOREST
# ============================================================

IF_CONTAMINATION = 0.01
IF_RANDOM_STATE = 42


# ============================================================
# ROLLING Z-SCORE
# ============================================================

ROLLING_WINDOW = 30
Z_MIN_HISTORY = 5
Z_THRESHOLD = 3.0


# ============================================================
# SUDDEN CHANGE / SPIKE THRESHOLDS
# ============================================================

TEMP_SPIKE_THRESHOLD = 1.1677
HUMIDITY_SPIKE_THRESHOLD = 3.4070
PRESSURE_SPIKE_THRESHOLD = 1.5923


# ============================================================
# FROZEN SENSOR
# ============================================================

FROZEN_MIN_COUNT = 5
FROZEN_TOLERANCE = 0.001


# ============================================================
# PERSISTENT DRIFT
# ============================================================

DRIFT_WINDOW = 10
DRIFT_MIN_COUNT = 8
DRIFT_Z_THRESHOLD = 2.0

DRIFT_NET_CHANGE = {
    "temperature_c": 2.0,
    "humidity_pct": 8.0,
    "pressure_hpa": 3.0
}


# ============================================================
# PHYSICAL LIMITS
# ============================================================

PHYSICAL_LIMITS = {
    "temperature_c": (-80.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (800.0, 1100.0)
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return np.nan


def sensor_name(sensor):

    return (
        sensor
        .replace("_c", "")
        .replace("_pct", "")
        .replace("_hpa", "")
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

def train_isolation_forest(
    baseline_df
):

    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        baseline_df[FEATURES]
    )

    return model


# ============================================================
# ROLLING Z-SCORE
# ============================================================

class RollingZScore:

    def __init__(
        self,
        window=ROLLING_WINDOW
    ):

        self.window = window

        self.buffers = {
            sensor: deque(
                maxlen=window
            )
            for sensor in FEATURES
        }


    def calculate(
        self,
        row
    ):

        z_scores = {}

        for sensor in FEATURES:

            buffer = self.buffers[
                sensor
            ]

            if len(buffer) >= Z_MIN_HISTORY:

                mean = float(
                    np.mean(buffer)
                )

                std = float(
                    np.std(buffer)
                )

                if std <= 1e-12:
                    std = 1e-6

                z_scores[sensor] = (
                    safe_float(
                        row[sensor]
                    )
                    - mean
                ) / std

            else:

                z_scores[sensor] = 0.0

        return z_scores


    def update(
        self,
        row
    ):

        for sensor in FEATURES:

            self.buffers[
                sensor
            ].append(
                safe_float(
                    row[sensor]
                )
            )


# ============================================================
# MAIN ANOMALY DETECTOR
# ============================================================

def detect_anomaly(
    row,
    iforest_flag,
    iforest_score,
    roller
):

    z_scores = roller.calculate(
        row
    )

    zscore_max = max(
        abs(value)
        for value in z_scores.values()
    )

    z_flag = (
        zscore_max
        > Z_THRESHOLD
    )

    is_anomaly = bool(
        iforest_flag
        or z_flag
    )

    roller.update(
        row
    )

    return {
        "is_anomaly": is_anomaly,
        "iforest_flag": bool(
            iforest_flag
        ),
        "iforest_score": float(
            iforest_score
        ),
        "z_flag": bool(
            z_flag
        ),
        "zscore_max": float(
            zscore_max
        ),
        "z_scores": z_scores
    }


# ============================================================
# SENSOR DELTA
# ============================================================

def calculate_sensor_deltas(
    row,
    previous_row
):

    if previous_row is None:

        return {
            sensor: 0.0
            for sensor in FEATURES
        }

    return {
        sensor:
            safe_float(
                row[sensor]
            )
            -
            safe_float(
                previous_row[sensor]
            )
        for sensor in FEATURES
    }


# ============================================================
# SPIKE DETECTOR
# ============================================================

def detect_spikes(
    deltas
):

    return {
        "temperature_c":
            abs(
                deltas[
                    "temperature_c"
                ]
            )
            > TEMP_SPIKE_THRESHOLD,

        "humidity_pct":
            abs(
                deltas[
                    "humidity_pct"
                ]
            )
            > HUMIDITY_SPIKE_THRESHOLD,

        "pressure_hpa":
            abs(
                deltas[
                    "pressure_hpa"
                ]
            )
            > PRESSURE_SPIKE_THRESHOLD
    }


# ============================================================
# FROZEN SENSOR DETECTOR
# ============================================================

def update_frozen_counts(
    deltas,
    frozen_counts
):

    for sensor in FEATURES:

        if (
            abs(
                deltas[sensor]
            )
            <= FROZEN_TOLERANCE
        ):

            frozen_counts[
                sensor
            ] += 1

        else:

            frozen_counts[
                sensor
            ] = 0

    return {
        sensor:
            frozen_counts[sensor]
            >= FROZEN_MIN_COUNT
        for sensor in FEATURES
    }


# ============================================================
# PERSISTENT DRIFT DETECTOR
# ============================================================

def detect_persistent_drift(
    sensor_history,
    z_scores
):

    drift_flags = {
        sensor: False
        for sensor in FEATURES
    }

    for sensor in FEATURES:

        history = sensor_history[
            sensor
        ]

        if (
            len(history)
            < DRIFT_WINDOW
        ):
            continue

        values = np.asarray(
            list(history),
            dtype=float
        )

        changes = np.diff(
            values
        )

        positive = int(
            np.sum(
                changes > 0
            )
        )

        negative = int(
            np.sum(
                changes < 0
            )
        )

        same_direction = max(
            positive,
            negative
        )

        net_change = abs(
            values[-1]
            - values[0]
        )

        z = z_scores[
            sensor
        ]

        if (
            same_direction
            >= DRIFT_MIN_COUNT

            and net_change
            >= DRIFT_NET_CHANGE[
                sensor
            ]

            and np.isfinite(z)

            and abs(z)
            >= DRIFT_Z_THRESHOLD
        ):

            drift_flags[
                sensor
            ] = True

    return drift_flags


# ============================================================
# PHYSICAL LIMIT CHECK
# ============================================================

def check_physical_limits(
    row
):

    flags = {}

    for sensor, (
        low,
        high
    ) in PHYSICAL_LIMITS.items():

        value = safe_float(
            row[sensor]
        )

        flags[sensor] = (
            not np.isfinite(
                value
            )
            or value < low
            or value > high
        )

    return flags


# ============================================================
# DEW POINT CHECK
# ============================================================

def calculate_dew_point(
    temperature_c,
    humidity_pct
):

    temperature = safe_float(
        temperature_c
    )

    humidity = safe_float(
        humidity_pct
    )

    if (
        not np.isfinite(
            temperature
        )
        or not np.isfinite(
            humidity
        )
    ):
        return np.nan

    if (
        humidity <= 0
        or humidity > 100
    ):
        return np.nan

    a = 17.27
    b = 237.7

    gamma = (
        (a * temperature)
        / (b + temperature)
        + np.log(
            humidity / 100.0
        )
    )

    denominator = (
        a - gamma
    )

    if abs(
        denominator
    ) < 1e-12:

        return np.nan

    return (
        b * gamma
    ) / denominator


def check_dew_point_consistency(
    row
):

    temperature = safe_float(
        row[
            "temperature_c"
        ]
    )

    humidity = safe_float(
        row[
            "humidity_pct"
        ]
    )

    dew_point = calculate_dew_point(
        temperature,
        humidity
    )

    if not np.isfinite(
        dew_point
    ):
        return False

    return bool(
        dew_point
        > temperature + 0.2
    )


# ============================================================
# COMMUNICATION CHECK
# ============================================================

def check_communication(
    previous_timestamp,
    current_timestamp
):

    if previous_timestamp is None:
        return False

    difference = (
        current_timestamp
        - previous_timestamp
    )

    return bool(
        difference
        > pd.Timedelta(
            minutes=1
        )
        or difference
        <= pd.Timedelta(0)
    )


# ============================================================
# SIMULTANEOUS SHIFT
# ============================================================

def detect_simultaneous_shift(
    spike_flags
):

    return bool(
        sum(
            spike_flags.values()
        ) == 3
    )


# ============================================================
# WEATHER CONSISTENCY
# ============================================================

def check_weather_consistency(
    deltas
):

    dt = deltas[
        "temperature_c"
    ]

    dh = deltas[
        "humidity_pct"
    ]

    dp = deltas[
        "pressure_hpa"
    ]

    return bool(
        dt < -0.5
        and dh > 0.2
        and dp < 0
    )


# ============================================================
# LOAD ANOMALOUS INPUT FILE
# ============================================================

if not os.path.isfile(
    ANOMALY_CSV
):

    raise FileNotFoundError(
        "\nCould not find:\n"
        f"{ANOMALY_CSV}\n\n"
        "ARTmodel.py and "
        "AWS_Weather_5000_With_Anomalies.csv "
        "must be in the same ART3 folder."
    )


df = pd.read_csv(
    ANOMALY_CSV
)


# ============================================================
# LOAD CLEAN BASELINE
# ============================================================

if not os.path.isfile(
    CLEAN_BASELINE_CSV
):

    raise FileNotFoundError(
        "\nCould not find the clean baseline:\n"
        f"{CLEAN_BASELINE_CSV}\n\n"
        "Expected TestAnomalyUnlabled.csv "
        "inside ModelV2."
    )


baseline_df = pd.read_csv(
    CLEAN_BASELINE_CSV
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = {
    "timestamp",
    *FEATURES
}

missing_input = (
    required_columns
    - set(df.columns)
)

if missing_input:

    raise ValueError(
        "AWS anomaly CSV is missing: "
        f"{sorted(missing_input)}"
    )


missing_baseline = (
    set(FEATURES)
    - set(baseline_df.columns)
)

if missing_baseline:

    raise ValueError(
        "Clean baseline CSV is missing: "
        f"{sorted(missing_baseline)}"
    )


# ============================================================
# CLEAN DATA TYPES
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

baseline_df["timestamp"] = pd.to_datetime(
    baseline_df["timestamp"],
    errors="coerce"
)


if df["timestamp"].isna().any():

    raise ValueError(
        "Invalid timestamp found in "
        "AWS_Weather_5000_With_Anomalies.csv."
    )


for sensor in FEATURES:

    df[sensor] = pd.to_numeric(
        df[sensor],
        errors="coerce"
    )

    baseline_df[sensor] = pd.to_numeric(
        baseline_df[sensor],
        errors="coerce"
    )


if df[FEATURES].isna().any().any():

    raise ValueError(
        "AWS anomaly CSV contains "
        "missing/non-numeric sensor values."
    )


if baseline_df[FEATURES].isna().any().any():

    raise ValueError(
        "Clean baseline contains "
        "missing/non-numeric sensor values."
    )


df = (
    df.sort_values(
        "timestamp"
    )
    .reset_index(
        drop=True
    )
)

baseline_df = (
    baseline_df.sort_values(
        "timestamp"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# TRAIN ML MODEL
# ============================================================

model = train_isolation_forest(
    baseline_df
)


# ============================================================
# BATCH ISOLATION FOREST PREDICTION
# ============================================================

iforest_predictions = (
    model.predict(
        df[FEATURES]
    )
)

iforest_scores = (
    model.decision_function(
        df[FEATURES]
    )
)

iforest_flags = (
    iforest_predictions == -1
)


# ============================================================
# INITIALISE REAL-TIME DETECTORS
# ============================================================

roller = RollingZScore()

sensor_history = {
    sensor: deque(
        maxlen=DRIFT_WINDOW
    )
    for sensor in FEATURES
}

frozen_counts = {
    sensor: 0
    for sensor in FEATURES
}

previous_row = None
previous_timestamp = None

results = []


# ============================================================
# PROCESS THE 5000 AWS READINGS
# ============================================================

for position, series in df.iterrows():

    row = series.to_dict()

    # --------------------------------------------------------
    # Isolation Forest + Rolling Z-score
    # --------------------------------------------------------

    ml_result = detect_anomaly(
        row=row,

        iforest_flag=bool(
            iforest_flags[
                position
            ]
        ),

        iforest_score=float(
            iforest_scores[
                position
            ]
        ),

        roller=roller
    )


    # --------------------------------------------------------
    # Sensor changes
    # --------------------------------------------------------

    deltas = calculate_sensor_deltas(
        row,
        previous_row
    )


    # --------------------------------------------------------
    # Spike detector
    # --------------------------------------------------------

    spike_flags = detect_spikes(
        deltas
    )


    # --------------------------------------------------------
    # Frozen sensor detector
    # --------------------------------------------------------

    frozen_flags = (
        update_frozen_counts(
            deltas,
            frozen_counts
        )
    )


    # --------------------------------------------------------
    # Sensor history
    # --------------------------------------------------------

    for sensor in FEATURES:

        sensor_history[
            sensor
        ].append(
            safe_float(
                row[sensor]
            )
        )


    # --------------------------------------------------------
    # Persistent drift detector
    # --------------------------------------------------------

    drift_flags = (
        detect_persistent_drift(
            sensor_history,
            ml_result[
                "z_scores"
            ]
        )
    )


    # --------------------------------------------------------
    # Physical checks
    # --------------------------------------------------------

    physical_flags = (
        check_physical_limits(
            row
        )
    )


    dew_point_issue = (
        check_dew_point_consistency(
            row
        )
    )


    communication_issue = (
        check_communication(
            previous_timestamp,
            row["timestamp"]
        )
    )


    # --------------------------------------------------------
    # Multi-sensor check
    # --------------------------------------------------------

    simultaneous_shift = (
        detect_simultaneous_shift(
            spike_flags
        )
    )


    # --------------------------------------------------------
    # Weather consistency
    # --------------------------------------------------------

    weather_consistent = (
        check_weather_consistency(
            deltas
        )
    )

    # Weather consistency is contextual only.
    # It does NOT create an anomaly by itself.
    _ = weather_consistent


    # --------------------------------------------------------
    # FINAL ANOMALY DECISION
    # --------------------------------------------------------

    rule_based_anomaly = bool(

        any(
            physical_flags.values()
        )

        or dew_point_issue

        or communication_issue

        or any(
            frozen_flags.values()
        )

        or any(
            spike_flags.values()
        )

        or any(
            drift_flags.values()
        )
    )


    is_anomaly = bool(
        ml_result[
            "is_anomaly"
        ]
        or rule_based_anomaly
    )


    # --------------------------------------------------------
    # IDENTIFY FUNCTION THAT TRIGGERED ANOMALY
    # --------------------------------------------------------

    triggers = []


    if ml_result[
        "iforest_flag"
    ]:

        triggers.append(
            "IsolationForest"
        )


    if ml_result[
        "z_flag"
    ]:

        triggers.append(
            "RollingZScore"
        )


    physical_sensors = [
        sensor_name(
            sensor
        )
        for sensor, flag
        in physical_flags.items()
        if flag
    ]

    if physical_sensors:

        triggers.append(
            "check_physical_limits("
            + ",".join(
                physical_sensors
            )
            + ")"
        )


    if dew_point_issue:

        triggers.append(
            "check_dew_point_consistency"
        )


    if communication_issue:

        triggers.append(
            "check_communication"
        )


    frozen_sensors = [
        sensor_name(
            sensor
        )
        for sensor, flag
        in frozen_flags.items()
        if flag
    ]

    if frozen_sensors:

        triggers.append(
            "update_frozen_counts("
            + ",".join(
                frozen_sensors
            )
            + ")"
        )


    spike_sensors = [
        sensor_name(
            sensor
        )
        for sensor, flag
        in spike_flags.items()
        if flag
    ]

    if spike_sensors:

        triggers.append(
            "detect_spikes("
            + ",".join(
                spike_sensors
            )
            + ")"
        )


    drift_sensors = [
        sensor_name(
            sensor
        )
        for sensor, flag
        in drift_flags.items()
        if flag
    ]

    if drift_sensors:

        triggers.append(
            "detect_persistent_drift("
            + ",".join(
                drift_sensors
            )
            + ")"
        )


    if simultaneous_shift:

        triggers.append(
            "detect_simultaneous_shift"
        )


    # --------------------------------------------------------
    # SAVE ONLY ANOMALY ROWS TO MEMORY FOR DISPLAY
    # --------------------------------------------------------

    if is_anomaly:

        if not triggers:

            triggers = [
                "final_anomaly_decision"
            ]

        results.append({

            "timestamp":
                row["timestamp"],

            "is_anomaly":
                True,

            "anomaly_type":
                " + ".join(
                    triggers
                )
        })


    # --------------------------------------------------------
    # UPDATE PREVIOUS READING
    # --------------------------------------------------------

    previous_row = {
        sensor:
            safe_float(
                row[sensor]
            )
        for sensor in FEATURES
    }

    previous_timestamp = (
        row["timestamp"]
    )


# ============================================================
# FINAL DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "timestamp",
        "is_anomaly",
        "anomaly_type"
    ]
)


# ============================================================
# PRINT ONLY ANOMALIES
# ============================================================

print("\n")
print("=" * 110)
print("ART MODEL - AWS WEATHER ANOMALY DETECTION")
print("=" * 110)

print(
    f"Input file        : "
    f"{os.path.basename(ANOMALY_CSV)}"
)

print(
    f"Total readings    : "
    f"{len(df)}"
)

print(
    f"Anomalies detected: "
    f"{len(results_df)}"
)

print("=" * 110)


if results_df.empty:

    print(
        "No anomalies detected."
    )

else:

    print(
        results_df.to_string(
            index=False
        )
    )


print("=" * 110)

