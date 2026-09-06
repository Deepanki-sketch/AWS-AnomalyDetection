import streamlit as st
import time
import pandas as pd
from pathlib import Path

from model_runner import StreamProcessor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# PAGE STYLING
# =========================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.1rem;
        }

        .subtitle {
            color: #888888;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .batch-title {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .small-text {
            color: #888888;
            font-size: 0.88rem;
        }

        .big-number {
            font-size: 1.5rem;
            font-weight: 700;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASETS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV_PATH = PROJECT_ROOT / "Model Evaluation & Testing" / "AWS_Weather_5000_Clean.csv"
STREAM_CSV_PATH = PROJECT_ROOT / "ModelV2" / "ART3" / "AWS_Weather_5000_With_Anomalies.csv"


@st.cache_data
def load_data():

    clean_df = pd.read_csv(CLEAN_CSV_PATH)
    stream_df = pd.read_csv(STREAM_CSV_PATH)

    for name, frame in (("clean training", clean_df), ("stream", stream_df)):
        if "timestamp" not in frame.columns:
            raise ValueError(f"{name.title()} CSV must contain a timestamp column.")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)

    return clean_df, stream_df


try:
    clean_df, df = load_data()
except Exception as e:
    st.error(f"Could not load datasets: {e}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "processor" not in st.session_state:

    st.session_state.processor = None


if "visible_blocks" not in st.session_state:

    # Number of newest batches shown in the live feed.
    # Older batches are revealed with the bottom arrow.
    st.session_state.visible_blocks = 1


if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "stream_speed" not in st.session_state:
    st.session_state.stream_speed = 0.6


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Processing Controls")

    st.caption(
        "Configure the data stream before starting "
        "anomaly detection."
    )

    st.divider()

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    baseline_size = st.number_input(
        "Baseline Size",
        min_value=1,
        max_value=len(clean_df),
        value=len(clean_df),
        step=100,

        help=(
            "Number of initial readings used "
            "to train the ART3 model."
        )
    )

    # -----------------------------------------------------
    # Detection start
    # -----------------------------------------------------

    start_index = st.number_input(
        "Detection Start",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        step=10,

        help=(
            "Index in the anomaly stream from which "
            "live anomaly detection begins."
        )
    )

    # -----------------------------------------------------
    # Detection end
    # -----------------------------------------------------

    end_index = st.number_input(
        "Detection End",
        min_value=1,
        max_value=len(df),
        value=len(df),
        step=10,

        help=(
            "Dataset index at which anomaly "
            "detection stops."
        )
    )

    # -----------------------------------------------------
    # Batch size
    # -----------------------------------------------------

    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=len(df),
        value=50,
        step=10,

        help=(
            "Internal grouping size used for the Live "
            "Processing Feed. Live mode still processes one "
            "telemetry reading at a time."
        )
    )

    st.divider()

    # -----------------------------------------------------
    # Buttons
    # -----------------------------------------------------

    apply_settings = st.button(
        "Apply Settings",
        use_container_width=True,
        type="primary"
    )

    process_next = st.button(
        "Process Next Batch",
        use_container_width=True
    )

    process_all = st.button(
        "Process All",
        use_container_width=True
    )

    reset_detection = st.button(
        "Reset Detection",
        use_container_width=True
    )

    st.divider()

    st.subheader("Live Telemetry")

    live_col1, live_col2 = st.columns(2)

    with live_col1:
        start_stream = st.button(
            "▶ Start Stream" if not st.session_state.is_streaming else "⏸ Streaming...",
            use_container_width=True,
            disabled=st.session_state.is_streaming
        )

    with live_col2:
        pause_stream = st.button(
            "⏸ Pause",
            use_container_width=True,
            disabled=not st.session_state.is_streaming
        )

    stream_speed = st.slider(
        "Stream Speed (seconds / reading)",
        min_value=0.2,
        max_value=2.0,
        value=float(st.session_state.stream_speed),
        step=0.1
    )
    st.session_state.stream_speed = stream_speed

    if start_stream:
        # Starting Live Stream also initializes the ART3 processor when the
        # user has not manually pressed Apply Settings first.
        if st.session_state.processor is None:
            try:
                st.session_state.processor = StreamProcessor(
                    df=df,
                    baseline_size=baseline_size,
                    start_index=start_index,
                    end_index=end_index,
                    batch_size=batch_size,
                    baseline_df=clean_df
                )
                st.session_state.visible_blocks = 1
            except Exception as e:
                st.error(f"Could not start the live stream: {e}")
                st.stop()

        st.session_state.is_streaming = True
        st.rerun()

    if pause_stream:
        st.session_state.is_streaming = False
        st.rerun()

    st.divider()

    st.caption(
        f"Dataset Records: {len(df)}"
    )

    st.caption(
        "Detection Model: ART3"
    )


# =========================================================
# APPLY SETTINGS
# =========================================================

if apply_settings:

    try:

        st.session_state.is_streaming = False

        st.session_state.processor = StreamProcessor(
            df=df,

            baseline_size=baseline_size,

            start_index=start_index,

            end_index=end_index,

            batch_size=batch_size,

            baseline_df=clean_df
        )

        st.session_state.visible_blocks = 1

        st.success(
            "Processing settings applied."
        )

    except Exception as e:

        st.error(str(e))


# =========================================================
# RESET DETECTION
# =========================================================

if reset_detection:

    st.session_state.is_streaming = False

    if st.session_state.processor is not None:

        st.session_state.processor.reset()

    st.session_state.visible_blocks = 1

    st.info(
        "Detection has been reset."
    )


# =========================================================
# PROCESS NEXT BATCH
# =========================================================

if process_next:

    if st.session_state.processor is None:

        st.warning(
            "Apply Settings before processing data."
        )

    elif st.session_state.processor.finished():

        st.info(
            "All selected readings have already "
            "been processed."
        )

    else:

        st.session_state.processor.process_next()


# =========================================================
# PROCESS ALL
# =========================================================

if process_all:

    if st.session_state.processor is None:

        st.warning(
            "Apply Settings before processing data."
        )

    else:

        st.session_state.processor.process_all()


# =========================================================
# GET PROCESSOR
# =========================================================

processor = st.session_state.processor


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">SKYGUARD AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Intelligent Weather Anomaly Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LIVE PROCESSING FEED
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Live Processing Feed'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# NO PROCESSOR YET
# =========================================================

if processor is None:

    st.info(
        "Configure the processing settings from the "
        "sidebar and click Apply Settings."
    )


# =========================================================
# PROCESSOR EXISTS
# =========================================================

else:

    processed, total, remaining = (
        processor.progress()
    )

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    progress_value = (

        processed / total

        if total > 0

        else 0
    )

    st.progress(
        progress_value,

        text=(
            f"{processed} of {total} "
            f"detection readings processed"
        )
    )

    if st.session_state.is_streaming:
        st.info(
            f"Live stream running — processing one telemetry reading every "
            f"{st.session_state.stream_speed:.1f}s"
        )
    else:
        st.caption("Live stream paused. Processing position is preserved.")

    # -----------------------------------------------------
    # Batch history
    # -----------------------------------------------------

    # Reverse the history so the newest processed batch is
    # ALWAYS displayed first.
    batches = list(reversed(processor.batch_history))

    if not batches:

        st.info(
            "No readings have been processed yet. "
            "Press Start Stream to begin simulated live telemetry, or use Process Next Batch for manual testing."
        )

    else:

        # Keep the complete Live Processing Feed in one
        # fixed-height area so the rest of the dashboard
        # does not move down every time a new batch arrives.
        feed_container = st.container(
            height=680,
            border=True
        )

        with feed_container:

            visible_batches = batches[
                :st.session_state.visible_blocks
            ]

            # =================================================
            # DISPLAY NEWEST BATCH FIRST
            # =================================================

            for batch in visible_batches:

                # -------------------------------------------------
                # CARD
                # -------------------------------------------------

                with st.container(border=True):

                    latest = batch["latest_reading"]
                    status = batch["status"]

                    # -------------------------------------------------
                    # Batch heading
                    # -------------------------------------------------

                    heading_col, status_col = (
                        st.columns([4, 1])
                    )

                    with heading_col:

                        st.markdown(
                            f"### Processing Batch "
                            f"{batch['batch_number']}"
                        )

                        st.caption(
                            f"{batch['start_timestamp']}"
                            f" → "
                            f"{batch['end_timestamp']}"
                            f"   •   "
                            f"{batch['readings_processed']} readings"
                        )

                    with status_col:

                        if status == "ANOMALY":

                            st.error(
                                "ANOMALY DETECTED"
                            )

                        else:

                            st.success(
                                "NORMAL"
                            )

                    st.divider()

                    # -------------------------------------------------
                    # Sensor readings
                    # -------------------------------------------------

                    temp_col, humidity_col, pressure_col = (
                        st.columns(3)
                    )

                    with temp_col:

                        st.caption("Temperature")

                        temp = latest.get(
                            "temperature_c"
                        )

                        if pd.notna(temp):

                            st.markdown(
                                f"### {temp:.2f} °C"
                            )

                        else:

                            st.markdown("### —")

                    with humidity_col:

                        st.caption("Humidity")

                        humidity = latest.get(
                            "humidity_pct"
                        )

                        if pd.notna(humidity):

                            st.markdown(
                                f"### {humidity:.2f} %"
                            )

                        else:

                            st.markdown("### —")

                    with pressure_col:

                        st.caption("Pressure")

                        pressure = latest.get(
                            "pressure_hpa"
                        )

                        if pd.notna(pressure):

                            st.markdown(
                                f"### {pressure:.2f} hPa"
                            )

                        else:

                            st.markdown("### —")

                    st.divider()

                    # -------------------------------------------------
                    # Batch information
                    # -------------------------------------------------

                    info1, info2, info3 = (
                        st.columns(3)
                    )

                    with info1:

                        st.caption(
                            "Anomalies in Batch"
                        )

                        st.write(
                            f"**{batch['anomaly_count']}**"
                        )

                    with info2:

                        st.caption(
                            "Detection Result"
                        )

                        if batch["anomaly_count"] > 0:

                            st.write(
                                "**Anomaly detected**"
                            )

                        else:

                            st.write(
                                "**No anomaly**"
                            )

                    with info3:

                        st.caption(
                            "Adaptive Model"
                        )

                        if batch["model_refitted"]:

                            st.write(
                                "**Model retrained**"
                            )

                        else:

                            st.write(
                                "**No retraining**"
                            )

                    # =================================================
                    # DETECTION DETAILS
                    # =================================================

                    if batch["anomalies"]:

                        with st.expander(
                            "Detection Details"
                        ):

                            for number, anomaly in enumerate(
                                batch["anomalies"],
                                start=1
                            ):

                                st.markdown(
                                    f"#### Anomaly {number}"
                                )

                                detail1, detail2 = (
                                    st.columns(2)
                                )

                                with detail1:

                                    st.write(
                                        f"**Timestamp:** "
                                        f"{anomaly.get('timestamp', '—')}"
                                    )

                                    # ART3 provides the complete trigger list.
                                    triggers = anomaly.get("triggers", [])

                                    if triggers:
                                        detector_text = " + ".join(triggers)
                                    else:
                                        detector_text = anomaly.get(
                                            "anomaly_type", "Unknown"
                                        )

                                    st.write(
                                        f"**Triggered By:** "
                                        f"{detector_text}"
                                    )

                                with detail2:

                                    score = anomaly.get(
                                        "iforest_score"
                                    )

                                    zscore = anomaly.get(
                                        "zscore_max"
                                    )

                                    if isinstance(
                                        score,
                                        (int, float)
                                    ):

                                        st.write(
                                            f"**Isolation Forest "
                                            f"Score:** "
                                            f"{score:.4f}"
                                        )

                                    else:

                                        st.write(
                                            "**Isolation Forest "
                                            "Score:** —"
                                        )

                                    if isinstance(
                                        zscore,
                                        (int, float)
                                    ):

                                        st.write(
                                            f"**Maximum Z-Score:** "
                                            f"{zscore:.4f}"
                                        )

                                    else:

                                        st.write(
                                            "**Maximum Z-Score:** —"
                                        )

                                if (
                                    number
                                    < len(
                                        batch["anomalies"]
                                    )
                                ):

                                    st.divider()

            # -------------------------------------------------
            # Feed navigation hint
            # -------------------------------------------------

            if len(batches) > st.session_state.visible_blocks:

                st.markdown(
                    "<div style='text-align:center; "
                    "color:#888; padding:0.2rem 0;'>"
                    "More processed batches are available"
                    "</div>",
                    unsafe_allow_html=True
                )

        # -----------------------------------------------------
        # Feed navigation
        # -----------------------------------------------------
        # By default only the latest/current batch is visible.
        # Show More Batches reveals older batches.
        # Hide Batches returns the feed to the latest batch only.

        has_older_batches = (
            len(batches) > st.session_state.visible_blocks
        )

        if st.session_state.visible_blocks > 1:

            nav_col1, nav_col2 = st.columns(2)

            with nav_col1:

                if has_older_batches:

                    if st.button(
                        "⌄  Show More Batches",
                        use_container_width=True
                    ):

                        st.session_state.visible_blocks += 3
                        st.rerun()

            with nav_col2:

                if st.button(
                    "⌃  Hide Batches",
                    use_container_width=True
                ):

                    st.session_state.visible_blocks = 1
                    st.rerun()

        elif has_older_batches:

            if st.button(
                "⌄  Show More Batches",
                use_container_width=True
            ):

                st.session_state.visible_blocks += 3
                st.rerun()

        else:

            st.caption(
                "All processed batches are visible."
            )


# =========================================================
# SYSTEM INTELLIGENCE
# =========================================================

st.markdown(
    '<div class="section-title">'
    'System Intelligence'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# BOTTOM TWO PANELS
# =========================================================

left_panel, right_panel = st.columns(2)


# =========================================================
# LAST ANOMALY
# =========================================================

with left_panel:

    with st.container(border=True):

        st.subheader(
            "Last Anomaly"
        )

        # -------------------------------------------------
        # No anomaly yet
        # -------------------------------------------------

        if (
            processor is None
            or processor.last_anomaly is None
        ):

            st.info(
                "No anomaly has been detected yet."
            )

        # -------------------------------------------------
        # Anomaly exists
        # -------------------------------------------------

        else:

            anomaly = (
                processor.last_anomaly
            )

            st.error(
                "ANOMALY DETECTED"
            )

            st.write(
                f"**Timestamp:** "
                f"{anomaly.get('timestamp', '—')}"
            )

            # -------------------------------------------------
            # ART3 detector triggers
            # -------------------------------------------------

            triggers = anomaly.get("triggers", [])

            if triggers:
                detector_text = " + ".join(triggers)
            else:
                detector_text = anomaly.get(
                    "anomaly_type", "Unknown"
                )

            st.write(
                f"**Triggered By:** "
                f"{detector_text}"
            )

            # -------------------------------------------------
            # Scores
            # -------------------------------------------------

            score = anomaly.get(
                "iforest_score"
            )

            zscore = anomaly.get(
                "zscore_max"
            )

            if isinstance(
                score,
                (int, float)
            ):

                st.write(
                    f"**Isolation Forest Score:** "
                    f"{score:.4f}"
                )

            else:

                st.write(
                    "**Isolation Forest Score:** —"
                )

            if isinstance(
                zscore,
                (int, float)
            ):

                st.write(
                    f"**Maximum Z-Score:** "
                    f"{zscore:.4f}"
                )

            else:

                st.write(
                    "**Maximum Z-Score:** —"
                )

            # -------------------------------------------------
            # Get original sensor values
            # -------------------------------------------------

            try:

                anomaly_timestamp = (
                    pd.to_datetime(
                        anomaly.get(
                            "timestamp"
                        )
                    )
                )

                matching_rows = df[
                    df["timestamp"]
                    == anomaly_timestamp
                ]

                if not matching_rows.empty:

                    anomaly_row = (
                        matching_rows.iloc[-1]
                    )

                    st.divider()

                    temp_col, humidity_col, pressure_col = (
                        st.columns(3)
                    )

                    with temp_col:

                        st.caption(
                            "Temperature"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['temperature_c']:.2f}"
                            f" °C**"
                        )

                    with humidity_col:

                        st.caption(
                            "Humidity"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['humidity_pct']:.2f}"
                            f" %**"
                        )

                    with pressure_col:

                        st.caption(
                            "Pressure"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['pressure_hpa']:.2f}"
                            f" hPa**"
                        )

            except Exception:

                pass


# =========================================================
# SYSTEM / MODEL INTELLIGENCE
# =========================================================

with right_panel:

    with st.container(border=True):

        st.subheader(
            "System / Model Intelligence"
        )

        if processor is None:

            st.info(
                "The model is not initialized yet."
            )

        else:

            processed, total, remaining = (
                processor.progress()
            )

            # -------------------------------------------------
            # Count anomalies
            # -------------------------------------------------

            anomaly_count = sum(
                result.get(
                    "is_anomaly",
                    False
                )

                for result in processor.results
            )

            # -------------------------------------------------
            # Anomaly rate
            # -------------------------------------------------

            anomaly_rate = (

                (anomaly_count / processed)
                * 100

                if processed > 0

                else 0
            )

            # -------------------------------------------------
            # Model
            # -------------------------------------------------

            st.write(
                "**Model:** "
                "ART3 / AdaptiveRTModel"
            )

            st.write(
                "**Isolation Forest:** ACTIVE"
            )

            st.write(
                "**Rolling Z-Score:** ACTIVE"
            )

            st.write(
                "**ART3 Rule Detectors:** ACTIVE"
            )

            st.divider()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            stat1, stat2 = (
                st.columns(2)
            )

            with stat1:

                st.caption(
                    "Readings Processed"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{processed}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat2:

                st.caption(
                    "Anomalies Detected"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{anomaly_count}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            stat3, stat4 = (
                st.columns(2)
            )

            with stat3:

                st.caption(
                    "Anomaly Rate"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{anomaly_rate:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat4:

                st.caption(
                    "Current Batch"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{processor.batch_number}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.divider()

            # -------------------------------------------------
            # Model details
            # -------------------------------------------------

            st.write(
                f"**Baseline Size:** "
                f"{processor.baseline_size}"
            )

            st.write(
                f"**Readings Since Refit:** "
                f"{processor.adaptive.count_since_refit}"
            )

            st.write(
                f"**Remaining:** "
                f"{remaining}"
            )

            # -------------------------------------------------
            # Latest model update
            # -------------------------------------------------

            if processor.last_batch is not None:

                if processor.last_batch[
                    "model_refitted"
                ]:

                    st.success(
                        "Adaptive model was retrained "
                        "during the latest batch."
                    )

                else:

                    st.caption(
                        "No model retraining occurred "
                        "during the latest batch."
                    )

# =========================================================
# LIVE STREAM LOOP
# =========================================================
# One simulated telemetry record is processed per cycle.
# Streamlit reruns the script after each record, preserving
# all detector state in st.session_state.

if st.session_state.is_streaming:

    if processor is None:
        st.session_state.is_streaming = False
        st.warning("Apply Settings before starting the live stream.")
        st.stop()

    if processor.finished():
        st.session_state.is_streaming = False
        st.success("Live telemetry stream completed.")
        st.rerun()

    time.sleep(st.session_state.stream_speed)
    processor.process_next_row()
    st.rerun()
