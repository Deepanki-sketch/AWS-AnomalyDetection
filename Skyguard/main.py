import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide"
)

df = pd.read_csv("TestAnomalyUnlabled.csv", parse_dates=["timestamp"])

st.title("SKYGUARD AI")
st.caption("Anomaly Detection System for Weather Data")
st.divider()

line_c = st.sidebar.selectbox(
    "Line Charts",
    ["None", "Temperature", "Humidity", "Atmospheric Pressure"]
)

if line_c == "None":

    pass

elif line_c == "Temperature":

    st.subheader("Temperature")

    st.line_chart(
        df,
        x="timestamp",
        y="temperature_c"
    )

elif line_c == "Humidity":

    st.subheader("Humidity")

    st.line_chart(
        df,
        x="timestamp",
        y="humidity_pct"
    )

elif line_c == "Atmospheric Pressure":

    st.subheader("Atmospheric Pressure")

    st.line_chart(
        df,
        x="timestamp",
        y="pressure_hpa"
    )


st.subheader("System Overview")
col1, col2, col3, col4, col5 = st.columns(5)
#if (df["anomaly"] == -1).any():
 #   status = "ALERT"
#else:
 #   status = "NORMAL"

#col1.metric("System Status", status)"""

col1.metric("System Status", "Data Loaded")
col2.metric("Total Records", len(df))   
col3.metric("Temperature", f"{df['temperature_c'].iloc[-1]:.1f} °C")
col4.metric("Humidity", f"{df['humidity_pct'].iloc[-1]:.1f} %")
col5.metric("Atmospheric Pressure", f"{df['pressure_hpa'].iloc[-1]:.1f} hPa")

st.divider()

st.subheader("Anomaly Analysis")

anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

anomaly_col1.metric(
    "Anomalies Detected",
    "N/A"
)

anomaly_col2.metric(
    "Anomaly Rate",
    "N/A"
)

anomaly_col3.metric(
    "Detection Status",
    "Model Not Connected"
)
st.sidebar.divider()

data_range = st.sidebar.selectbox(
    "Data Range",
    ["All Data", "Last 100 Records", "Last 250 Records", "Last 500 Records"]
)
if data_range == "All Data":
    chart_df = df

elif data_range == "Last 100 Records":
    chart_df = df.tail(100)

elif data_range == "Last 250 Records":
    chart_df = df.tail(250)

elif data_range == "Last 500 Records":
    chart_df = df.tail(500)
    
st.sidebar.divider()

show_dataset = st.sidebar.checkbox( "Show Dataset Information")
if show_dataset:
    st.divider()

    st.subheader("Dataset Information")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric(
           "Missing Values",
        df.isnull().sum().sum()
    )

    info_col2.metric(
        "Number of Columns",
        len(df.columns)
    )

    info_col3.metric(
        "Data Points",
        df.shape[0] * df.shape[1]
    )

    info_col4.metric(
        "Data Period",
        f"{df['timestamp'].min().strftime('%d %b')} - "
        f"{df['timestamp'].max().strftime('%d %b')}"
    )

st.sidebar.divider()

show_raw_dataset = st.sidebar.checkbox("Show Raw Dataset")
if show_raw_dataset:
    st.divider()

    st.subheader("Raw Dataset")
    with st.expander("View Raw Dataset"):
        st.dataframe(df, use_container_width=True)


        
