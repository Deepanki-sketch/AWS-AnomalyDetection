import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide"
)

df=pd.read_csv("TestAnomalyUnlabled.csv")

st.title("SKYGUARD AI")
st.caption("Anomaly Detection System for Weather Data")
st.divider()

line_c=st.sidebar.selectbox(" Line Charts", ["None", "Temperature", "Humidity", "Atmospheric Pressure"])
if line_c=="None":
    pass
elif line_c=="Temperature":
    st.subheader("Temperature")
    st.line_chart(df["temperature_c"])
elif line_c=="Humidity":
    st.subheader("Humidity")
    st.line_chart(df["humidity_pct"])
if line_c=="Atmospheric Pressure":
    st.subheader("Atmospheric Pressure")
    st.line_chart(df["pressure_hpa"])   
st.sidebar.divider()


st.subheader("System Overview")
col1, col2, col3, col4, col5 = st.columns(5)
"""if (df["anomaly"] == -1).any():
    status = "ALERT"
else:
    status = "NORMAL"

col1.metric("System Status", status)"""

col1.metric("System Status", "Normal")
col2.metric("Total Records", len(df))   
col3.metric("Temperature", f"{df['temperature_c'].iloc[-1]:.1f} °C")
col4.metric("Humidity", f"{df['humidity_pct'].iloc[-1]:.1f} %")
col5.metric("Atmospheric Pressure", f"{df['pressure_hpa'].iloc[-1]:.1f} hPa")

st.divider()
with st.expander("View Raw Dataset"):
    st.dataframe(df, use_container_width=True)