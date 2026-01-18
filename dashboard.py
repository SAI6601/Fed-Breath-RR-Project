import streamlit as st
import pandas as pd
import time
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fed-Breath Command Center", layout="wide")

st.title("🫁 Fed-Breath: Federated Learning Dashboard")
st.markdown("### Real-time Respiratory Rate Estimation Monitor")

# --- LAYOUT ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Placeholders for live updates
with col1:
    st.markdown("#### 📉 Global Model Accuracy (Lower is Better)")
    chart_mae = st.line_chart([])

with col2:
    st.markdown("#### 🏥 Hospital Data Quality (RQI Score)")
    chart_rqi = st.bar_chart([])

metric_placeholder = st.empty()

LOG_FILE = "gui_data.csv"

# --- LIVE UPDATE LOOP ---
st.markdown("---")
st.caption("Waiting for simulation to start...")

last_update = 0

while True:
    if os.path.exists(LOG_FILE):
        try:
            # Load data
            df = pd.read_csv(LOG_FILE)
            
            if not df.empty:
                # 1. Update Metrics
                latest = df.iloc[-1]
                with metric_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Round", int(latest["Round"]))
                    m2.metric("Validation MAE", f"{latest['MAE']:.2f} BrPM", delta="-Improving")
                    m3.metric("Active Clients", 2)

                # 2. Update Charts
                # MAE Chart
                chart_mae.line_chart(df.set_index("Round")["MAE"])
                
                # RQI Bar Chart (Last Round)
                rqi_data = pd.DataFrame({
                    "Hospital": ["Client 0", "Client 1"],
                    "Quality Score": [latest["Client_0_RQI"], latest["Client_1_RQI"]]
                })
                chart_rqi.bar_chart(rqi_data.set_index("Hospital"))
                
                # Stop if reached Round 10
                if latest["Round"] == 10:
                    st.success("Simulation Complete!")
                    break
                    
        except Exception as e:
            pass # File might be locked for writing, just skip this frame
            
    time.sleep(2) # Refresh every 2 seconds