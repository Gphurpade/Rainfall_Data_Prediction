
import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page layout and style
st.set_page_config(page_title="Spatio-Temporal Predictor", layout="wide")
sns.set_theme(style="whitegrid")

# =====================================================================
# BACKGROUND MODEL ENGINE & SPATIAL DATA GENERATION
# =====================================================================
@st.cache_resource
def train_prediction_engine():
    np.random.seed(42)
    simulated_series = np.random.exponential(scale=15, size=365)
    simulated_series[0:100] = 0
    
    lookback = 5
    X, y = [], []
    for i in range(len(simulated_series) - lookback):
        X.append(simulated_series[i : i + lookback])
        y.append(simulated_series[i + lookback])
        
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model

@st.cache_data
def generate_spatial_map(center_value):
    np.random.seed(int(center_value * 10) % 1000 + 1)
    x = np.linspace(-3, 3, 135)
    y = np.linspace(-3, 3, 129)
    X, Y = np.meshgrid(x, y)
    z = np.exp(-((X)**2 + (Y)**2)) * center_value * 1.5
    noise = np.random.gamma(shape=1.5, scale=center_value/4, size=(129, 135))
    grid = np.clip(z + noise, 0, 150)
    grid[0:30, 0:40] = np.nan    
    grid[0:45, 95:] = np.nan     
    grid[105:, 0:35] = np.nan    
    return grid

model = train_prediction_engine()

# =====================================================================
# CUSTOM HTML & CSS INJECTION 
# =====================================================================
# 1. HTML Header Title Banner
html_header = """
<div style="background-color:#1e3d59; padding:20px; border-radius:10px; margin-bottom:25px;">
    <h1 style="color:white; text-align:center; font-family:Arial, sans-serif; margin:0;">
        Spatio-Temporal Predictor Model for Rainfall Forecasting & Visualization
    </h1>
    <p style="color:#ffc13b; text-align:center; font-family:Arial, sans-serif; font-size:16px; margin:5px 0 0 0;">
        Interactive Meteorological Interface Built with Advanced Data Wrangling & XGBoost
    </p>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)

# 2. Sidebar/Control Panel Split
col_controls, col_display = st.columns([1, 1.2], gap="large")

with col_controls:
    st.subheader("Configure Weather Input Parameters")
    mode = st.radio("Choose Input Method:", ["Preset Scenario Templates", "Manual Real-Time Sliders"])

    input_sequence = [0.0, 0.0, 0.0, 0.0, 0.0]

    if mode == "Preset Scenario Templates":
        scenario_choice = st.selectbox(
            "Select an Atmospheric Condition:",
            ["Persistent Dry Spell (Drought)", "Scattered Light Showers", "Heavy Monsoonal Escalation"]
        )
        if scenario_choice == "Persistent Dry Spell (Drought)":
            input_sequence = [0.0, 0.0, 0.0, 0.0, 0.0]
        elif scenario_choice == "Scattered Light Showers":
            input_sequence = [2.5, 0.0, 4.1, 1.2, 3.0]
        elif scenario_choice == "Heavy Monsoonal Escalation":
            input_sequence = [12.0, 25.4, 40.1, 55.0, 82.3]
    else:
        st.markdown("**Fine-tune timelines manually via grid cell sliders:**")
        # 1. Force every single input variable to be completely flat (1D)
        d1_flat = np.array(d1).ravel()
        d2_flat = np.array(d2).ravel()
        d3_flat = np.array(d3).ravel()
        d4_flat = np.array(d4).ravel()
        d5_flat = np.array(d5).ravel()
        
        # 2. Concatenate them horizontally into one single row
        final_features = np.concatenate([d1_flat, d2_flat, d3_flat, d4_flat, d5_flat]).reshape(1, -1)
        
        # 3. Print the shape to your Streamlit screen for temporary debugging
        st.write(f"DEBUG - Final Features Shape: {final_features.shape}")
        
        # 4. Run the prediction
        prediction_output = max(0.0, float(model.predict(final_features)[0]))

    # 3. Custom HTML Metrics Card Component
    html_metric_card = f"""
    <div style="background-color:#f5f7fa; padding:15px; border-left: 6px solid #1e3d59; border-radius:5px; margin:20px 0;">
        <span style="font-size:14px; text-transform:uppercase; color:#5c6b73; font-weight:bold; font-family:sans-serif;">
            Predicted Next-Day Target Cell Rainfall
        </span>
        <h2 style="margin:5px 0 0 0; color:#1e3d59; font-size:36px; font-family:sans-serif;">
            {prediction_output:.2f} <span style="font-size:20px; color:#5c6b73;">mm</span>
        </h2>
    </div>
    """
    st.markdown(html_metric_card, unsafe_allow_html=True)

    # 4. Dynamic HTML Dynamic Alert Banners
    if prediction_output == 0:
        alert_html = '<div style="background-color:#fff3cd; color:#856404; padding:12px; border-radius:5px; font-family:sans-serif;">☀️ <b>Dry System:</b> Region is clear of rainclouds.</div>'
    elif prediction_output < 25:
        alert_html = '<div style="background-color:#d4edda; color:#155724; padding:12px; border-radius:5px; font-family:sans-serif;">🌦️ <b>Minor System:</b> Localized light patches showing.</div>'
    else:
        alert_html = '<div style="background-color:#f8d7da; color:#721c24; padding:12px; border-radius:5px; font-family:sans-serif;">🚨 <b>Alert:</b> Intense, high-precipitation cloud formation ahead!</div>'
        
    st.markdown(alert_html, unsafe_allow_html=True)

# --- RIGHT COLUMN: VISUALIZATION CANVAS ---
with col_display:
    st.subheader("Live Map Matrix & Trend Tracking")
    tab_map, tab_trend = st.tabs([" Present Spatial Heatmap Grid", "Time-Series Trend Line"])
    
    with tab_map:
        st.markdown("*Visualization of array elements across row/column axes:*")
        current_map_array = generate_spatial_map(prediction_output)
        
        fig_map, ax_map = plt.subplots(figsize=(7, 5.5))
        im = ax_map.imshow(current_map_array, cmap='YlGnBu', origin='lower', aspect='equal', vmin=0, vmax=120)
        ax_map.scatter(70, 60, color='red', marker='X', s=100, label='Target Station Point')
        fig_map.colorbar(im, ax=ax_map, label='Rainfall Precipitation Scale (mm)')
        ax_map.set_title("IMD 0.25° Spatial Grid Array Visualizer", fontsize=12, fontweight='bold', pad=10)
        ax_map.set_xlabel("Grid Columns")
        ax_map.set_ylabel("Grid Rows")
        ax_map.legend(loc='upper right')
        st.pyplot(fig_map)

    with tab_trend:
        fig_trend, ax_trend = plt.subplots(figsize=(7, 5))
        days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Predicted Next Day']
        values = input_sequence + [prediction_output]
        ax_trend.plot(days[:-1], values[:-1], marker='o', color='#1f77b4', label='Input Progression Trace', linewidth=2)
        ax_trend.plot(days[-2:], values[-2:], linestyle='--', color='#e74c3c', linewidth=2)
        ax_trend.scatter(days[-1], values[-1], color='#e74c3c', s=120, zorder=5, label='XGBoost Forecast Output')
        ax_trend.set_ylabel("Rainfall Matrix Value (mm)")
        ax_trend.set_title("Local Station Point Time-Series Trace", fontsize=11, fontweight='bold')
        ax_trend.set_ylim(0, max(values) + 25)
        ax_trend.legend()
        st.pyplot(fig_trend)
