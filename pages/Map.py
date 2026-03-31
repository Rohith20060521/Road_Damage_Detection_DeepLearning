import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_modern_card, render_top_nav
import os

st.set_page_config(page_title="Strategic Map | SRIMS", page_icon="🗺️", layout="wide")

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Maps")

render_sidebar_alerts()

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-earth-americas" style="color: var(--primary-blue);"></i> Geospatial Intelligence Hub</h1>
    <p style="color: var(--text-slate-600);">Real-time spatial visualization of infrastructure health and hazard clusters.</p>
</div>
""", unsafe_allow_html=True)

# Load Data
try:
    df = pd.read_csv("gps_log.csv")
except:
    st.warning("No GPS data found. Run a detection mission to populate the spatial hub!")
    st.stop()

if df.empty:
    st.warning("The spatial database is currently empty. Initiate a survey to see data.")
    st.stop()

# --- SIDEBAR: ADVANCED INTELLIGENCE FILTERS ---
st.sidebar.markdown("### <i class='fa-solid fa-filter' style='color: var(--primary-blue);'></i> Tactical Filters", unsafe_allow_html=True)

selected_types = st.sidebar.multiselect(
    "Structural Classification",
    options=df['damage_type'].unique(),
    default=df['damage_type'].unique()
)

selected_status = st.sidebar.multiselect(
    "Maintenance Status",
    options=df['status'].unique(),
    default=df['status'].unique()
)

min_conf = st.sidebar.select_slider(
    "Neural Trust Threshold",
    options=[round(i*0.1, 1) for i in range(11)],
    value=0.0
)

# Apply Filters
filtered_df = df[
    (df['damage_type'].isin(selected_types)) &
    (df['status'].isin(selected_status)) &
    (df['confidence'] >= min_conf)
]

# --- EXECUTIVE SUMMARY METRICS ---
m1, m2, m3, m4 = st.columns(4)

pending_count = len(filtered_df[filtered_df['status'] == 'Pending'])
high_severity = len(filtered_df[filtered_df['severity'] == 'high'])
most_common = filtered_df['damage_type'].mode()[0] if not filtered_df.empty else "N/A"
avg_conf = filtered_df['confidence'].mean() if not filtered_df.empty else 0

with m1:
    render_modern_card("Pending Hazards", f"{pending_count}", "fa-clock-rotate-left")
with m2:
    render_modern_card("Critical Risks", f"{high_severity}", "fa-radiation")
with m3:
    render_modern_card("Primary Mode", f"{most_common}", "fa-road-circle-exclamation")
with m4:
    render_modern_card("Mean Precision", f"{avg_conf:.1%}", "fa-bullseye")

# --- GEOSPATIAL ENGINE ---
st.markdown("<br>", unsafe_allow_html=True)
if not filtered_df.empty:
    center_lat = filtered_df['lat'].mean()
    center_lon = filtered_df['lon'].mean()
    
    # Using Positron for the Subtle Light Theme
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
    
    # Layer Groups
    marker_layer = folium.FeatureGroup(name="Detections").add_to(m)
    heat_layer = folium.FeatureGroup(name="Heatmap", show=False).add_to(m)

    # 🔹 Markers
    for _, row in filtered_df.iterrows():
        if row['status'] == 'Resolved':
            color = "#10B981" # Emerald
        elif row['status'] == 'In Progress':
            color = "#3B82F6" # Blue
        else:
            color = "#EF4444" if row['severity'] == 'high' else "#F59E0B"
        
        popup_html = f"""
        <div style='font-family: "Inter", sans-serif; min-width: 200px; color: #1E293B;'>
            <h4 style='margin:0; color:#0F172A;'>{row['damage_type']}</h4>
            <p style='color:#64748B; font-size:10px;'>TASK: {row['task_id']}</p>
            <hr style='margin:8px 0; border: 0; border-top: 1px solid #E2E8F0;'>
            <p><b>Status:</b> <span style='color:{color}; text-transform:uppercase;'>{row['status']}</span></p>
            <p><b>Confidence:</b> {row['confidence']:.2%}</p>
            <p><b>ID:</b> {row['filename']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(marker_layer)

    # 🔹 Heatmap
    heat_data = filtered_df[['lat', 'lon']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10).add_to(heat_layer)
    
    folium.LayerControl().add_to(m)
    
    # Render with Streamlit
    st_folium(m, width='stretch', height=600)
else:
    st.info("No data points match current tactical filters.")

# --- DATA EXPLORER ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("<i class='fa-solid fa-table-list'></i> Spatial Registry", expanded=False):
    st.dataframe(
        filtered_df,
        width='stretch',
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1),
            "media_type": "Ingest Source"
        }
    )