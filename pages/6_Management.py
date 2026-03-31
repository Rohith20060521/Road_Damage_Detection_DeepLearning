import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_modern_card, render_top_nav

st.set_page_config(page_title="Ops Control | SRIMS", page_icon="🏢", layout="wide")

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Management")

# Sidebar
render_sidebar_alerts()

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-building-user"></i> Infrastructure Management Portal</h1>
    <p style="color: #8892B0;">Command and control for city-wide maintenance operations and field mobilization.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists("gps_log.csv"):
    st.warning("No geospatial log data found. Please run a detection mission first.")
    st.stop()

def load_data():
    return pd.read_csv("gps_log.csv")

def save_data(df):
    df.to_csv("gps_log.csv", index=False)

df = load_data()

# --- OPERATIONAL VIEWS ---
tab1, tab2 = st.tabs(["📊 Executive Command", "🛠️ Field Ops Worklist"])

with tab1:
    st.markdown("### <i class='fa-solid fa-tower-observation'></i> Sector Health Metrics", unsafe_allow_html=True)
    active_df = df[df['status'] != 'Resolved']
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        render_modern_card("Total Hazards Detected", len(df), "fa-database")
    with col_m2:
        render_modern_card("Critical Structural Risks", len(active_df[active_df['severity'] == 'high']), "fa-radiation")
    with col_m3:
        render_modern_card("Operational Resolutions", len(df[df['status'] == 'Resolved']), "fa-check-double")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### <i class='fa-solid fa-magnifying-glass-location'></i> Tactical Distribution", unsafe_allow_html=True)
    st.dataframe(
        active_df[['location_label', 'damage_type', 'severity', 'confidence', 'status']], 
        width='stretch',
        column_config={
            "confidence": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1),
            "severity": st.column_config.TextColumn("Risk Level")
        }
    )

with tab2:
    st.markdown("### <i class='fa-solid fa-screwdriver-wrench'></i> Field Engineer Worklist", unsafe_allow_html=True)
    
    # Priority Sorting
    sev_map = {'high': 1, 'medium': 2, 'low': 3}
    worklist = df[df['status'] != 'Resolved'].copy()
    worklist['prio_val'] = worklist['severity'].map(sev_map)
    worklist = worklist.sort_values(by=['prio_val', 'confidence'], ascending=[True, False])

    if worklist.empty:
        st.success("🎉 ALL CLEAR: City infrastructure health is at 100%. No pending hazards.")
    else:
        for idx, row in worklist.iterrows():
            with st.expander(f"⚠️ {row['severity'].upper()} | {row['damage_type']} at {row['location_label']} (ID: {row['task_id']})"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div style="font-family: 'Outfit', sans-serif;">
                        <p><i class="fa-solid fa-location-crosshairs"></i> <b>GPS:</b> {row['lat']:.4f}, {row['lon']:.4f}</p>
                        <p><i class="fa-regular fa-file-video"></i> <b>Evidence:</b> {row['filename']}</p>
                        <p><i class="fa-solid fa-signal"></i> <b>Confidence:</b> {row['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    if st.button(f"RESOLVE TASK", key=f"fin_{row['task_id']}", width='stretch'):
                        idx_orig = df[df['task_id'] == row['task_id']].index
                        df.at[idx_orig[0], 'status'] = "Resolved"
                        if 'prio_val' in df.columns:
                            save_data(df.drop(columns=['prio_val']))
                        else:
                            save_data(df)
                        st.toast(f"✅ Hazard {row['task_id']} successfully resolved.")
                        st.rerun()
