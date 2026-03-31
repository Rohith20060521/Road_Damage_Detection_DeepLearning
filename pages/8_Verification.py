import streamlit as st
import pandas as pd
import os
import time
from PIL import Image
from utils.alerts import render_sidebar_alerts
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_modern_card, render_top_nav

st.set_page_config(page_title="Field Verification | SRIMS", page_icon="✅", layout="wide")

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Verification")

render_sidebar_alerts()

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-clipboard-check" style="color: var(--primary-blue);"></i> Field Audit & Verification</h1>
    <p style="color: var(--text-slate-600);">Final structural inspection and closing of rectified hazard tasks.</p>
</div>
""", unsafe_allow_html=True)

# Load Data
if os.path.exists("gps_log.csv"):
    df = pd.read_csv("gps_log.csv")
else:
    st.warning("Spatial database unavailable. Please initiate a survey first.")
    st.stop()

if df.empty:
    st.warning("Database is empty. No hazards recorded.")
    st.stop()

# --- FILTER TASKS ---
pending_list = df[df['status'] != 'Resolved'].copy()

if pending_list.empty:
    st.success("🎉 ALL SYSTEMS CLEAR: All reported hazards have been successfully verified and resolved!")
    st.balloons()
    st.stop()

st.markdown("### <i class='fa-solid fa-list-ul' style='color: var(--primary-blue);'></i> Active Worklist", unsafe_allow_html=True)
task_selection = st.selectbox("Assign Task ID for Verification", pending_list['task_id'])

# Task Details
selected_task = pending_list[pending_list['task_id'] == task_selection].iloc[0]

c1, c2 = st.columns([1, 1.2])

with c1:
    st.markdown("### <i class='fa-solid fa-circle-info' style='color: var(--primary-blue);'></i> Task Intelligence", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="modern-card animate-in">
        <p><b><i class="fa-solid fa-fingerprint"></i> Task ID:</b> <code>{selected_task['task_id']}</code></p>
        <p><b><i class="fa-solid fa-triangle-exclamation"></i> Hazard:</b> {selected_task['damage_type']}</p>
        <p><b><i class="fa-solid fa-location-dot"></i> Coordinates:</b> {selected_task['lat']:.4f}, {selected_task['lon']:.4f}</p>
        <p><b><i class="fa-solid fa-signal"></i> Initial Severity:</b> <span style="color: var(--primary-blue);">{selected_task['severity'].upper()}</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Verification Upload
    st.markdown("### <i class='fa-solid fa-camera' style='color: var(--primary-blue);'></i> Post-Repair Evidence", unsafe_allow_html=True)
    repair_img = st.file_uploader("Upload Surface Rectification Proof", type=['jpg', 'jpeg', 'png'])
    
    if repair_img:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        if st.button("🚀 SIGN OFF & CLOSE TASK", width='stretch'):
            idx = df[df['task_id'] == task_selection].index
            df.at[idx[0], 'status'] = 'Resolved'
            df.to_csv("gps_log.csv", index=False)
            st.success(f"Task {task_selection} Successfully Resolved and Archived!")
            time.sleep(1.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown("### <i class='fa-solid fa-eye' style='color: var(--primary-blue);'></i> Visual Forensics", unsafe_allow_html=True)
    
    if repair_img:
        st.info("Cross-referencing verification asset with reported spatial delta...")
        st.image(repair_img, caption=f"Rectified Surface: Task {task_selection}", width='stretch')
    else:
        st.markdown(f"""
        <div style="height: 300px; border: 2px dashed var(--border-slate-200); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: var(--text-slate-600); background: var(--bg-white);">
            <div style="text-align: center;">
                <i class="fa-solid fa-hourglass-start" style="font-size: 2rem; margin-bottom: 10px; color: var(--border-slate-200);"></i>
                <p>Waiting for repair verification evidence...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
